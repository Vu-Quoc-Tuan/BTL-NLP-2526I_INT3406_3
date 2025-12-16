# scripts/rl_train_grpo.py
"""
GRPO (Group Relative Policy Optimization) for Machine Translation
Optimized with batch generation and shared base model.
"""
import argparse
import os
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sacrebleu
from tqdm import tqdm


def build_prompt_en2vi(src):
    return (
        "You are a professional medical translator.\n"
        "Translate the following English medical sentence into Vietnamese.\n\n"
        f"English: {src}\nVietnamese:"
    )


def build_prompt_vi2en(src):
    return (
        "You are a professional medical translator.\n"
        "Translate the following Vietnamese medical sentence into English.\n\n"
        f"Vietnamese: {src}\nEnglish:"
    )


def sentence_reward(hyp, ref, alpha=0.6, beta=0.4):
    """Compute reward as weighted combination of BLEU and chrF."""
    if not hyp.strip():
        return 0.0
    bleu = sacrebleu.sentence_bleu(hyp, [ref]).score / 100.0
    chrf = sacrebleu.sentence_chrf(hyp, [ref]).score / 100.0
    return alpha * bleu + beta * chrf


def batch_sentence_rewards(hyps, refs, alpha=0.6, beta=0.4):
    """Compute rewards for a batch."""
    return [sentence_reward(h, r, alpha, beta) for h, r in zip(hyps, refs)]


def load_dataset(src_path, tgt_path):
    """Load parallel corpus."""
    src = open(src_path, encoding="utf8").read().splitlines()
    tgt = open(tgt_path, encoding="utf8").read().splitlines()
    assert len(src) == len(tgt), f"Mismatch: {len(src)} vs {len(tgt)}"
    return list(zip(src, tgt))


def get_logprobs_batch(model, tokenizer, prompts, gen_texts, device):
    """
    Compute log probabilities for generated texts given prompts.
    Batched version for efficiency.
    """
    all_logprobs = []
    
    for prompt, gen_text in zip(prompts, gen_texts):
        if not gen_text.strip():
            all_logprobs.append(torch.tensor(0.0, device=device))
            continue
            
        # Tokenize full sequence
        full_text = prompt + " " + gen_text
        full_ids = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        prompt_len = prompt_ids.shape[1]
        
        if full_ids.shape[1] <= prompt_len:
            all_logprobs.append(torch.tensor(0.0, device=device))
            continue
        
        # Forward pass
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(full_ids, return_dict=True)
            logits = outputs.logits
        
        # Get log probs for generated tokens
        shift_logits = logits[:, prompt_len-1:-1, :].float()
        shift_labels = full_ids[:, prompt_len:]
        
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
        seq_log_prob = token_log_probs.sum()
        
        all_logprobs.append(seq_log_prob)
    
    return all_logprobs



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--sft_adapter", required=True)
    p.add_argument("--init_adapter", required=True)
    p.add_argument("--rl_src", required=True)
    p.add_argument("--rl_tgt", required=True)
    p.add_argument("--run_id", required=True)
    p.add_argument("--direction", choices=["en2vi", "vi2en"], required=True)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--beta", type=float, default=0.4)
    p.add_argument("--kl_coef", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum_steps", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--baseline_decay", type=float, default=0.99)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=500)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    tokenizer.padding_side = "left"  # For batch generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check for flash attention
    attn_impl = None
    if torch.cuda.is_available():
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
            print("Using Flash Attention 2")
        except ImportError:
            attn_impl = "sdpa"
            print("Using SDPA")

    # ============================================================
    # Load SINGLE base model, then create 2 adapters
    # This saves ~50% VRAM compared to loading 2 separate models
    # ============================================================
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )

    # Load SFT adapter (frozen reference)
    print("Loading SFT adapter...")
    sft_model = PeftModel.from_pretrained(base_model, args.sft_adapter)
    sft_model.eval()
    for param in sft_model.parameters():
        param.requires_grad = False

    # For cur_model, we need separate base (PEFT limitation)
    print("Loading trainable model...")
    cur_base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    cur_model = PeftModel.from_pretrained(cur_base, args.init_adapter, is_trainable=True)
    cur_model.train()

    # Get trainable params
    trainable_params = [p for n, p in cur_model.named_parameters() if "lora" in n.lower() and p.requires_grad]
    for p in trainable_params:
        p.requires_grad = True
    
    print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # Load data
    data = load_dataset(args.rl_src, args.rl_tgt)
    print(f"Loaded RL data: {len(data)} pairs")

    run_dir = os.path.join("runs", args.run_id)
    os.makedirs(run_dir, exist_ok=True)

    build_prompt = build_prompt_en2vi if args.direction == "en2vi" else build_prompt_vi2en
    baseline = 0.0
    global_step = 0


    # ============================================================
    # Training Loop with BATCH GENERATION
    # ============================================================
    for epoch in range(args.epochs):
        random.shuffle(data)
        epoch_rewards = []
        epoch_losses = []

        pbar = tqdm(range(0, len(data), args.batch_size), desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch_idx, i in enumerate(pbar):
            batch = data[i:i + args.batch_size]
            srcs = [s for s, _ in batch]
            refs = [r for _, r in batch]
            prompts = [build_prompt(s) for s in srcs]

            # ============================================================
            # BATCH GENERATION - Much faster than sequential
            # ============================================================
            inputs = tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=256
            ).to(device)

            with torch.no_grad():
                gen_outputs = cur_model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=0.9,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )

            # Decode all at once
            gen_texts = tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)
            
            # Extract translations (remove prompt)
            hyps = []
            for gen_text, prompt in zip(gen_texts, prompts):
                if prompt in gen_text:
                    hyp = gen_text.split(prompt, 1)[1].strip()
                else:
                    hyp = gen_text.strip()
                hyps.append(hyp)

            # Compute rewards
            rewards = batch_sentence_rewards(hyps, refs, args.alpha, args.beta)
            batch_avg_reward = sum(rewards) / len(rewards)
            
            # Update baseline
            baseline = args.baseline_decay * baseline + (1 - args.baseline_decay) * batch_avg_reward
            epoch_rewards.extend(rewards)

            # ============================================================
            # Compute loss for each sample
            # ============================================================
            optimizer.zero_grad()
            batch_loss = 0.0

            # Get log probs from current model
            cur_logprobs = get_logprobs_batch(cur_model, tokenizer, prompts, hyps, device)
            
            # Get log probs from SFT model (for KL)
            with torch.no_grad():
                sft_logprobs = get_logprobs_batch(sft_model, tokenizer, prompts, hyps, device)

            # Compute REINFORCE loss
            for j, (r, cur_lp, sft_lp) in enumerate(zip(rewards, cur_logprobs, sft_logprobs)):
                advantage = r - baseline
                kl = (cur_lp - sft_lp).clamp(min=0)  # KL >= 0
                loss = -advantage * cur_lp + args.kl_coef * kl
                
                scaled_loss = loss / (args.batch_size * args.grad_accum_steps)
                scaled_loss.backward(retain_graph=(j < len(rewards) - 1))
                batch_loss += loss.item()

            # Gradient step
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_losses.append(batch_loss / len(batch))

                if global_step % args.log_interval == 0:
                    avg_loss = sum(epoch_losses[-10:]) / min(len(epoch_losses), 10)
                    avg_reward = sum(epoch_rewards[-100:]) / max(len(epoch_rewards[-100:]), 1)
                    pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'reward': f'{avg_reward:.4f}',
                        'baseline': f'{baseline:.4f}'
                    })

                if global_step % args.save_interval == 0:
                    ckpt_dir = os.path.join(run_dir, f"checkpoint-{global_step}")
                    cur_model.save_pretrained(ckpt_dir)
                    print(f"\nSaved checkpoint to {ckpt_dir}")

        # End of epoch
        if epoch_rewards:
            print(f"\nEpoch {epoch+1} avg reward: {sum(epoch_rewards)/len(epoch_rewards):.4f}")

        cur_model.save_pretrained(os.path.join(run_dir, f"epoch-{epoch+1}"))

    # Final save
    cur_model.save_pretrained(os.path.join(run_dir, "final_model"))
    print(f"\nRL training completed!")


if __name__ == "__main__":
    main()
