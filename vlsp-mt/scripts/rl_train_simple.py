import argparse
import os
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sacrebleu
from tqdm import tqdm


def parse_hf_path(path):
    if os.path.exists(path):
        return path, None
    if path.startswith("runs/") or path.startswith("./") or path.startswith("../"):
        return path, None
    parts = path.split('/')
    if len(parts) > 2:
        return '/'.join(parts[:2]), '/'.join(parts[2:])
    return path, None


def build_prompt_en2vi(src):
    return (
        "<|im_start|>system\n"
        "You are a professional medical translator. Translate the following English medical sentence into Vietnamese.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{src}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_prompt_vi2en(src):
    return (
        "<|im_start|>system\n"
        "You are a professional medical translator. Translate the following Vietnamese medical sentence into English.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{src}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def compute_reward(hyp, ref):
    """Simple reward: chrF++"""
    if not hyp.strip():
        return -0.5
    
    # Length penalty
    len_ratio = len(hyp) / max(len(ref), 1)
    if len_ratio < 0.3 or len_ratio > 3.0:
        return 0.0
    
    chrf = sacrebleu.sentence_chrf(hyp, [ref], word_order=2).score / 100.0
    return chrf


def load_dataset(src_path, tgt_path):
    src = open(src_path, encoding="utf8").read().splitlines()
    tgt = open(tgt_path, encoding="utf8").read().splitlines()
    assert len(src) == len(tgt)
    return list(zip(src, tgt))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--rl_src", required=True)
    p.add_argument("--rl_tgt", required=True)
    p.add_argument("--run_id", required=True)
    p.add_argument("--direction", choices=["en2vi", "vi2en"], required=True)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum_steps", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--save_interval", type=int, default=200)
    args = p.parse_args()

    device = "cuda"
    
    # Load tokenizer
    repo, subfolder = parse_hf_path(args.adapter_path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo, subfolder=subfolder, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model - single adapter only
    print("Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    
    if len(tokenizer) != base_model.config.vocab_size:
        base_model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(base_model, repo, subfolder=subfolder, is_trainable=True)
    model.train()
    
    # Enable gradient checkpointing with proper config
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
    # Make sure LoRA params require grad
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
    
    trainable_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora" in n.lower()]
    print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")

    # Data & optimizer
    data = load_dataset(args.rl_src, args.rl_tgt)
    print(f"Loaded {len(data)} pairs")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    
    run_dir = os.path.join("runs", args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    build_prompt = build_prompt_en2vi if args.direction == "en2vi" else build_prompt_vi2en
    baseline = 0.5
    global_step = 0

    # Training loop
    for epoch in range(args.epochs):
        random.shuffle(data)
        epoch_rewards = []
        
        pbar = tqdm(range(0, len(data), args.batch_size), desc=f"Epoch {epoch+1}")
        
        for batch_idx, i in enumerate(pbar):
            batch = data[i:i + args.batch_size]
            srcs = [s for s, _ in batch]
            refs = [r for _, r in batch]
            prompts = [build_prompt(s) for s in srcs]
            
            # Generate
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, 
                             truncation=True, max_length=384, add_special_tokens=False).to(device)
            input_len = inputs.input_ids.shape[1]
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=0.9,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            hyps = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
            hyps = [h.strip() for h in hyps]
            
            # Compute rewards
            rewards = [compute_reward(h, r) for h, r in zip(hyps, refs)]
            avg_reward = sum(rewards) / len(rewards)
            baseline = 0.9 * baseline + 0.1 * avg_reward
            epoch_rewards.extend(rewards)
            
            # Forward pass for log probs
            full_texts = [p + h for p, h in zip(prompts, hyps)]
            tokenizer.padding_side = "right"
            enc = tokenizer(full_texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=512, add_special_tokens=False).to(device)
            tokenizer.padding_side = "left"
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(enc.input_ids, attention_mask=enc.attention_mask)
                logits = out.logits
            
            # Compute log probs of generated tokens
            log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
            labels = enc.input_ids[:, 1:]
            token_lp = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
            
            # Mask: only generation part
            prompt_enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
            prompt_lens = prompt_enc.attention_mask.sum(dim=1).to(device)
            
            mask = torch.zeros_like(token_lp)
            for j in range(len(prompts)):
                start = int(prompt_lens[j].item()) - 1
                end = int(enc.attention_mask[j].sum().item()) - 1
                if start < end:
                    mask[j, start:end] = 1.0
            
            # REINFORCE loss
            rewards_t = torch.tensor(rewards, device=device)
            advantage = rewards_t - baseline
            
            seq_lp = (token_lp * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            loss = -(advantage * seq_lp).mean()
            
            (loss / args.grad_accum_steps).backward()
            
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                if global_step % 20 == 0:
                    pbar.set_postfix({
                        'reward': f'{sum(epoch_rewards[-50:])/max(len(epoch_rewards[-50:]),1):.3f}',
                        'loss': f'{loss.item():.4f}'
                    })
                
                if global_step % args.save_interval == 0:
                    ckpt = os.path.join(run_dir, f"ckpt-{global_step}")
                    model.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)
                    print(f"\nSaved {ckpt}")
            
            if batch_idx % 30 == 0:
                torch.cuda.empty_cache()
        
        print(f"Epoch {epoch+1} reward: {sum(epoch_rewards)/len(epoch_rewards):.4f}")
        model.save_pretrained(os.path.join(run_dir, f"epoch-{epoch+1}"))
        tokenizer.save_pretrained(os.path.join(run_dir, f"epoch-{epoch+1}"))

    model.save_pretrained(os.path.join(run_dir, "final"))
    tokenizer.save_pretrained(os.path.join(run_dir, "final"))
    print("Done!")


if __name__ == "__main__":
    main()
