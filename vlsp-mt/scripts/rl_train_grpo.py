# scripts/rl_train_grpo.py
"""
GRPO (Group Relative Policy Optimization) for Machine Translation
Optimized version with proper gradient handling, KL direction, and memory efficiency.
"""
import argparse, os, torch, random
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

def get_seq_logprob_and_kl(cur_model, sft_model, tokenizer, prompt_ids, gen_ids):
    """
    Compute log probability of generated sequence under current model,
    and KL(cur || sft) - penalize current policy diverging from SFT policy.
    
    Args:
        cur_model: Current policy being trained
        sft_model: Reference SFT policy (frozen)
        tokenizer: Tokenizer
        prompt_ids: Tokenized prompt [1, prompt_len]
        gen_ids: Generated token ids [1, gen_len]
    
    Returns:
        seq_logprob: Sum of log probs for generated tokens under cur_model
        kl: KL(cur || sft) divergence estimate
    """
    device = gen_ids.device
    full_ids = torch.cat([prompt_ids, gen_ids], dim=1)
    attention_mask = torch.ones_like(full_ids)
    prompt_len = prompt_ids.shape[1]
    gen_len = gen_ids.shape[1]

    # Forward pass through current model (with gradients)
    cur_out = cur_model(full_ids, attention_mask=attention_mask, return_dict=True)
    cur_logits = cur_out.logits  # [1, L, V]

    # Forward pass through SFT model (no gradients)
    with torch.no_grad():
        sft_out = sft_model(full_ids, attention_mask=attention_mask, return_dict=True)
        sft_logits = sft_out.logits

    # Shift logits and targets for next-token prediction
    # logits[:, t, :] predicts token at position t+1
    shifted_cur_logits = cur_logits[:, prompt_len-1:-1, :]  # Logits predicting gen tokens
    shifted_sft_logits = sft_logits[:, prompt_len-1:-1, :]
    target_ids = gen_ids  # [1, gen_len]

    # Log probabilities
    cur_log_probs = torch.nn.functional.log_softmax(shifted_cur_logits, dim=-1)
    sft_log_probs = torch.nn.functional.log_softmax(shifted_sft_logits, dim=-1)

    # Sequence log probability: sum of log P(token_t | context) for generated tokens
    token_logprobs = torch.gather(
        cur_log_probs, 2, target_ids.unsqueeze(-1)
    ).squeeze(-1)  # [1, gen_len]
    seq_logprob = token_logprobs.sum(dim=1)  # [1]

    # KL(cur || sft) = sum_x P_cur(x) * [log P_cur(x) - log P_sft(x)]
    # This penalizes the current policy for being different from SFT
    cur_probs = torch.exp(cur_log_probs)
    kl_per_token = torch.sum(cur_probs * (cur_log_probs - sft_log_probs), dim=-1)  # [1, gen_len]
    kl = kl_per_token.mean()  # Average over tokens

    return seq_logprob.squeeze(0), kl


def load_dataset(src_path, tgt_path):
    """Load parallel corpus."""
    src = open(src_path, encoding="utf8").read().splitlines()
    tgt = open(tgt_path, encoding="utf8").read().splitlines()
    assert len(src) == len(tgt), f"Mismatch: {len(src)} vs {len(tgt)}"
    return list(zip(src, tgt))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--sft_adapter", required=True, help="Path to SFT adapter (frozen reference)")
    p.add_argument("--init_adapter", required=True, help="Path to adapter to be trained")
    p.add_argument("--rl_src", required=True)
    p.add_argument("--rl_tgt", required=True)
    p.add_argument("--run_id", required=True)
    p.add_argument("--direction", choices=["en2vi", "vi2en"], required=True)
    p.add_argument("--alpha", type=float, default=0.6, help="BLEU weight in reward")
    p.add_argument("--beta", type=float, default=0.4, help="chrF weight in reward")
    p.add_argument("--kl_coef", type=float, default=0.01, help="KL penalty coefficient")
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--num_samples", type=int, default=1, help="Number of samples per input for GRPO")
    p.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    p.add_argument("--baseline_decay", type=float, default=0.99, help="EMA decay for baseline")
    p.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    p.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps for learning rate")
    p.add_argument("--log_interval", type=int, default=10, help="Log every N batches")
    p.add_argument("--save_interval", type=int, default=500, help="Save checkpoint every N batches")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ============================================================
    # Load SFT Model (Reference Policy - Frozen)
    # ============================================================
    print("Loading SFT reference model...")
    sft_base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    sft_model = PeftModel.from_pretrained(sft_base, args.sft_adapter)
    sft_model.eval()
    for param in sft_model.parameters():
        param.requires_grad = False
    print("SFT model loaded and frozen.")

    # ============================================================
    # Load Current Model (Policy to be trained)
    # ============================================================
    print("Loading trainable model...")
    cur_base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    cur_model = PeftModel.from_pretrained(
        cur_base,
        args.init_adapter,
        is_trainable=True
    )
    cur_model.train()

    # Ensure LoRA params are trainable
    trainable_params = []
    total_params = 0
    for name, param in cur_model.named_parameters():
        total_params += param.numel()
        if "lora" in name.lower():
            param.requires_grad = True
            trainable_params.append(param)

    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")

    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found! Check LoRA adapter.")


    # ============================================================
    # Optimizer with warmup scheduler
    # ============================================================
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    
    # Load data
    data = load_dataset(args.rl_src, args.rl_tgt)
    print(f"Loaded RL data: {len(data)} pairs")

    total_steps = (len(data) * args.epochs) // (args.batch_size * args.grad_accum_steps)
    
    def get_lr(step):
        """Linear warmup then constant."""
        if step < args.warmup_steps:
            return step / args.warmup_steps
        return 1.0

    # Setup output directory
    run_dir = os.path.join("runs", args.run_id)
    os.makedirs(run_dir, exist_ok=True)

    # ============================================================
    # Training Loop
    # ============================================================
    baseline = 0.0  # Moving average baseline for variance reduction
    global_step = 0
    best_avg_reward = -float('inf')

    build_prompt = build_prompt_en2vi if args.direction == "en2vi" else build_prompt_vi2en

    for epoch in range(args.epochs):
        random.shuffle(data)
        epoch_rewards = []
        epoch_losses = []

        pbar = tqdm(range(0, len(data), args.batch_size), desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, i in enumerate(pbar):
            batch = data[i:i + args.batch_size]
            batch_loss = 0.0
            batch_rewards = []
            valid_samples = 0

            optimizer.zero_grad()

            for sample_idx, (src, ref) in enumerate(batch):
                prompt = build_prompt(src)
                prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

                # ============================================================
                # GRPO: Sample multiple hypotheses per input
                # ============================================================
                sample_losses = []
                sample_rewards_list = []

                for _ in range(args.num_samples):
                    # Generate with sampling
                    with torch.no_grad():
                        gen_out = cur_model.generate(
                            prompt_ids,
                            do_sample=True,
                            temperature=args.temperature,
                            top_p=0.9,
                            max_new_tokens=args.max_new_tokens,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )

                    # Extract generated tokens (exclude prompt)
                    gen_ids = gen_out[:, prompt_ids.shape[1]:]

                    if gen_ids.shape[1] == 0:
                        continue

                    # Decode hypothesis
                    hyp = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

                    # Compute reward
                    r = sentence_reward(hyp, ref, alpha=args.alpha, beta=args.beta)
                    sample_rewards_list.append(r)

                    # Compute log prob and KL
                    seq_logprob, kl = get_seq_logprob_and_kl(
                        cur_model, sft_model, tokenizer, prompt_ids, gen_ids
                    )

                    # Advantage = reward - baseline (variance reduction)
                    advantage = r - baseline

                    # REINFORCE loss: -advantage * log_prob + kl_penalty
                    loss = -advantage * seq_logprob + args.kl_coef * kl
                    sample_losses.append(loss)

                if len(sample_losses) == 0:
                    continue

                # Average loss over samples for this input
                avg_loss = torch.stack(sample_losses).mean()
                
                # Scale by gradient accumulation
                scaled_loss = avg_loss / (args.batch_size * args.grad_accum_steps)
                scaled_loss.backward()

                batch_loss += avg_loss.item()
                batch_rewards.extend(sample_rewards_list)
                valid_samples += 1

            if valid_samples == 0:
                continue

            # Update baseline with batch average reward
            if batch_rewards:
                batch_avg_reward = sum(batch_rewards) / len(batch_rewards)
                baseline = args.baseline_decay * baseline + (1 - args.baseline_decay) * batch_avg_reward
                epoch_rewards.extend(batch_rewards)

            # Gradient step every grad_accum_steps
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                # Learning rate warmup
                lr_scale = get_lr(global_step)
                for pg in optimizer.param_groups:
                    pg['lr'] = args.lr * lr_scale

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_losses.append(batch_loss / valid_samples)

                # Logging
                if global_step % args.log_interval == 0:
                    avg_loss = sum(epoch_losses[-args.log_interval:]) / min(len(epoch_losses), args.log_interval)
                    avg_reward = sum(epoch_rewards[-100:]) / max(len(epoch_rewards[-100:]), 1)
                    pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'reward': f'{avg_reward:.4f}',
                        'baseline': f'{baseline:.4f}',
                        'lr': f'{args.lr * lr_scale:.2e}'
                    })

                # Save checkpoint
                if global_step % args.save_interval == 0:
                    ckpt_dir = os.path.join(run_dir, f"checkpoint-{global_step}")
                    cur_model.save_pretrained(ckpt_dir)
                    print(f"\nSaved checkpoint to {ckpt_dir}")

        # End of epoch
        if epoch_rewards:
            epoch_avg_reward = sum(epoch_rewards) / len(epoch_rewards)
            print(f"\nEpoch {epoch+1} completed. Avg reward: {epoch_avg_reward:.4f}")

            # Save best model
            if epoch_avg_reward > best_avg_reward:
                best_avg_reward = epoch_avg_reward
                best_dir = os.path.join(run_dir, "best_model")
                cur_model.save_pretrained(best_dir)
                print(f"New best model saved to {best_dir}")

        # Save epoch checkpoint
        epoch_dir = os.path.join(run_dir, f"epoch-{epoch+1}")
        cur_model.save_pretrained(epoch_dir)
        print(f"Epoch checkpoint saved to {epoch_dir}")

    # Final save
    final_dir = os.path.join(run_dir, "final_model")
    cur_model.save_pretrained(final_dir)
    print(f"\nRL training completed. Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
