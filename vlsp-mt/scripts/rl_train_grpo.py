import argparse
import os
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, get_peft_model, LoraConfig
import sacrebleu
from tqdm import tqdm


def build_prompt_en2vi(src):
    """ChatML format for Qwen2.5 - English to Vietnamese translation."""
    return (
        "<|im_start|>system\n"
        "You are a professional medical translator. Translate the following English medical sentence into Vietnamese.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{src}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_prompt_vi2en(src):
    """ChatML format for Qwen2.5 - Vietnamese to English translation."""
    return (
        "<|im_start|>system\n"
        "You are a professional medical translator. Translate the following Vietnamese medical sentence into English.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{src}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def sentence_reward(hyp, ref, alpha=0.5, beta=0.3, gamma=0.2):
    """Compute reward as weighted combination of BLEU, chrF, and chrF++."""
    if not hyp.strip():
        return 0.0
    bleu = sacrebleu.sentence_bleu(hyp, [ref], smooth_method='exp').score / 100.0
    chrf = sacrebleu.sentence_chrf(hyp, [ref]).score / 100.0
    chrf_pp = sacrebleu.sentence_chrf(hyp, [ref], word_order=2).score / 100.0
    return alpha * bleu + beta * chrf + gamma * chrf_pp


def batch_sentence_rewards(hyps, refs, alpha=0.5, beta=0.3, gamma=0.2):
    """Compute rewards for a batch."""
    return [sentence_reward(h, r, alpha, beta, gamma) for h, r in zip(hyps, refs)]


def load_dataset(src_path, tgt_path):
    """Load parallel corpus."""
    src = open(src_path, encoding="utf8").read().splitlines()
    tgt = open(tgt_path, encoding="utf8").read().splitlines()
    assert len(src) == len(tgt), f"Mismatch: {len(src)} vs {len(tgt)}"
    return list(zip(src, tgt))


def get_logprobs_batch_vectorized(model, tokenizer, prompts, gen_texts, device):
    """
    gom lại các sample thành 1 batch và chạy
    tự động skip sample nếu mismatch
    toekn ko hđ theo kiểu công chuỗi -> phát hiện Tokenizer merge -> skip
    
    Returns:
        avg_log_probs: tensor [batch_size] - average log prob per token
        stats: dict with 'skip_rate' and 'avg_gen_len' for debugging
    """
    batch_size = len(prompts)
    
    # Tách kwargs: prompt không cần truncation, full text thì có
    prompt_kwargs = {
        "return_tensors": "pt",
        "padding": True,
        "add_special_tokens": False
    }
    full_kwargs = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
        "max_length": 1024,
        "add_special_tokens": False
    }
    
    # 1. Tokenize PROMPTS (không truncate để prefix check chính xác)
    prompt_encodings = tokenizer(prompts, **prompt_kwargs).to(device)
    prompt_ids = prompt_encodings.input_ids
    prompt_lens = prompt_encodings.attention_mask.sum(dim=1)
    
    # 2. Tokenize FULL TEXT (prompt + generation)
    full_texts = [p + g for p, g in zip(prompts, gen_texts)]
    full_encodings = tokenizer(full_texts, **full_kwargs).to(device)
    input_ids = full_encodings.input_ids
    attention_mask = full_encodings.attention_mask
    
    real_bs = input_ids.size(0)
    
    # 3. SAFETY CHECK: Prefix Matching + Track skipped samples
    num_skipped = 0
    original_prompt_lens = prompt_lens.clone()
    
    for i in range(real_bs):
        p_len = int(prompt_lens[i].item())
        full_len = int(attention_mask[i].sum().item())
        
        # Check if prompt fits in full sequence
        if p_len > full_len:
            prompt_lens[i] = full_len
            num_skipped += 1
        elif not torch.equal(input_ids[i, :p_len], prompt_ids[i, :p_len]):
            # MISMATCH detected: Skip mẫu này
            prompt_lens[i] = full_len
            num_skipped += 1
    
    # 4. Forward Pass
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model(input_ids, attention_mask=attention_mask, return_dict=True)
        logits = outputs.logits.float()
    
    # 5. Log Softmax & Shift
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    # Shift: logits[t] dự đoán input[t+1]
    shift_log_probs = log_probs[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    
    # Gather logprob của token thực tế
    token_log_probs = torch.gather(
        shift_log_probs, 2, shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    
    # 6. Masking (Generation part only)
    seq_len = token_log_probs.shape[1]
    mask_indices = torch.arange(seq_len, device=device).expand(real_bs, seq_len)
    
    # Clamp min=0 để an toàn boundary
    start_mask = (prompt_lens.unsqueeze(1) - 1).clamp(min=0)
    end_mask = (attention_mask.sum(dim=1).unsqueeze(1) - 1).clamp(min=0)
    
    # Tạo mask: >= start và < end
    gen_mask = (mask_indices >= start_mask) & (mask_indices < end_mask)
    
    # 7. Normalize (Average Logprob per token)
    masked_log_probs = token_log_probs * gen_mask.float()
    gen_lens = gen_mask.sum(dim=1).float()
    gen_lens_clamped = torch.clamp(gen_lens, min=1.0)  # Avoid div 0
    
    avg_log_probs = masked_log_probs.sum(dim=1) / gen_lens_clamped
    
    # Stats for debugging
    stats = {
        'skip_rate': num_skipped / real_bs if real_bs > 0 else 0.0,
        'avg_gen_len': gen_lens.mean().item(),
        'num_skipped': num_skipped
    }
    
    return avg_log_probs, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--sft_adapter", required=True)
    p.add_argument("--init_adapter", required=True)
    p.add_argument("--rl_src", required=True)
    p.add_argument("--rl_tgt", required=True)
    p.add_argument("--run_id", required=True)
    p.add_argument("--direction", choices=["en2vi", "vi2en"], required=True)
    p.add_argument("--alpha", type=float, default=0.5, help="BLEU weight in reward")
    p.add_argument("--beta", type=float, default=0.3, help="chrF weight in reward")
    p.add_argument("--gamma", type=float, default=0.2, help="chrF++ weight in reward")
    p.add_argument("--kl_coef", type=float, default=0.02, help="KL penalty coefficient")
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum_steps", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--baseline_decay", type=float, default=0.99)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=500)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer từ SFT adapter (đã có medical tokens nếu đã thêm)
    # Fallback về base model nếu adapter không có tokenizer
    tokenizer_path = args.sft_adapter
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        print(f"Loaded tokenizer from adapter: {tokenizer_path}")
    except Exception:
        print(f"No tokenizer in adapter, loading from base model: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Tokenizer vocab size: {len(tokenizer)}")

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
    # Load SINGLE base model with TWO adapters
    # Saves ~14GB VRAM (for 3B model)
    # ============================================================
    print("Loading base model (single instance)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    
    # Resize embeddings nếu tokenizer có vocab khác base model (medical vocab)
    if len(tokenizer) != base_model.config.vocab_size:
        print(f"Resizing embeddings: {base_model.config.vocab_size} -> {len(tokenizer)}")
        base_model.resize_token_embeddings(len(tokenizer))

    # Load policy adapter (trainable) - this becomes the "default" adapter
    print(f"Loading policy adapter from: {args.init_adapter}")
    model = PeftModel.from_pretrained(
        base_model, 
        args.init_adapter, 
        adapter_name="policy",
        is_trainable=True
    )
    
    # Load reference adapter (frozen) onto the SAME base model
    print(f"Loading reference adapter from: {args.sft_adapter}")
    model.load_adapter(args.sft_adapter, adapter_name="reference")
    
    # Freeze reference adapter
    for name, param in model.named_parameters():
        if "reference" in name:
            param.requires_grad = False
    
    # Set policy as active for training
    model.set_adapter("policy")
    model.train()

    # Get trainable params (only policy adapter)
    trainable_params = [
        p for n, p in model.named_parameters() 
        if "policy" in n and "lora" in n.lower() and p.requires_grad
    ]
    print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")

    # Load data
    data = load_dataset(args.rl_src, args.rl_tgt)
    print(f"Loaded RL data: {len(data)} pairs")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    
    total_steps = max(1, (len(data) // args.batch_size) * args.epochs // args.grad_accum_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    print(f"Total training steps: {total_steps}")

    run_dir = os.path.join("runs", args.run_id)
    os.makedirs(run_dir, exist_ok=True)

    build_prompt = build_prompt_en2vi if args.direction == "en2vi" else build_prompt_vi2en
    baseline = 0.0
    global_step = 0

    # ============================================================
    # Training Loop
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

            # Batch generation với Left Padding
            # add_special_tokens=False vì prompt ChatML đã có tags
            inputs = tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512,
                add_special_tokens=False
            ).to(device)
            
            input_width = inputs.input_ids.shape[1]  # Lưu width để slice chính xác

            # Use policy adapter for generation
            model.set_adapter("policy")
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                gen_outputs = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=0.9,
                    top_k=50,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=1,  # Tránh empty generation
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    repetition_penalty=1.1,
                )

            # Robust Decoding: Slice tensor theo input_width thay vì string split
            gen_tokens = gen_outputs[:, input_width:]
            hyps = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            hyps = [h.strip() for h in hyps]

            # Compute rewards
            rewards = batch_sentence_rewards(hyps, refs, args.alpha, args.beta, args.gamma)
            batch_avg_reward = sum(rewards) / len(rewards)
            baseline = args.baseline_decay * baseline + (1 - args.baseline_decay) * batch_avg_reward
            epoch_rewards.extend(rewards)

            # ============================================================
            # Logprobs với Safety Skip
            # Switch to Right Padding cho masking logic
            # ============================================================
            tokenizer.padding_side = "right"
            
            # Get policy log probs (with gradients)
            model.set_adapter("policy")
            policy_lp, policy_stats = get_logprobs_batch_vectorized(model, tokenizer, prompts, hyps, device)
            
            # Get reference log probs (no gradients)
            model.set_adapter("reference")
            with torch.no_grad():
                ref_lp, _ = get_logprobs_batch_vectorized(model, tokenizer, prompts, hyps, device)
            
            # Switch back to policy và Left Padding cho generation tiếp theo
            model.set_adapter("policy")
            tokenizer.padding_side = "left"

            # ============================================================
            # Loss: REINFORCE + Quadratic KL Penalty (L2 Approx)
            # Ổn định hơn và rẻ hơn exp formulation
            # ============================================================
            
            # Clip rewards và tính advantage
            rewards_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)
            rewards_clipped = torch.clamp(rewards_tensor, 0.0, 1.0)
            advantage = rewards_clipped - baseline
            
            # KL divergence: Quadratic approximation (L2)
            # 0.5 * (log_pi - log_ref)^2 ≈ KL khi policy gần ref
            log_ratio = policy_lp - ref_lp.detach()
            kl = 0.5 * (log_ratio ** 2)
            
            # Loss = - (Advantage * log_pi) + beta * KL
            # Detach advantage để không backprop qua nó
            loss = -(advantage.detach() * policy_lp) + args.kl_coef * kl
            total_loss = loss.mean()
            
            # Scale loss for gradient accumulation
            scaled_loss = total_loss / args.grad_accum_steps
            scaled_loss.backward()
            
            batch_loss = total_loss.item()

            # Gradient step
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_losses.append(batch_loss)  # total_loss đã là mean rồi

                if global_step % args.log_interval == 0:
                    avg_loss = sum(epoch_losses[-10:]) / min(len(epoch_losses), 10)
                    avg_reward = sum(epoch_rewards[-100:]) / max(len(epoch_rewards[-100:]), 1)
                    pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'reward': f'{avg_reward:.4f}',
                        'baseline': f'{baseline:.4f}',
                        'skip%': f'{policy_stats["skip_rate"]*100:.1f}',
                        'gen_len': f'{policy_stats["avg_gen_len"]:.1f}'
                    })

                if global_step % args.save_interval == 0:
                    ckpt_dir = os.path.join(run_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(ckpt_dir)
                    print(f"\nSaved checkpoint to {ckpt_dir}")

            # Clear cache periodically to prevent memory buildup
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()

        # End of epoch
        if epoch_rewards:
            print(f"\nEpoch {epoch+1} avg reward: {sum(epoch_rewards)/len(epoch_rewards):.4f}")

        model.save_pretrained(os.path.join(run_dir, f"epoch-{epoch+1}"))

    # Final save
    model.save_pretrained(os.path.join(run_dir, "final_model"))
    print(f"\nRL training completed!")


if __name__ == "__main__":
    main()
