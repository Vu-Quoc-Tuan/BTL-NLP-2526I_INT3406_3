# scripts/train_qwen_lora.py
"""
SFT training with LoRA for Qwen model on medical translation task.
Full performance version with NEFTune, label smoothing, early stopping, etc.
"""
import os
import argparse
import json
import torch
import random
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def tokenize_example(example, tokenizer, direction, max_len):
    """
    Tokenize source-target pair with proper label masking.
    Only compute loss on target tokens, not prompt.
    """
    src, tgt = example["src"], example["tgt"]
    prompt = build_prompt_en2vi(src) if direction == "en2vi" else build_prompt_vi2en(src)
    
    # Tokenize prompt (limit to half of max_len to leave room for target)
    max_prompt_len = max_len // 2
    prompt_ids = tokenizer(
        prompt, 
        add_special_tokens=True,
        truncation=True,
        max_length=max_prompt_len
    )["input_ids"]
    
    # Target with space prefix and EOS
    max_target_len = max_len - len(prompt_ids)
    target_ids = tokenizer(
        " " + tgt + tokenizer.eos_token,
        add_special_tokens=False,
        truncation=True,
        max_length=max(1, max_target_len)  # At least 1 token
    )["input_ids"]
    
    # Concatenate and ensure total length <= max_len
    input_ids = prompt_ids + target_ids
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
    
    prompt_len = len(prompt_ids)
    target_len = len(input_ids) - prompt_len
    
    attention_mask = [1] * len(input_ids)
    
    # Labels: -100 for prompt tokens (no loss), actual ids for target
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    
    # Pad to max_len
    pad_len = max_len - len(input_ids)
    if pad_len > 0:
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len
        labels = labels + [-100] * pad_len
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def make_dataset(src_path, tgt_path, subset=None, shuffle_seed=None):
    """Load parallel corpus into HuggingFace Dataset."""
    with open(src_path, encoding="utf8") as f:
        src_lines = [l.strip() for l in f if l.strip()]
    with open(tgt_path, encoding="utf8") as f:
        tgt_lines = [l.strip() for l in f if l.strip()]
    
    assert len(src_lines) == len(tgt_lines), \
        f"Mismatch: {len(src_lines)} src vs {len(tgt_lines)} tgt"
    
    # Shuffle before subset for better distribution
    if shuffle_seed is not None:
        combined = list(zip(src_lines, tgt_lines))
        random.Random(shuffle_seed).shuffle(combined)
        src_lines, tgt_lines = zip(*combined)
        src_lines, tgt_lines = list(src_lines), list(tgt_lines)
    
    if subset:
        src_lines = src_lines[:subset]
        tgt_lines = tgt_lines[:subset]
    
    return Dataset.from_dict({"src": src_lines, "tgt": tgt_lines})



class NEFTuneTrainer(Trainer):
    """
    Trainer with NEFTune: adds noise to embeddings during training.
    Paper: https://arxiv.org/abs/2310.05914
    Improves instruction tuning by ~2-3 BLEU.
    """
    def __init__(self, neftune_noise_alpha=5.0, **kwargs):
        super().__init__(**kwargs)
        self.neftune_noise_alpha = neftune_noise_alpha
        self._neftune_hook_handle = None
    
    def _setup_neftune_hook(self):
        """Setup hook once at the beginning of training."""
        if self._neftune_hook_handle is not None:
            return
        
        # Get embeddings layer
        if hasattr(self.model, 'base_model'):
            embed_layer = self.model.base_model.model.model.embed_tokens
        else:
            embed_layer = self.model.model.embed_tokens
        
        neftune_alpha = self.neftune_noise_alpha
        
        def neftune_hook(module, input, output):
            if module.training:
                # Faster: avoid creating tensor for dims
                seq_len, hidden_dim = output.size(1), output.size(2)
                mag_norm = neftune_alpha / (seq_len * hidden_dim) ** 0.5
                output = output + torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
            return output
        
        self._neftune_hook_handle = embed_layer.register_forward_hook(neftune_hook)
    
    def train(self, *args, **kwargs):
        self._setup_neftune_hook()
        try:
            return super().train(*args, **kwargs)
        finally:
            if self._neftune_hook_handle:
                self._neftune_hook_handle.remove()
                self._neftune_hook_handle = None
    
    def evaluate(self, *args, **kwargs):
        """Override evaluate to clear CUDA cache after validation to prevent OOM."""
        result = super().evaluate(*args, **kwargs)
        # Clear CUDA cache after validation to free memory before resuming training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result


def main():
    p = argparse.ArgumentParser(description="Train Qwen with LoRA for medical translation (Full Performance)")
    
    # Model args
    p.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct",
                   help="HuggingFace model name")
    p.add_argument("--direction", choices=["en2vi", "vi2en"], required=True)
    
    # Data args
    p.add_argument("--src", required=True, help="Source file path")
    p.add_argument("--tgt", required=True, help="Target file path")
    p.add_argument("--val_src", default=None, help="Validation source file")
    p.add_argument("--val_tgt", default=None, help="Validation target file")
    p.add_argument("--subset", type=int, default=None, help="Use subset of data")
    p.add_argument("--max_len", type=int, default=256, help="Max sequence length")
    
    # Output args
    p.add_argument("--run_id", required=True)
    p.add_argument("--out_dir", default="runs")
    
    # LoRA args
    p.add_argument("--lora_r", type=int, default=32, help="LoRA rank (higher=more capacity)")
    p.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # Training args
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    p.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    p.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    p.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    p.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    p.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    
    # Performance args
    p.add_argument("--neftune_alpha", type=float, default=5.0, 
                   help="NEFTune noise alpha (0 to disable)")
    p.add_argument("--label_smoothing", type=float, default=0.1,
                   help="Label smoothing factor")
    p.add_argument("--early_stopping_patience", type=int, default=3,
                   help="Early stopping patience (0 to disable)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--no_grad_checkpoint", action="store_true",
                   help="Disable gradient checkpointing (faster but uses more VRAM)")
    
    args = p.parse_args()
    
    # Set seed
    set_seed(args.seed)

    # Setup directories
    run_dir = os.path.join(args.out_dir, args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    adapter_name = f"lora_{args.direction}_sft"
    adapter_save_path = os.path.join(run_dir, adapter_name)

    # ============================================================
    # Load tokenizer
    # ============================================================
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ============================================================
    # Load model
    # ============================================================
    print(f"Loading model: {args.model_name}")
    
    # Check if bf16 is supported
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Using dtype: {dtype}")

    # Check if flash_attn is available
    attn_impl = None
    if torch.cuda.is_available():
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
            print("Using Flash Attention 2")
        except ImportError:
            attn_impl = "sdpa"  # Use PyTorch's scaled dot product attention
            print("Flash Attention not installed, using SDPA")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=dtype,
        attn_implementation=attn_impl,
    )
    
    # Gradient checkpointing: trade speed for memory
    if not args.no_grad_checkpoint:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        print("Gradient checkpointing ENABLED (slower but less VRAM)")
    else:
        print("Gradient checkpointing DISABLED (faster but more VRAM)")


    # ============================================================
    # Setup LoRA
    # ============================================================
    # Target all linear layers for maximum learning capacity
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"       # MLP
    ]
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # RSLoRA: scales alpha by sqrt(r) for better training dynamics
        use_rslora=True,
    )
    
    model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # ============================================================
    # Prepare datasets
    # ============================================================
    print("Loading training data...")
    train_ds = make_dataset(args.src, args.tgt, args.subset, shuffle_seed=args.seed)
    print(f"Training samples: {len(train_ds)}")
    
    train_tokenized = train_ds.map(
        lambda ex: tokenize_example(ex, tokenizer, args.direction, args.max_len),
        batched=False,
        remove_columns=["src", "tgt"],
        desc="Tokenizing train",
    )
    train_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Validation set
    eval_tokenized = None
    if args.val_src and args.val_tgt:
        print("Loading validation data...")
        val_ds = make_dataset(args.val_src, args.val_tgt)
        print(f"Validation samples: {len(val_ds)}")
        
        eval_tokenized = val_ds.map(
            lambda ex: tokenize_example(ex, tokenizer, args.direction, args.max_len),
            batched=False,
            remove_columns=["src", "tgt"],
            desc="Tokenizing val",
        )
        eval_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # ============================================================
    # Training arguments
    # ============================================================
    effective_batch_size = args.batch_size * args.grad_accum
    steps_per_epoch = len(train_tokenized) // effective_batch_size
    total_steps = steps_per_epoch * args.epochs
    
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")

    training_args = TrainingArguments(
        output_dir=run_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,  # Can use larger for eval
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=args.max_grad_norm,
        
        # Label smoothing
        label_smoothing_factor=args.label_smoothing,
        
        # Precision
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        bf16_full_eval=use_bf16,
        
        # Logging
        logging_steps=max(10, steps_per_epoch // 10),
        logging_first_step=True,
        
        # Saving
        save_strategy="steps",
        save_steps=max(50, steps_per_epoch // 2),
        save_total_limit=3,
        
        # Evaluation
        eval_strategy="steps" if eval_tokenized else "no",
        eval_steps=max(50, steps_per_epoch // 2) if eval_tokenized else None,
        load_best_model_at_end=True if eval_tokenized else False,
        metric_for_best_model="eval_loss" if eval_tokenized else None,
        greater_is_better=False,
        
        # Performance optimizations for A100
        dataloader_num_workers=12,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,
        gradient_checkpointing=not args.no_grad_checkpoint,
        gradient_checkpointing_kwargs={"use_reentrant": False} if not args.no_grad_checkpoint else None,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        torch_compile=False,  # Disabled - can cause slowdown with LoRA + NEFTune
        
        # Other
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )

    # ============================================================
    # Callbacks
    # ============================================================
    callbacks = []
    if args.early_stopping_patience > 0 and eval_tokenized:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        )
        print(f"Early stopping enabled with patience={args.early_stopping_patience}")

    # ============================================================
    # Data collator - simple default collator since we already padded
    # ============================================================
    from transformers import default_data_collator
    data_collator = default_data_collator


    # ============================================================
    # Trainer
    # ============================================================
    TrainerClass = NEFTuneTrainer if args.neftune_alpha > 0 else Trainer
    
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks if callbacks else None,
    )
    
    if args.neftune_alpha > 0:
        print(f"NEFTune enabled with alpha={args.neftune_alpha}")
        trainer = NEFTuneTrainer(neftune_noise_alpha=args.neftune_alpha, **trainer_kwargs)
    else:
        trainer = Trainer(**trainer_kwargs)

    # ============================================================
    # Train
    # ============================================================
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    trainer.train()

    # ============================================================
    # Save final model
    # ============================================================
    print(f"\nSaving adapter to {adapter_save_path}")
    model.save_pretrained(adapter_save_path)
    tokenizer.save_pretrained(adapter_save_path)

    # Save training metadata
    meta = {
        **vars(args),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "target_modules": target_modules,
        "effective_batch_size": effective_batch_size,
        "total_steps": total_steps,
    }
    meta_path = os.path.join(run_dir, "meta.json")
    with open(meta_path, "w", encoding="utf8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    # Save training history
    if trainer.state.log_history:
        history_path = os.path.join(run_dir, "training_history.json")
        with open(history_path, "w", encoding="utf8") as f:
            json.dump(trainer.state.log_history, f, indent=2)
    
    print("\n" + "="*60)
    print(f"Training complete!")
    print(f"Adapter saved to: {adapter_save_path}")
    print(f"Metadata saved to: {meta_path}")
    print("="*60)


if __name__ == "__main__":
    main()
