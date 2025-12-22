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
    EarlyStoppingCallback,
    GenerationConfig
)
from peft import LoraConfig, get_peft_model
import evaluate


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def tokenize_example(example, tokenizer, direction, max_len):
    """
    Tokenize source-target pair with proper label masking.
    Only compute loss on target tokens, not prompt.
    FIX: Tính lại prompt_len sau khi truncate để tránh label bị lệch.
    """
    src, tgt = example["src"], example["tgt"]
    prompt = build_prompt_en2vi(src) if direction == "en2vi" else build_prompt_vi2en(src)
    
    # 1. Tokenize riêng lẻ (không truncate ở đây)
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    target_ids = tokenizer(
        " " + tgt + tokenizer.eos_token, 
        add_special_tokens=False
    )["input_ids"]
    
    # 2. Ghép và truncate nếu cần
    input_ids = prompt_ids + target_ids
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
    
    # 3. Tính lại prompt_len thực tế sau khi cắt (QUAN TRỌNG!)
    # Prompt nằm ở đầu nên thường không bị cắt, trừ khi max_len quá bé
    real_prompt_len = min(len(prompt_ids), len(input_ids))
    
    # 4. Tạo Labels: -100 cho prompt (no loss), giữ nguyên target
    labels = [-100] * real_prompt_len + input_ids[real_prompt_len:]
    
    # Attention mask
    attention_mask = [1] * len(input_ids)
    
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
    """
    def __init__(self, neftune_noise_alpha=5.0, **kwargs):
        super().__init__(**kwargs)
        self.neftune_noise_alpha = neftune_noise_alpha
        self._neftune_hook_handle = None
    
    def _setup_neftune_hook(self):
        """Setup hook once at the beginning of training."""
        if self._neftune_hook_handle is not None:
            return
        
        embed_layer = self.model.get_input_embeddings()
        neftune_alpha = self.neftune_noise_alpha
        
        def neftune_hook(module, input, output):
            if module.training:
                dims = torch.tensor(
                    output.size(1) * output.size(2), 
                    device=output.device,
                    dtype=output.dtype
                )
                mag_norm = neftune_alpha / torch.sqrt(dims)
                noise = torch.zeros_like(output).uniform_(-1, 1) * mag_norm
                output = output + noise
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
        """Override evaluate to clear CUDA cache after validation."""
        result = super().evaluate(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result


class BLEUEvalMixin:
    """Mixin class for BLEU evaluation functionality."""
    
    def _get_tokenizer(self):
        """Get tokenizer compatible with different transformers versions."""
        if hasattr(self, 'processing_class') and self.processing_class:
            return self.processing_class
        return self.tokenizer
    
    def _compute_bleu(self):
        """Generate translations and compute BLEU."""
        self.model.eval()
        tokenizer = self._get_tokenizer()
        
        # Sample subset để tính BLEU (tránh quá lâu)
        n_samples = min(self.bleu_sample_size, len(self.eval_src_texts))
        indices = random.sample(range(len(self.eval_src_texts)), n_samples)
        
        src_samples = [self.eval_src_texts[i] for i in indices]
        tgt_samples = [self.eval_tgt_texts[i] for i in indices]
        
        predictions = []
        build_prompt = build_prompt_en2vi if self.direction == "en2vi" else build_prompt_vi2en
        
        # Generate từng batch nhỏ
        batch_size = 4  # Smaller batch for stability
        for i in range(0, len(src_samples), batch_size):
            batch_src = src_samples[i:i+batch_size]
            batch_prompts = [build_prompt(s) for s in batch_src]
            
            # Tokenize
            inputs = tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=256
            ).to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.gen_max_len,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Decode - chỉ lấy phần generated (bỏ prompt)
            for j, output in enumerate(outputs):
                prompt_len = inputs.input_ids[j].shape[0]
                generated = output[prompt_len:]
                text = tokenizer.decode(generated, skip_special_tokens=True).strip()
                predictions.append(text)
        
        # Compute BLEU
        references = [[ref] for ref in tgt_samples]
        result = self.bleu_metric.compute(predictions=predictions, references=references)
        
        return result["bleu"] * 100


class BLEUEvalTrainer(Trainer, BLEUEvalMixin):
    """Trainer with BLEU evaluation during training."""
    
    def __init__(self, eval_src_texts=None, eval_tgt_texts=None, 
                 direction="en2vi", bleu_metric=None, gen_max_len=128,
                 bleu_sample_size=200, **kwargs):
        super().__init__(**kwargs)
        self.eval_src_texts = list(eval_src_texts) if eval_src_texts else []
        self.eval_tgt_texts = list(eval_tgt_texts) if eval_tgt_texts else []
        self.direction = direction
        self.bleu_metric = bleu_metric
        self.gen_max_len = gen_max_len
        self.bleu_sample_size = bleu_sample_size
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Gọi evaluate gốc của Trainer để lấy loss
        output = Trainer.evaluate(self, eval_dataset, ignore_keys, metric_key_prefix)
        
        # Debug info
        print(f"\n[DEBUG] eval_src_texts len: {len(self.eval_src_texts)}")
        print(f"[DEBUG] eval_tgt_texts len: {len(self.eval_tgt_texts)}")
        print(f"[DEBUG] bleu_metric: {self.bleu_metric is not None}")
        
        # Tính BLEU - LUÔN thêm eval_bleu vào output
        if self.eval_src_texts and self.eval_tgt_texts and self.bleu_metric:
            try:
                bleu_score = self._compute_bleu()
                output[f"{metric_key_prefix}_bleu"] = bleu_score
                print(f"\n>>> BLEU Score: {bleu_score:.2f}")
            except Exception as e:
                print(f"\n[WARNING] BLEU computation failed: {e}")
                import traceback
                traceback.print_exc()
                output[f"{metric_key_prefix}_bleu"] = 0.0
        else:
            # Fallback: thêm eval_bleu = 0 để tránh KeyError
            print(f"\n[WARNING] BLEU skipped - missing data")
            output[f"{metric_key_prefix}_bleu"] = 0.0
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return output


class NEFTuneBLEUTrainer(Trainer, BLEUEvalMixin):
    """Combined NEFTune + BLEU evaluation trainer."""
    
    def __init__(self, neftune_noise_alpha=5.0, eval_src_texts=None, eval_tgt_texts=None,
                 direction="en2vi", bleu_metric=None, gen_max_len=128,
                 bleu_sample_size=200, **kwargs):
        super().__init__(**kwargs)
        # NEFTune params
        self.neftune_noise_alpha = neftune_noise_alpha
        self._neftune_hook_handle = None
        # BLEU params
        self.eval_src_texts = list(eval_src_texts) if eval_src_texts else []
        self.eval_tgt_texts = list(eval_tgt_texts) if eval_tgt_texts else []
        self.direction = direction
        self.bleu_metric = bleu_metric
        self.gen_max_len = gen_max_len
        self.bleu_sample_size = bleu_sample_size
    
    def _setup_neftune_hook(self):
        if self._neftune_hook_handle is not None:
            return
        
        embed_layer = self.model.get_input_embeddings()
        neftune_alpha = self.neftune_noise_alpha
        
        def neftune_hook(module, input, output):
            if module.training:
                dims = torch.tensor(
                    output.size(1) * output.size(2), 
                    device=output.device,
                    dtype=output.dtype
                )
                mag_norm = neftune_alpha / torch.sqrt(dims)
                noise = torch.zeros_like(output).uniform_(-1, 1) * mag_norm
                output = output + noise
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
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Gọi evaluate gốc của Trainer
        output = Trainer.evaluate(self, eval_dataset, ignore_keys, metric_key_prefix)
        
        # Debug info
        print(f"\n[DEBUG] eval_src_texts len: {len(self.eval_src_texts)}")
        print(f"[DEBUG] eval_tgt_texts len: {len(self.eval_tgt_texts)}")
        print(f"[DEBUG] bleu_metric: {self.bleu_metric is not None}")
        
        # Tính BLEU - LUÔN thêm eval_bleu vào output
        if self.eval_src_texts and self.eval_tgt_texts and self.bleu_metric:
            try:
                bleu_score = self._compute_bleu()
                output[f"{metric_key_prefix}_bleu"] = bleu_score
                print(f"\n>>> BLEU Score: {bleu_score:.2f}")
            except Exception as e:
                print(f"\n[WARNING] BLEU computation failed: {e}")
                import traceback
                traceback.print_exc()
                output[f"{metric_key_prefix}_bleu"] = 0.0
        else:
            # Fallback: thêm eval_bleu = 0 để tránh KeyError
            print(f"\n[WARNING] BLEU skipped - missing data")
            output[f"{metric_key_prefix}_bleu"] = 0.0
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return output


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
    p.add_argument("--lr", type=float, default=1.5e-4, help="Learning rate")
    p.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    p.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--epochs", type=int, default=2, help="Number of epochs (2 recommended for NEFTune)")
    p.add_argument("--warmup_ratio", type=float, default=0.005, help="Warmup ratio")
    p.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    p.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    p.add_argument("--eval_steps", type=int, default=1000, 
                   help="Evaluate every N steps (0 = auto: 2x per epoch)")
    
    # Performance args
    p.add_argument("--neftune_alpha", type=float, default=3.0, 
                   help="NEFTune noise alpha (0 to disable)")
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="Label smoothing factor (0 recommended with NEFTune)")
    p.add_argument("--early_stopping_patience", type=int, default=3,
                   help="Early stopping patience (0 to disable)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--no_grad_checkpoint", action="store_true",
                   help="Disable gradient checkpointing (faster but uses more VRAM)")
    
    # BLEU evaluation args
    p.add_argument("--eval_bleu", action="store_true",
                   help="Compute BLEU during evaluation (slower but more accurate)")
    p.add_argument("--bleu_sample_size", type=int, default=200,
                   help="Number of samples to use for BLEU computation")
    
    # Medical vocabulary args
    p.add_argument("--medical_vocab", type=str, default=None,
                   help="Path to medical vocabulary file (one token per line)")
    p.add_argument("--init_new_embeddings_avg", action="store_true",
                   help="Initialize new token embeddings with average of existing (faster convergence)")
    
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
    # [NEW] ADD MEDICAL TOKENS - "Tiêm thuốc" cho tokenizer
    # ============================================================
    num_added_toks = 0
    if args.medical_vocab:
        if os.path.isfile(args.medical_vocab):
            # Load từ file txt (mỗi dòng 1 từ)
            print(f"Loading medical vocabulary from: {args.medical_vocab}")
            with open(args.medical_vocab, encoding="utf8") as f:
                new_tokens = [line.strip() for line in f if line.strip()]
        else:
            print(f"[WARNING] Medical vocab file not found: {args.medical_vocab}")
            new_tokens = []
        
        if new_tokens:
            num_added_toks = tokenizer.add_tokens(new_tokens)
            if num_added_toks > 0:
                print(f"[INFO] Added {num_added_toks} medical tokens to vocabulary.")
                print(f"[INFO] New vocab size: {len(tokenizer)}")
            else:
                print("[INFO] All medical tokens already exist in vocabulary.")

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
    
    # ============================================================
    # [NEW] RESIZE EMBEDDINGS - Quan trọng! Phải làm sau khi load model
    # ============================================================
    if num_added_toks > 0:
        print(f"[INFO] Resizing model embeddings from {model.config.vocab_size} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
        
        # [Mẹo nâng cao] Khởi tạo embedding mới bằng trung bình cộng để hội tụ nhanh hơn
        if args.init_new_embeddings_avg:
            print("[INFO] Initializing new embeddings with average of existing embeddings...")
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data
            input_embeddings_avg = input_embeddings[:-num_added_toks].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_added_toks].mean(dim=0, keepdim=True)
            input_embeddings[-num_added_toks:] = input_embeddings_avg
            output_embeddings[-num_added_toks:] = output_embeddings_avg
    
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
    eval_src_texts = []
    eval_tgt_texts = []
    if args.val_src and args.val_tgt:
        print("Loading validation data...")
        val_ds = make_dataset(args.val_src, args.val_tgt)
        print(f"Validation samples: {len(val_ds)}")
        
        # Lưu raw text cho BLEU evaluation
        eval_src_texts = val_ds["src"]
        eval_tgt_texts = val_ds["tgt"]
        
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

    # Tính eval_steps và save_steps đồng bộ
    # Nếu user truyền --eval_steps=0 hoặc không truyền -> auto (2x per epoch)
    # Nếu user truyền --eval_steps=1000 -> dùng 1000
    if args.eval_steps > 0:
        eval_save_steps = args.eval_steps
    else:
        eval_save_steps = max(50, steps_per_epoch // 2)
    
    print(f"Eval/Save every: {eval_save_steps} steps")

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
        
        # Saving & Evaluation - ĐỒNG BỘ để load_best_model hoạt động đúng
        save_strategy="steps" if eval_tokenized else "steps",
        save_steps=eval_save_steps,
        save_total_limit=2,  # Tiết kiệm ổ cứng: chỉ giữ 2 checkpoint gần nhất
        eval_strategy="steps" if eval_tokenized else "no",
        eval_steps=eval_save_steps if eval_tokenized else None,
        load_best_model_at_end=True if eval_tokenized else False,
        # Dùng BLEU nếu bật --eval_bleu, ngược lại dùng loss
        metric_for_best_model="eval_bleu" if (eval_tokenized and args.eval_bleu) else ("eval_loss" if eval_tokenized else None),
        greater_is_better=True if args.eval_bleu else False,  # BLEU cao = tốt, loss thấp = tốt
        
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
    # Data collator - use DataCollatorForSeq2Seq for dynamic padding
    # This is more efficient than static padding to max_len
    # ============================================================
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,  # Tensor core optimization
        return_tensors="pt"
    )


    # ============================================================
    # Trainer
    # ============================================================
    # Load BLEU metric nếu cần
    bleu_metric = None
    if args.eval_bleu and eval_tokenized:
        print("Loading BLEU metric for evaluation...")
        bleu_metric = evaluate.load("bleu")
    
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks if callbacks else None,
    )
    
    # Chọn Trainer class phù hợp
    use_neftune = args.neftune_alpha > 0
    use_bleu = args.eval_bleu and eval_tokenized
    
    if use_neftune and use_bleu:
        print(f"NEFTune enabled with alpha={args.neftune_alpha}")
        print(f"BLEU evaluation enabled (sample_size={args.bleu_sample_size})")
        trainer = NEFTuneBLEUTrainer(
            neftune_noise_alpha=args.neftune_alpha,
            eval_src_texts=eval_src_texts,
            eval_tgt_texts=eval_tgt_texts,
            direction=args.direction,
            bleu_metric=bleu_metric,
            bleu_sample_size=args.bleu_sample_size,
            **trainer_kwargs
        )
    elif use_bleu:
        print(f"BLEU evaluation enabled (sample_size={args.bleu_sample_size})")
        trainer = BLEUEvalTrainer(
            eval_src_texts=eval_src_texts,
            eval_tgt_texts=eval_tgt_texts,
            direction=args.direction,
            bleu_metric=bleu_metric,
            bleu_sample_size=args.bleu_sample_size,
            **trainer_kwargs
        )
    elif use_neftune:
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
