"""
Back-translation for data augmentation.
Dịch ngược target -> source để tạo thêm training data.

Usage:
    # Bước 1: Train model vi2en trước
    # Bước 2: Dùng model vi2en để back-translate
    python scripts/back_translate.py \
        --model_name Qwen/Qwen2.5-3B-Instruct \
        --adapter_path runs/qwen_vi2en_v1/lora_vi2en_sft \
        --input data/monolingual/vi.txt \
        --output data/augment/bt.en \
        --direction vi2en \
        --batch_size 16
"""
import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm


def parse_hf_path(path):
    """
    Parse HuggingFace path với subfolder.
    Input: 'user/repo/subfolder/path' hoặc 'local/path'
    Output: (repo_id, subfolder) hoặc (local_path, None)
    """
    # Nếu là local path (check cả absolute và relative)
    if os.path.exists(path):
        return path, None
    
    # Check nếu bắt đầu bằng "runs/" hoặc "./" hoặc "../" → local path
    if path.startswith("runs/") or path.startswith("./") or path.startswith("../"):
        return path, None
    
    # Nếu là HF path với subfolder (có nhiều hơn 2 phần)
    parts = path.split('/')
    if len(parts) > 2:
        repo_id = '/'.join(parts[:2])  # user/repo
        subfolder = '/'.join(parts[2:])  # subfolder/path
        return repo_id, subfolder
    
    # HF path không có subfolder
    return path, None


def build_prompt(src: str, direction: str) -> str:
    """ChatML format for Qwen2.5."""
    if direction == "vi2en":
        return (
            "<|im_start|>system\n"
            "You are a professional medical translator. Translate the following Vietnamese medical sentence into English.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{src}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    else:
        return (
            "<|im_start|>system\n"
            "You are a professional medical translator. Translate the following English medical sentence into Vietnamese.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{src}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

import re

def extract_translation(full_text: str, src_text: str) -> str:
    """
    Trích xuất bản dịch dựa trên src_text và dùng Regex để làm sạch rác hội thoại.
    Phiên bản tối ưu cho Qwen Instruct.
    """
    text = full_text.strip()

    # 1. Cắt bỏ phần Prompt (User input) nếu nó bị lặp lại trong output
    # Tìm vị trí src_text xuất hiện
    idx = text.find(src_text)
    
    # Chỉ cắt nếu src_text nằm ở phần đầu (tránh cắt nhầm nếu nó xuất hiện trong bản dịch)
    if idx != -1 and idx < len(text) * 0.3:
        # Lấy phần text phía sau src_text
        text = text[idx + len(src_text):].strip()

    # 2. Dùng Regex cực mạnh để xóa các câu dẫn dắt (Chat fillers)
    # Bắt các mẫu: "Sure,", "Here is the translation:", "Vietnamese:", "Answer:"...
    # (?i) là flag ignore case
    cleanup_pattern = r'^(sure[,!]?|here is|below is|this is|the translation is)?\s*(the\s*)?(translation|answer|response|meaning|dịch|vietnamese|english|target)\s*[:：\-]\s*'
    
    text = re.sub(
        cleanup_pattern,
        '',
        text,
        flags=re.IGNORECASE | re.MULTILINE
    )

    # 3. Cắt bỏ các token kết thúc (nếu skip_special_tokens=False chưa lọc hết)
    for marker in ["<|im_end|>", "<|endoftext|>", "<|im_start|>", "\nUser:", "\nEnglish:"]:
        if marker in text:
            text = text.split(marker)[0]

    # 4. Chuẩn hóa khoảng trắng (xóa \n thừa)
    return " ".join(text.split())


def is_good_translation(src: str, tgt: str) -> bool:
    """
    Filter để loại bỏ back-translation kém chất lượng.
    Returns True nếu translation ok, False nếu nên bỏ.
    """
    # 1. Quá ngắn
    if len(tgt.split()) < 3:
        return False
    
    # 2. Quá dài so với source (có thể bị lặp)
    len_ratio = len(tgt) / max(len(src), 1)
    if len_ratio > 3.0 or len_ratio < 0.2:
        return False
    
    # 3. Lặp lại source (copy nguyên văn)
    if src.lower().strip() == tgt.lower().strip():
        return False
    
    # 4. Chứa quá nhiều từ giống source (>80% overlap)
    src_words = set(src.lower().split())
    tgt_words = set(tgt.lower().split())
    if len(src_words) > 0:
        overlap = len(src_words & tgt_words) / len(src_words)
        if overlap > 0.8:
            return False
    
    # 5. Chứa các pattern lỗi phổ biến
    bad_patterns = [
        "i cannot", "i can't", "sorry", "as an ai",
        "không thể dịch", "xin lỗi", "tôi không",
    ]
    tgt_lower = tgt.lower()
    for pattern in bad_patterns:
        if pattern in tgt_lower:
            return False
    
    # 6. Toàn số hoặc ký tự đặc biệt
    alpha_ratio = sum(c.isalpha() for c in tgt) / max(len(tgt), 1)
    if alpha_ratio < 0.5:
        return False
    
    return True
    

def main():
    p = argparse.ArgumentParser(description="Back-translation for data augmentation")
    p.add_argument("--model_name", required=True)
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--input", required=True, help="Monolingual input file")
    p.add_argument("--output", required=True, help="Output translations")
    p.add_argument("--direction", choices=["en2vi", "vi2en"], required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7, 
                   help="Sampling temperature (higher = more diverse)")
    p.add_argument("--num_samples", type=int, default=1,
                   help="Number of translations per sentence (for diversity)")
    p.add_argument("--filter", action="store_true",
                   help="Filter out bad translations")
    args = p.parse_args()

    # ============================================================
    # Load tokenizer - thử từ adapter trước, fallback về base model
    # ============================================================
    print(f"Loading tokenizer from adapter: {args.adapter_path}")
    adapter_repo, adapter_subfolder = parse_hf_path(args.adapter_path)
    try:
        if adapter_subfolder:
            tokenizer = AutoTokenizer.from_pretrained(
                adapter_repo, 
                subfolder=adapter_subfolder,
                use_fast=False
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(adapter_repo, use_fast=False, local_files_only=True)
        print(f"Loaded tokenizer from adapter (may include medical vocab)")
    except Exception:
        print(f"Tokenizer not found in adapter, loading from base model: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    attn_impl = None
    if torch.cuda.is_available():
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"

    # Load base model
    print(f"Loading base model: {args.model_name}")
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=dtype,
        attn_implementation=attn_impl,
    )
    
    # ============================================================
    # Đảm bảo base model có cùng vocab size với tokenizer đã train
    # ============================================================
    if len(tokenizer) != base.config.vocab_size:
        print(f"[INFO] Resizing base model embeddings: {base.config.vocab_size} -> {len(tokenizer)}")
        base.resize_token_embeddings(len(tokenizer))
    
    # Load adapter
    print(f"Loading adapter from: {args.adapter_path}")
    adapter_repo, adapter_subfolder = parse_hf_path(args.adapter_path)
    model = PeftModel.from_pretrained(base, adapter_repo, subfolder=adapter_subfolder)
    model.eval()
    device = next(model.parameters()).device

    # Load input
    with open(args.input, encoding="utf8") as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(lines)} sentences")

    # Generate
    all_translations = []
    
    for sample_idx in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\nSample {sample_idx + 1}/{args.num_samples}")
        
        translations = []
        for i in tqdm(range(0, len(lines), args.batch_size), desc="Translating"):
            batch = lines[i:i + args.batch_size]
            prompts = [build_prompt(line, args.direction) for line in batch]
            
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=0.9,
                    repetition_penalty=1.1,  # [FIX] Tránh lặp lại câu nguồn
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )

            gen_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            for full_text, src_text in zip(gen_texts, batch):
                hyp = extract_translation(full_text, src_text)
                
                # Filter nếu bật --filter
                if args.filter and not is_good_translation(src_text, hyp):
                    hyp = ""  # Mark as bad, sẽ filter sau
                
                translations.append((src_text, hyp))
        
        all_translations.append(translations)

    # Save output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    if args.num_samples == 1:
        # Filter out empty translations (bad ones)
        good_pairs = [(src, tgt) for src, tgt in all_translations[0] if tgt.strip()]
        
        # Save translations
        with open(args.output, "w", encoding="utf8") as f:
            for _, tgt in good_pairs:
                f.write(tgt + "\n")
        
        # Save corresponding source (for pairing)
        if args.output.endswith(".en"):
            src_output = args.output[:-3] + ".src.vi"
        elif args.output.endswith(".vi"):
            src_output = args.output[:-3] + ".src.en"
        else:
            src_output = args.output + ".src"
        
        if src_output != args.output:
            with open(src_output, "w", encoding="utf8") as f:
                for src, _ in good_pairs:
                    f.write(src + "\n")
        
        filtered_count = len(all_translations[0]) - len(good_pairs)
        print(f"\nSaved {len(good_pairs)} translations to {args.output}")
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} bad translations ({filtered_count/len(all_translations[0])*100:.1f}%)")
    else:
        # Save multiple samples
        for idx, translations in enumerate(all_translations):
            out_path = args.output.replace(".", f".{idx}.")
            with open(out_path, "w", encoding="utf8") as f:
                for t in translations:
                    f.write(t + "\n")
            print(f"Saved sample {idx} to {out_path}")


if __name__ == "__main__":
    main()
