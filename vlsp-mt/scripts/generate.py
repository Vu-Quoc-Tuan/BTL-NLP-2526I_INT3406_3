import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import re
import unicodedata


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


# Ký tự Private Use Area để làm placeholder an toàn cho dấu ba chấm
ELL_PLACEHOLDER = "\uE000"

def _common_clean(text: str) -> str:
    if not text: return ""
    
    # 1. Normalize NFC & Fancy Quotes (QUAN TRỌNG: Làm trước tiên)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("’", "'") # Fix (D): Chuẩn hóa apostrophe trước
    
    # 2. Bảo vệ dấu ba chấm (...)
    text = text.replace("...", ELL_PLACEHOLDER) # Fix (B)
    
    # 3. Xử lý dấu câu lặp (Fix A: Chỉ gom dấu giống nhau)
    # !!! -> !, ??? -> ?, nhưng ?! -> ?!
    # \1+ nghĩa là lặp lại group 1 ít nhất 1 lần nữa
    text = re.sub(r'([.,!?;:])\1+', r'\1', text)
    
    # Trả lại dấu ba chấm
    text = text.replace(ELL_PLACEHOLDER, "...")

    # 4. Chuẩn hóa khoảng trắng trước dấu câu ( "hello !" -> "hello!")
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    
    # 5. Smart Quote Handling (Giữ nguyên logic cũ vì ổn)
    # Chỉ xóa khoảng trắng BÊN TRONG ngoặc: " hello " -> "hello"
    text = re.sub(r'"\s+([^"]+?)\s+"', r'"\1"', text)

    # 6. Dọn dẹp khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def _dedup_leading_repetition(text: str, max_phrase_words: int = 10) -> str:
    """Loại bỏ lặp từ đầu câu (Stuttering error của LLM)"""
    words = text.split()
    if len(words) < 6:
        return text
    
    # Check các cụm từ độ dài từ 3 đến max_phrase_words
    for phrase_len in range(3, min(max_phrase_words, len(words)//3) + 1):
        phrase = words[:phrase_len]
        # Nếu phrase lặp lại ngay sau nó
        if words[phrase_len:2*phrase_len] == phrase:
            k = 2
            # Tìm xem lặp bao nhiêu lần tiếp theo
            while k*phrase_len < len(words) and words[k*phrase_len:(k+1)*phrase_len] == phrase:
                k += 1
            words = phrase + words[k*phrase_len:]
            return " ".join(words)
    return text

def postprocess_vi(text: str) -> str:
    text = _common_clean(text)
    text = _dedup_leading_repetition(text)
    if text:
        text = text[0].upper() + text[1:]
    return text

def postprocess_en(text: str) -> str:
    text = _common_clean(text)
    text = _dedup_leading_repetition(text)

    # 1. Fix Apostrophe Spacing (Fix D: Đã chuẩn hóa ’ thành ' ở _common_clean rồi)
    # Xử lý: "do ' nt", "teacher ' s" -> "do'nt", "teacher's"
    text = re.sub(r"(\w)\s*'\s*(\w)", r"\1'\2", text)

    # 2. Fix Tokenizer Split Errors (Fix E: Quan trọng cho LLM)
    # Tokenizer hay tách: "do", "n't" -> model sinh ra "do n't"
    # Cần flags=re.I để bắt cả "Do n't"
    text = re.sub(r"\bdo\s+n't\b", "don't", text, flags=re.IGNORECASE)
    text = re.sub(r"\bca\s+n't\b", "can't", text, flags=re.IGNORECASE) # can not -> can't
    text = re.sub(r"\bwo\s+n't\b", "won't", text, flags=re.IGNORECASE) # will not -> won't
    text = re.sub(r"\bdoes\s+n't\b", "doesn't", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdid\s+n't\b", "didn't", text, flags=re.IGNORECASE)
    text = re.sub(r"\bshould\s+n't\b", "shouldn't", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcould\s+n't\b", "couldn't", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwould\s+n't\b", "wouldn't", text, flags=re.IGNORECASE)
    text = re.sub(r"\bhave\s+n't\b", "haven't", text, flags=re.IGNORECASE)
    text = re.sub(r"\bhas\s+n't\b", "hasn't", text, flags=re.IGNORECASE)
    text = re.sub(r"\bhad\s+n't\b", "hadn't", text, flags=re.IGNORECASE)

    # 3. Fix các contraction thông thường (nếu bị tách kiểu Dont -> Don't)
    text = re.sub(r"\b([Dd])ont\b", r"\1on't", text)
    text = re.sub(r"\b([Cc])ant\b", r"\1an't", text)
    text = re.sub(r"\b([Ww])ont\b", r"\1on't", text)
    
    # 4. Fix "i" -> "I" (đứng riêng lẻ)
    text = re.sub(r"\bi\b", "I", text)
    # Fix "i'm" -> "I'm" (an toàn hơn, tránh động vào IM y khoa)
    text = re.sub(r"\bi'm\b", "I'm", text)

    # Viết hoa chữ cái đầu
    if text:
        text = text[0].upper() + text[1:]
    return text



# Pattern cho cả 2 format: có và không có special token markers
ASSISTANT_PATTERN = re.compile(
    r"(?:<\|im_start\|>)?assistant\s*(.*?)(?:<\|im_end\|>|$)",
    re.DOTALL | re.IGNORECASE
)

# Pattern đơn giản hơn: tìm "assistant" cuối cùng (không cần special tokens)
SIMPLE_ASSISTANT_PATTERN = re.compile(
    r"assistant\s*\n?\s*(.*)$",
    re.DOTALL | re.IGNORECASE
)

CLEANUP_FILLER = re.compile(
    r'^(?:Sure|Certainly|Of course|Okay|Ok|Yes|No problem)[!.,]?\s*',
    re.IGNORECASE
)

CLEANUP_INTRO = re.compile(
    r'^(?:Here is|Below is|This is|The)?\s*(?:the\s+)?'
    r'(?:translation|meaning|answer|response|result|target|vietnamese|english)'
    r'(?:\s+is)?\s*[:：\.\-–—]\s*',
    re.IGNORECASE
)

STRICT_SEPARATOR_STR = (
    r'(?:'
    r'\s*\n+\s*'
    r'|'
    r'\s+(?:translation|dịch|meaning|answer|target|vietnamese|english)\s*[:：\.\-–—]\s*'
    r'|'
    r'\s*(?:[-=]+>|→)\s*'
    r')'
)

def extract_translation(full_text: str, src_text: str = None, direction: str = "en2vi") -> str:
    """Extract translation from model output, handling both special token formats."""
    
    # 1) Tìm vị trí "assistant" cuối cùng và lấy text sau nó
    # Dùng rfind để lấy assistant CUỐI CÙNG (không phải cái trong prompt)
    assistant_idx = full_text.lower().rfind("assistant")
    if assistant_idx != -1:
        # Lấy phần sau "assistant" + có thể có newline
        hyp = full_text[assistant_idx + len("assistant"):].strip()
        # Bỏ newline đầu nếu có
        hyp = hyp.lstrip('\n').strip()
    else:
        hyp = full_text.strip()
    
    # 2) Cắt tại các stop markers
    stop_markers = ["<|im_end|>", "<|endoftext|>", "<|im_start|>", "\nsystem", "\nuser", "System", "User"]
    if direction == "en2vi":
        stop_markers += ["\nEnglish:", "English:"]
    elif direction == "vi2en":
        stop_markers += ["\nVietnamese:", "Vietnamese:"]

    cut_pos = None
    for marker in stop_markers:
        idx = hyp.find(marker)
        if idx != -1:
            cut_pos = idx if cut_pos is None else min(cut_pos, idx)
    if cut_pos is not None:
        hyp = hyp[:cut_pos]

    # 3) Source repetition (dynamic) - nếu model lặp lại source
    if src_text:
        src_words = src_text.strip().split()
        if src_words and len(src_words) <= 20:  # Chỉ check với source ngắn
            flexible_src_regex = r"\s+".join(re.escape(w) for w in src_words[:10])
            src_repetition_pattern = (
                r'^(?:source|original|text|câu gốc|src)?\s*[:\-]?\s*'
                + flexible_src_regex
                + STRICT_SEPARATOR_STR
            )
            m2 = re.search(src_repetition_pattern, hyp, flags=re.IGNORECASE | re.DOTALL)
            if m2:
                hyp = hyp[m2.end():].strip()

    # 4) Cleanup filler phrases
    for _ in range(3):
        new = CLEANUP_FILLER.sub('', hyp)
        new = CLEANUP_INTRO.sub('', new)
        if new == hyp:
            break
        hyp = new

    return " ".join(hyp.split())


def load_input_file(path: str, max_samples: int = None) -> list:
    """Load input file, optionally limiting samples."""
    with open(path, encoding="utf8") as f:
        lines = [line.strip() for line in f]
    
    if max_samples:
        lines = lines[:max_samples]
    
    return lines



def generate_batch(
    model, 
    tokenizer, 
    prompts: list, 
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    num_beams: int,
    length_penalty: float,
    repetition_penalty: float,
    device,
    direction: str = "en2vi"
) -> list:
    """Generate translations for a batch of prompts."""
    # Tokenize with padding
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    
    # Generate with optimizations
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        if do_sample:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=num_beams,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
    
    # Decode - giữ special tokens để extract_translation hoạt động đúng
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    
    # Extract translations
    translations = []
    for full_text, prompt in zip(generated_texts, prompts):
        hyp = extract_translation(full_text, prompt, direction=direction)
        translations.append(hyp)
    
    return translations


def main():
    p = argparse.ArgumentParser(description="Generate translations with LoRA model")
    
    # Model args
    p.add_argument("--model_name", required=True, help="Base model name")
    p.add_argument("--adapter_path", required=True, help="Path to LoRA adapter")
    p.add_argument("--direction", choices=["en2vi", "vi2en"], required=True)
    
    # I/O args
    p.add_argument("--input", required=True, help="Input file path")
    p.add_argument("--output", required=True, help="Output file path")
    p.add_argument("--max_samples", type=int, default=None, help="Limit samples for testing")
    
    # Generation args
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    p.add_argument("--do_sample", action="store_true", help="Use sampling instead of beam search")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    p.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    p.add_argument("--num_beams", type=int, default=4, help="Beam search beams (1=greedy, 4-5 recommended)")
    p.add_argument("--length_penalty", type=float, default=1.0, help="Length penalty (>1 longer, <1 shorter)")
    p.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty to avoid repeating")
    
    args = p.parse_args()

    # ============================================================
    # Load model
    # ============================================================
    print(f"Loading model: {args.model_name}")
    print(f"Loading adapter: {args.adapter_path}")
    
    # Load tokenizer from adapter path first (may have extended vocab from training)
    # Fallback to base model if adapter doesn't have tokenizer
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
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
        print(f"Loaded tokenizer from base model")
    
    tokenizer.padding_side = "left"  # Important for batch generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Use bfloat16 for efficiency
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    # Check for flash attention
    attn_impl = None
    if torch.cuda.is_available():
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"
    
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        trust_remote_code=True, 
        device_map="auto",
        torch_dtype=dtype,
        attn_implementation=attn_impl,
    )
    
    # Resize embeddings nếu tokenizer có vocab khác base model (medical vocab)
    if len(tokenizer) != base.config.vocab_size:
        print(f"Resizing embeddings: {base.config.vocab_size} -> {len(tokenizer)}")
        base.resize_token_embeddings(len(tokenizer))
    
    adapter_repo, adapter_subfolder = parse_hf_path(args.adapter_path)
    model = PeftModel.from_pretrained(base, adapter_repo, subfolder=adapter_subfolder)
    model.eval()
    
    device = next(model.parameters()).device
    print(f"Model loaded on {device}, dtype={dtype}")

    # ============================================================
    # Load input
    # ============================================================
    lines = load_input_file(args.input, args.max_samples)
    print(f"Loaded {len(lines)} sentences from {args.input}")
    
    build_prompt = build_prompt_en2vi if args.direction == "en2vi" else build_prompt_vi2en
    postprocess = postprocess_vi if args.direction == "en2vi" else postprocess_en

    # ============================================================
    # Generate
    # ============================================================
    all_translations = []
    
    # Process in batches
    for i in tqdm(range(0, len(lines), args.batch_size), desc="Generating"):
        batch_lines = lines[i:i + args.batch_size]
        
        # Handle empty lines
        batch_prompts = []
        batch_indices = []
        for j, line in enumerate(batch_lines):
            if line.strip():
                batch_prompts.append(build_prompt(line))
                batch_indices.append(j)
        
        # Generate for non-empty lines
        if batch_prompts:
            translations = generate_batch(
                model, tokenizer, batch_prompts,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                repetition_penalty=args.repetition_penalty,
                device=device,
                direction=args.direction
            )
        else:
            translations = []
        
        # Reconstruct with empty lines + post-process
        batch_results = [""] * len(batch_lines)
        for idx, trans in zip(batch_indices, translations):
            batch_results[idx] = postprocess(trans)
        

        all_translations.extend(batch_results)

    # ============================================================
    # Save output
    # ============================================================
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf8") as f:
        for hyp in all_translations:
            f.write(hyp + "\n")
    
    print(f"\nSaved {len(all_translations)} translations to {args.output}")


if __name__ == "__main__":
    main()
