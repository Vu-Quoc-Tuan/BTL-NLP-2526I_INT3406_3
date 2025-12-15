# scripts/generate.py
"""
Generate translations using trained LoRA model.
Supports batching for faster inference.
"""
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
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


def extract_translation(full_text: str, prompt: str) -> str:
    """Extract translation from generated text, handling various edge cases."""
    # Method 1: Find the prompt and take everything after
    if prompt in full_text:
        hyp = full_text.split(prompt, 1)[1]
    else:
        # Method 2: Look for the language marker
        markers = ["Vietnamese:", "English:"]
        hyp = full_text
        for marker in markers:
            if marker in full_text:
                parts = full_text.rsplit(marker, 1)
                if len(parts) > 1:
                    hyp = parts[1]
                    break
    
    # Clean up
    hyp = hyp.strip()
    
    # Remove any trailing incomplete sentences or artifacts
    # Stop at common end markers if they appear mid-generation
    stop_markers = ["\n\nEnglish:", "\n\nVietnamese:", "\n\nYou are", "<|im_end|>", "<|endoftext|>"]
    for marker in stop_markers:
        if marker in hyp:
            hyp = hyp.split(marker)[0]
    
    # Replace newlines with spaces
    hyp = hyp.replace("\n", " ").strip()
    
    return hyp


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
    device
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
    
    # Generate
    with torch.no_grad():
        if do_sample:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=num_beams,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    
    # Decode
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # Extract translations
    translations = []
    for full_text, prompt in zip(generated_texts, prompts):
        hyp = extract_translation(full_text, prompt)
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
    p.add_argument("--num_beams", type=int, default=1, help="Beam search beams (1=greedy, faster)")
    
    args = p.parse_args()

    # ============================================================
    # Load model
    # ============================================================
    print(f"Loading model: {args.model_name}")
    print(f"Loading adapter: {args.adapter_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
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
    model = PeftModel.from_pretrained(base, args.adapter_path)
    model.eval()
    
    device = next(model.parameters()).device
    print(f"Model loaded on {device}, dtype={dtype}")

    # ============================================================
    # Load input
    # ============================================================
    lines = load_input_file(args.input, args.max_samples)
    print(f"Loaded {len(lines)} sentences from {args.input}")
    
    build_prompt = build_prompt_en2vi if args.direction == "en2vi" else build_prompt_vi2en

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
                device=device
            )
        else:
            translations = []
        
        # Reconstruct with empty lines
        batch_results = [""] * len(batch_lines)
        for idx, trans in zip(batch_indices, translations):
            batch_results[idx] = trans
        
        all_translations.extend(batch_results)

    # ============================================================
    # Save output
    # ============================================================
    with open(args.output, "w", encoding="utf8") as f:
        for hyp in all_translations:
            f.write(hyp + "\n")
    
    print(f"\nSaved {len(all_translations)} translations to {args.output}")


if __name__ == "__main__":
    main()
