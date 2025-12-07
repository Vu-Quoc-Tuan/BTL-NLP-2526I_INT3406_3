# scripts/generate.py
import argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def build_prompt_en2vi(src):
    return ("You are a professional medical translator.\nTranslate the following English medical sentence into Vietnamese.\n\nEnglish: "+src+"\nVietnamese:")
def build_prompt_vi2en(src):
    return ("You are a professional medical translator.\nTranslate the following Vietnamese medical sentence into English.\n\nVietnamese: "+src+"\nEnglish:")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--direction", choices=["en2vi","vi2en"], required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max_new_tokens", type=int, default=128)
    args=p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    base = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, device_map="auto")
    model = PeftModel.from_pretrained(base, args.adapter_path)
    model.eval()
    device = next(model.parameters()).device

    with open(args.input, encoding="utf8") as fin, open(args.output,"w",encoding="utf8") as fout:
        for line in fin:
            s=line.strip()
            if not s:
                fout.write("\n"); continue
            prompt = build_prompt_en2vi(s) if args.direction=="en2vi" else build_prompt_vi2en(s)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            out = model.generate(input_ids, max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=4)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            hyp = text[len(prompt):].strip() if text.startswith(prompt) else text
            fout.write(hyp.replace("\n"," ") + "\n")
