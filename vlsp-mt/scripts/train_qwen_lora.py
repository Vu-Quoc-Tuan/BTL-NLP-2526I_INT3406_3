# scripts/train_qwen_lora.py
import os, argparse, json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

def build_prompt_en2vi(src):
    return ("You are a professional medical translator.\n"
            "Translate the following English medical sentence into Vietnamese.\n\n"
            f"English: {src}\nVietnamese:")

def build_prompt_vi2en(src):
    return ("You are a professional medical translator.\n"
            "Translate the following Vietnamese medical sentence into English.\n\n"
            f"Vietnamese: {src}\nEnglish:")

def tokenize_example(example, tokenizer, direction, max_len):
    src, tgt = example["src"], example["tgt"]
    prompt = build_prompt_en2vi(src) if direction=="en2vi" else build_prompt_vi2en(src)
    full = prompt + " " + tgt
    tok = tokenizer(full, truncation=True, max_length=max_len, padding="max_length")
    # mask prompt tokens
    prompt_len = len(tokenizer(prompt, truncation=True, max_length=max_len, add_special_tokens=False)["input_ids"])
    labels = tok["input_ids"].copy()
    for i in range(prompt_len):
        labels[i] = -100
    tok["labels"] = labels
    return tok

def make_dataset(src_path, tgt_path, subset=None):
    with open(src_path, encoding="utf8") as f:
        src_lines = [l.strip() for l in f if l.strip()]
    with open(tgt_path, encoding="utf8") as f:
        tgt_lines = [l.strip() for l in f if l.strip()]
    assert len(src_lines)==len(tgt_lines)
    if subset:
        src_lines = src_lines[:subset]; tgt_lines=tgt_lines[:subset]
    return Dataset.from_dict({"src": src_lines, "tgt": tgt_lines})

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="qwen/qwen-2.5b-instruct")
    p.add_argument("--direction", choices=["en2vi","vi2en"], required=True)
    p.add_argument("--src", required=True); p.add_argument("--tgt", required=True)
    p.add_argument("--run_id", required=True); p.add_argument("--out_dir", default="runs")
    p.add_argument("--lora_r", type=int, default=16); p.add_argument("--lora_alpha", type=int, default=32); p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=2e-4); p.add_argument("--batch_size", type=int, default=4); p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max_len", type=int, default=256); p.add_argument("--subset", type=int, default=None)
    args = p.parse_args()

    run_dir = os.path.join(args.out_dir, args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    adapter_name = "lora_en2vi" if args.direction=="en2vi" else "lora_vi2en"
    adapter_save_path = os.path.join(run_dir, adapter_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, device_map="auto")
    # set up LoRA
    lora_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=["q_proj","v_proj"], lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    ds = make_dataset(args.src, args.tgt, args.subset)
    tokenized = ds.map(lambda ex: tokenize_example(ex, tokenizer, args.direction, args.max_len), batched=False)
    tokenized.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

    training_args = TrainingArguments(
        output_dir=run_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=100,
        save_strategy="epoch",
        save_total_limit=3,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized, tokenizer=tokenizer,
                      data_collator=lambda data: {"input_ids": torch.stack([d["input_ids"] for d in data]),
                                                  "attention_mask": torch.stack([d["attention_mask"] for d in data]),
                                                  "labels": torch.stack([d["labels"] for d in data])})
    trainer.train()
    # save adapter
    model.save_pretrained(adapter_save_path)
    # save meta
    meta = vars(args)
    with open(os.path.join(run_dir,"meta.json"), "w", encoding="utf8") as f: json.dump(meta,f,indent=2)
    print("Saved adapter to", adapter_save_path)

if __name__=="__main__":
    main()
