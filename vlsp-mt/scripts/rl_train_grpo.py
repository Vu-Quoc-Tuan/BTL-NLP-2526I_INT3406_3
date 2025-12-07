# scripts/rl_train_grpo.py
import argparse, os, torch, random, math
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sacrebleu
from tqdm import tqdm

def build_prompt_en2vi(src):
    return ("You are a professional medical translator.\nTranslate the following English medical sentence into Vietnamese.\n\nEnglish: "+src+"\nVietnamese:")
def build_prompt_vi2en(src):
    return ("You are a professional medical translator.\nTranslate the following Vietnamese medical sentence into English.\n\nVietnamese: "+src+"\nEnglish:")

def sentence_reward(hyp, ref, alpha=0.6, beta=0.4):
    bleu = sacrebleu.sentence_bleu(hyp, [ref]).score/100.0
    chrf = sacrebleu.sentence_chrf(hyp, [ref]).score/100.0
    return alpha*bleu + beta*chrf

def get_seq_logprob_and_kl(cur_model, sft_model, tokenizer, prompt, gen_ids):
    """
    Returns: logprob (scalar tensor) under cur_model for generated tokens,
             kl (scalar tensor) estimate between sft_model and cur_model on full sequence
    """
    device = gen_ids.device
    # build concat input_ids = prompt_ids + gen_ids
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    full_ids = torch.cat([prompt_ids, gen_ids], dim=1)
    attention_mask = torch.ones_like(full_ids)

    # cur logits
    cur_out = cur_model(full_ids, attention_mask=attention_mask, return_dict=True)
    cur_logits = cur_out.logits  # [1, L, V]
    # sft logits
    with torch.no_grad():
        sft_out = sft_model(full_ids, attention_mask=attention_mask, return_dict=True)
        sft_logits = sft_out.logits

    # compute token logprobs for generated suffix
    # shifted: logits[:, :-1, :] predict next token at positions 0..L-2 for target ids 1..L-1
    shifted_logits = cur_logits[:, :-1, :].contiguous()
    target_ids = full_ids[:, 1:].contiguous()
    log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
    # pick positions corresponding to generated suffix tokens
    gen_len = gen_ids.shape[1]
    token_logprobs = torch.gather(log_probs, 2, target_ids.unsqueeze(-1)).squeeze(-1)  # [1, L-1]
    token_logprobs_gen = token_logprobs[:, -gen_len:]  # last gen_len tokens
    seq_logprob = token_logprobs_gen.sum(dim=1)  # [1]

    # KL: compute per-token KL(sft || cur) for whole sequence (or suffix only)
    sft_p = torch.nn.functional.log_softmax(sft_logits[:, :-1, :], dim=-1)
    cur_p = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
    # kl per token: sum p_sft * (log p_sft - log p_cur)
    kl_token = torch.sum(torch.exp(sft_p) * (sft_p - cur_p), dim=-1)  # [1, L-1]
    kl_seq = kl_token.mean()  # scalar
    return seq_logprob.squeeze(0), kl_seq

def load_dataset(src_path, tgt_path):
    src = open(src_path, encoding="utf8").read().splitlines()
    tgt = open(tgt_path, encoding="utf8").read().splitlines()
    assert len(src)==len(tgt)
    return list(zip(src,tgt))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--sft_adapter", required=True)   # path to sft adapter (frozen policy)
    p.add_argument("--init_adapter", required=True)  # path to init adapter to be updated (can be same as sft)
    p.add_argument("--rl_src", required=True)
    p.add_argument("--rl_tgt", required=True)
    p.add_argument("--run_id", required=True)
    p.add_argument("--direction", choices=["en2vi","vi2en"], required=True)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--beta", type=float, default=0.4)
    p.add_argument("--kl_coef", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=128)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    # load sft model (frozen) and cur model (trainable adapter)
    sft_base = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, device_map="auto")
    sft_model = PeftModel.from_pretrained(sft_base, args.sft_adapter).to(device)
    sft_model.eval()

    cur_base = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, device_map="auto")
    cur_model = PeftModel.from_pretrained(cur_base, args.init_adapter).to(device)
    cur_model.train()

    # ensure only adapter params are updated
    adapter_params = [p for n,p in cur_model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(adapter_params, lr=args.lr)

    data = load_dataset(args.rl_src, args.rl_tgt)
    print("Loaded RL data:", len(data))

    run_dir = os.path.join("runs", args.run_id)
    os.makedirs(run_dir, exist_ok=True)

    baseline = 0.0
    decay = 0.99

    for epoch in range(args.epochs):
        random.shuffle(data)
        for i in range(0, len(data), args.batch_size):
            batch = data[i:i+args.batch_size]
            losses = []
            optimizer.zero_grad()
            # accumulate per-example losses
            batch_loss = 0.0
            for (src, ref) in batch:
                prompt = build_prompt_en2vi(src) if args.direction=="en2vi" else build_prompt_vi2en(src)
                # sample one hypothesis
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                gen_out = cur_model.generate(input_ids, do_sample=True, top_p=0.9, max_new_tokens=args.max_new_tokens)
                text = tokenizer.decode(gen_out[0], skip_special_tokens=True)
                hyp = text[len(prompt):].strip() if text.startswith(prompt) else text
                # compute reward
                r = sentence_reward(hyp, ref, alpha=args.alpha, beta=args.beta)
                # create gen token ids for logprob computation
                gen_ids = tokenizer(hyp, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
                if gen_ids.shape[1]==0:
                    # empty generation -> skip
                    continue
                seq_logprob, kl = get_seq_logprob_and_kl(cur_model, sft_model, tokenizer, prompt, gen_ids)
                # update baseline
                baseline = decay*baseline + (1-decay)*r
                advantage = r - baseline
                # RL loss = -advantage * logprob + kl_coef * kl
                loss = - (advantage * seq_logprob) + args.kl_coef * kl
                # backprop accumulate
                loss.backward()
                batch_loss += loss.item()
            # step optimizer after batch processed
            torch.nn.utils.clip_grad_norm_(adapter_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            if (i//args.batch_size) % 10 == 0:
                print(f"Epoch {epoch} Batch {i//args.batch_size} loss={batch_loss:.4f} baseline={baseline:.4f}")
        # save adapter each epoch
        cur_model.save_pretrained(os.path.join(run_dir, "lora_rl"))
        print("Saved RL adapter to", os.path.join(run_dir, "lora_rl"))

    print("RL training done.")

if __name__=="__main__":
    main()
