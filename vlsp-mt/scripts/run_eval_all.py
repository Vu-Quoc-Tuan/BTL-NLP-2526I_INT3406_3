# scripts/run_eval_all.py
import argparse, subprocess, os
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--run_id", required=True)
    p.add_argument("--direction", choices=["en2vi","vi2en"], required=True)
    p.add_argument("--model_name", default="qwen/qwen-2.5b-instruct")
    args=p.parse_args()
    adapter = os.path.join("runs", args.run_id, "lora_en2vi" if args.direction=="en2vi" else "lora_vi2en")
    input_file = "data/clean/dev.en" if args.direction=="en2vi" else "data/clean/dev.vi"
    out_file = os.path.join("outputs", f"dev.hyp.{args.run_id}.{args.direction}.vi" if args.direction=="en2vi" else f"dev.hyp.{args.run_id}.{args.direction}.en")
    os.makedirs("outputs", exist_ok=True)
    cmd = f"python scripts/generate.py --model_name {args.model_name} --adapter_path {adapter} --direction {args.direction} --input {input_file} --output {out_file}"
    print("RUN:", cmd); subprocess.check_call(cmd, shell=True)
    ref = "data/clean/dev.vi" if args.direction=="en2vi" else "data/clean/dev.en"
    cmd2 = f"python scripts/eval_bleu.py --hyp {out_file} --ref {ref}"
    subprocess.check_call(cmd2, shell=True)
