# scripts/run_eval_all.py
"""
Run evaluation pipeline: generate translations then compute BLEU/chrF scores.
"""
import argparse
import subprocess
import os
import sys


def check_file_exists(path, desc):
    """Check if file/dir exists, exit with helpful message if not."""
    if not os.path.exists(path):
        print(f"ERROR: {desc} not found: {path}")
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description="Run full evaluation pipeline")
    p.add_argument("--run_id", required=True, help="Run ID (folder name under runs/)")
    p.add_argument("--direction", choices=["en2vi", "vi2en"], required=True)
    p.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct", 
                   help="Base model name on HuggingFace")
    p.add_argument("--adapter_name", default=None,
                   help="Adapter folder name (default: lora_{direction} or lora_rl)")
    p.add_argument("--input_file", default=None, help="Override input file path")
    p.add_argument("--ref_file", default=None, help="Override reference file path")
    p.add_argument("--split", default="dev", choices=["dev", "test"], 
                   help="Which split to evaluate")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Limit number of samples (for quick testing)")
    args = p.parse_args()

    run_dir = os.path.join("runs", args.run_id)
    check_file_exists(run_dir, "Run directory")

    # Determine adapter path
    if args.adapter_name:
        adapter = os.path.join(run_dir, args.adapter_name)
    else:
        # Try common adapter names
        candidates = [
            f"lora_{args.direction}",
            "lora_rl",
            "final_model",
            "best_model",
            f"lora_{args.direction}_sft"
        ]
        adapter = None
        for name in candidates:
            path = os.path.join(run_dir, name)
            if os.path.exists(path):
                adapter = path
                break
        if adapter is None:
            print(f"ERROR: No adapter found in {run_dir}")
            print(f"Tried: {candidates}")
            sys.exit(1)

    check_file_exists(adapter, "Adapter")
    print(f"Using adapter: {adapter}")

    # Determine input/output files
    src_lang = "en" if args.direction == "en2vi" else "vi"
    tgt_lang = "vi" if args.direction == "en2vi" else "en"

    input_file = args.input_file or f"data/clean/{args.split}.{src_lang}"
    ref_file = args.ref_file or f"data/clean/{args.split}.{tgt_lang}"

    check_file_exists(input_file, "Input file")
    check_file_exists(ref_file, "Reference file")

    # Output file
    os.makedirs("outputs", exist_ok=True)
    out_file = os.path.join(
        "outputs", 
        f"{args.split}.hyp.{args.run_id}.{args.direction}.{tgt_lang}"
    )

    # Step 1: Generate translations
    print("\n" + "="*60)
    print("Step 1: Generating translations...")
    print("="*60)

    gen_cmd = [
        "python", "scripts/generate.py",
        "--model_name", args.model_name,
        "--adapter_path", adapter,
        "--direction", args.direction,
        "--input", input_file,
        "--output", out_file
    ]
    if args.max_samples:
        gen_cmd.extend(["--max_samples", str(args.max_samples)])

    print(f"Running: {' '.join(gen_cmd)}")
    try:
        subprocess.check_call(gen_cmd)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Generation failed with code {e.returncode}")
        sys.exit(1)

    # Step 2: Evaluate BLEU/chrF
    print("\n" + "="*60)
    print("Step 2: Computing BLEU and chrF scores...")
    print("="*60)

    eval_cmd = [
        "python", "scripts/eval_bleu.py",
        "--hyp", out_file,
        "--ref", ref_file
    ]

    print(f"Running: {' '.join(eval_cmd)}")
    try:
        subprocess.check_call(eval_cmd)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Evaluation failed with code {e.returncode}")
        sys.exit(1)

    print("\n" + "="*60)
    print(f"Evaluation complete!")
    print(f"Hypotheses saved to: {out_file}")
    print("="*60)


if __name__ == "__main__":
    main()
