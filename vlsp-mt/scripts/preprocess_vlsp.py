# scripts/preprocess_vlsp.py
import argparse, os
def clean_lines(infile, outfile, max_tokens=512):
    with open(infile, encoding="utf8") as fi, open(outfile,"w",encoding="utf8") as fo:
        for line in fi:
            s = line.strip()
            if not s: continue
            if len(s.split()) > max_tokens: continue
            fo.write(s+"\n")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src_in", required=True)
    p.add_argument("--tgt_in", required=True)
    p.add_argument("--out_dir", default="data/clean")
    p.add_argument("--max_tokens", type=int, default=256)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    clean_lines(args.src_in, os.path.join(args.out_dir,"train.en"), args.max_tokens)
    clean_lines(args.tgt_in, os.path.join(args.out_dir,"train.vi"), args.max_tokens)
    print("Cleaned files saved to", args.out_dir)
