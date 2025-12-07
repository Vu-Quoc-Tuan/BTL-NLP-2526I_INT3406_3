# scripts/dedup_minhash.py
import argparse, os, sys, json
from datasketch import MinHash, MinHashLSH
import re

def char_shingles(text, k=5):
    s = re.sub(r'\s+',' ', text.strip().lower())
    return {s[i:i+k] for i in range(max(0, len(s)-k+1))}

def build_minhash(s, num_perm=128, k=5):
    m = MinHash(num_perm=num_perm)
    for sh in char_shingles(s, k=k):
        m.update(sh.encode('utf8'))
    return m

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--tgt", required=True)
    p.add_argument("--out_dir", default="data/dedup")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--num_perm", type=int, default=128)
    p.add_argument("--threshold", type=float, default=0.8)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    src_lines = open(args.src, encoding="utf8").read().splitlines()
    tgt_lines = open(args.tgt, encoding="utf8").read().splitlines()
    assert len(src_lines)==len(tgt_lines)
    n = len(src_lines)
    print("Loaded", n, "pairs")

    lsh = MinHashLSH(threshold=args.threshold, num_perm=args.num_perm)
    minhashes = {}
    for i, (s,t) in enumerate(zip(src_lines, tgt_lines)):
        key_text = s + " ||| " + t
        m = build_minhash(key_text, num_perm=args.num_perm, k=args.k)
        minhashes[i] = m
        lsh.insert(f"m{i}", m)
        if (i+1)%10000==0:
            print("Processed", i+1)
    visited = set()
    clusters = []
    for i in range(n):
        if i in visited: continue
        cand = lsh.query(minhashes[i])
        idxs = [int(c[1:]) for c in cand]
        for idx in idxs: visited.add(idx)
        clusters.append(idxs)
    print("Found", len(clusters), "clusters")

    # choose representative: shortest source length
    keep = []
    for c in clusters:
        rep = min(c, key=lambda i: len(src_lines[i]))
        keep.append(rep)
    keep_set = set(keep)
    print("Kept", len(keep_set), "pairs (ratio kept:", len(keep_set)/n, ")")

    out_src = os.path.join(args.out_dir, "train.en")
    out_tgt = os.path.join(args.out_dir, "train.vi")
    with open(out_src, "w", encoding="utf8") as fo1, open(out_tgt, "w", encoding="utf8") as fo2:
        for i in sorted(keep_set):
            fo1.write(src_lines[i].strip()+"\n")
            fo2.write(tgt_lines[i].strip()+"\n")
    print("Wrote deduped files to", args.out_dir)
    meta = {"original_pairs": n, "kept": len(keep_set)}
    with open(os.path.join(args.out_dir,"meta.json"), "w", encoding="utf8") as f:
        json.dump(meta,f,indent=2)

if __name__=="__main__":
    main()