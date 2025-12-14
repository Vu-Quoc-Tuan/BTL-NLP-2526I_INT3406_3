# scripts/dedup_minhash.py
"""
Fuzzy deduplication using MinHash LSH.
Removes near-duplicate sentence pairs based on character n-gram similarity.
"""
import argparse
import os
import json
import re
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm
from typing import List, Tuple, Set


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    return text


def char_shingles(text: str, k: int = 5) -> Set[str]:
    """Generate character k-shingles from text."""
    text = normalize_text(text)
    if len(text) < k:
        return {text}
    return {text[i:i+k] for i in range(len(text) - k + 1)}


def build_minhash(text: str, num_perm: int = 128, k: int = 5) -> MinHash:
    """Build MinHash signature for text."""
    m = MinHash(num_perm=num_perm)
    for shingle in char_shingles(text, k=k):
        m.update(shingle.encode('utf8'))
    return m


def load_parallel_corpus(src_path: str, tgt_path: str) -> List[Tuple[str, str]]:
    """Load parallel corpus."""
    with open(src_path, encoding="utf8") as f:
        src_lines = [line.strip() for line in f]
    with open(tgt_path, encoding="utf8") as f:
        tgt_lines = [line.strip() for line in f]
    
    assert len(src_lines) == len(tgt_lines), \
        f"Line count mismatch: {len(src_lines)} vs {len(tgt_lines)}"
    
    return list(zip(src_lines, tgt_lines))


def select_representative(
    indices: List[int], 
    pairs: List[Tuple[str, str]], 
    strategy: str = "longest"
) -> int:
    """
    Select representative from a cluster.
    
    Strategies:
    - shortest: shortest source (original behavior)
    - longest: longest source (usually more complete)
    - median: median length
    """
    if strategy == "shortest":
        return min(indices, key=lambda i: len(pairs[i][0]))
    elif strategy == "longest":
        return max(indices, key=lambda i: len(pairs[i][0]))
    elif strategy == "median":
        sorted_indices = sorted(indices, key=lambda i: len(pairs[i][0]))
        return sorted_indices[len(sorted_indices) // 2]
    else:
        return indices[0]



def deduplicate_minhash(
    pairs: List[Tuple[str, str]],
    threshold: float = 0.8,
    num_perm: int = 128,
    k: int = 5,
    dedup_by: str = "both",
    rep_strategy: str = "longest"
) -> Tuple[List[int], dict]:
    """
    Deduplicate pairs using MinHash LSH.
    
    Args:
        pairs: List of (src, tgt) tuples
        threshold: Jaccard similarity threshold for considering duplicates
        num_perm: Number of permutations for MinHash
        k: Shingle size
        dedup_by: "src", "tgt", or "both"
        rep_strategy: How to select representative from cluster
    
    Returns:
        List of indices to keep, and statistics dict
    """
    n = len(pairs)
    
    # Build LSH index
    print(f"Building MinHash LSH index (threshold={threshold}, k={k})...")
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    
    minhashes = []
    for i, (src, tgt) in enumerate(tqdm(pairs, desc="Computing MinHash")):
        if dedup_by == "src":
            key_text = src
        elif dedup_by == "tgt":
            key_text = tgt
        else:  # both
            key_text = src + " ||| " + tgt
        
        m = build_minhash(key_text, num_perm=num_perm, k=k)
        minhashes.append(m)
        lsh.insert(f"m{i}", m)
    
    # Find clusters
    print("Finding duplicate clusters...")
    visited = set()
    clusters = []
    
    for i in tqdm(range(n), desc="Clustering"):
        if i in visited:
            continue
        
        # Query for similar items
        candidates = lsh.query(minhashes[i])
        cluster_indices = [int(c[1:]) for c in candidates]
        
        # Mark all as visited
        for idx in cluster_indices:
            visited.add(idx)
        
        clusters.append(cluster_indices)
    
    # Select representatives
    keep_indices = []
    for cluster in clusters:
        rep = select_representative(cluster, pairs, strategy=rep_strategy)
        keep_indices.append(rep)
    
    # Sort to maintain original order
    keep_indices = sorted(keep_indices)
    
    # Statistics
    cluster_sizes = [len(c) for c in clusters]
    stats = {
        "original_count": n,
        "kept_count": len(keep_indices),
        "removed_count": n - len(keep_indices),
        "keep_ratio": len(keep_indices) / n,
        "num_clusters": len(clusters),
        "avg_cluster_size": sum(cluster_sizes) / len(cluster_sizes) if clusters else 0,
        "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
        "singleton_clusters": sum(1 for s in cluster_sizes if s == 1),
    }
    
    return keep_indices, stats


def main():
    p = argparse.ArgumentParser(description="Fuzzy deduplication using MinHash LSH")
    
    # Input/Output
    p.add_argument("--src", required=True, help="Source file")
    p.add_argument("--tgt", required=True, help="Target file")
    p.add_argument("--out_dir", default="data/dedup", help="Output directory")
    p.add_argument("--src_lang", default="en", help="Source language code for output filename")
    p.add_argument("--tgt_lang", default="vi", help="Target language code for output filename")
    p.add_argument("--prefix", default="train", help="Output file prefix")
    
    # MinHash parameters
    p.add_argument("--k", type=int, default=5, help="Shingle size")
    p.add_argument("--num_perm", type=int, default=128, help="Number of MinHash permutations")
    p.add_argument("--threshold", type=float, default=0.8, 
                   help="Jaccard similarity threshold (0.8 = 80% similar)")
    
    # Dedup options
    p.add_argument("--dedup_by", choices=["src", "tgt", "both"], default="both",
                   help="Deduplicate by source, target, or both")
    p.add_argument("--rep_strategy", choices=["shortest", "longest", "median"], 
                   default="longest", help="How to select representative from cluster")
    
    args = p.parse_args()

    print("="*60)
    print("MinHash LSH Deduplication")
    print("="*60)

    # Load data
    print(f"\nLoading data...")
    pairs = load_parallel_corpus(args.src, args.tgt)
    print(f"Loaded {len(pairs):,} pairs")

    # Deduplicate
    keep_indices, stats = deduplicate_minhash(
        pairs,
        threshold=args.threshold,
        num_perm=args.num_perm,
        k=args.k,
        dedup_by=args.dedup_by,
        rep_strategy=args.rep_strategy
    )

    # Print statistics
    print("\n" + "="*60)
    print("DEDUPLICATION RESULTS")
    print("="*60)
    print(f"  Original pairs:     {stats['original_count']:,}")
    print(f"  Kept pairs:         {stats['kept_count']:,}")
    print(f"  Removed pairs:      {stats['removed_count']:,}")
    print(f"  Keep ratio:         {stats['keep_ratio']:.2%}")
    print(f"  Number of clusters: {stats['num_clusters']:,}")
    print(f"  Avg cluster size:   {stats['avg_cluster_size']:.2f}")
    print(f"  Max cluster size:   {stats['max_cluster_size']}")
    print(f"  Singleton clusters: {stats['singleton_clusters']:,}")
    print("="*60)

    # Save deduplicated data
    os.makedirs(args.out_dir, exist_ok=True)
    
    out_src = os.path.join(args.out_dir, f"{args.prefix}.{args.src_lang}")
    out_tgt = os.path.join(args.out_dir, f"{args.prefix}.{args.tgt_lang}")
    
    print(f"\nSaving deduplicated data...")
    with open(out_src, "w", encoding="utf8") as f_src, \
         open(out_tgt, "w", encoding="utf8") as f_tgt:
        for i in keep_indices:
            f_src.write(pairs[i][0] + "\n")
            f_tgt.write(pairs[i][1] + "\n")
    
    print(f"  Source: {out_src}")
    print(f"  Target: {out_tgt}")

    # Save metadata
    meta = {
        **stats,
        "parameters": {
            "threshold": args.threshold,
            "k": args.k,
            "num_perm": args.num_perm,
            "dedup_by": args.dedup_by,
            "rep_strategy": args.rep_strategy,
        },
        "input_files": {
            "src": args.src,
            "tgt": args.tgt,
        },
        "output_files": {
            "src": out_src,
            "tgt": out_tgt,
        }
    }
    
    meta_path = os.path.join(args.out_dir, "dedup_meta.json")
    with open(meta_path, "w", encoding="utf8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  Metadata: {meta_path}")


if __name__ == "__main__":
    main()
