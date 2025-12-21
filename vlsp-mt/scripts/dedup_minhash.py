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


def get_smart_threshold(text: str) -> float:
    """
    Trả về threshold dedup dựa trên độ nhạy cảm của con số trong câu.
    - Số liệu y khoa (mg, ml, mmHg...) -> threshold cao (0.95) để bảo vệ
    - Câu thường hoặc số hành chính (tuổi, ngày...) -> threshold thấp (0.85) để lọc mạnh
    """
    text_lower = text.lower()
    
    # DANH SÁCH ĐƠN VỊ Y KHOA NHẠY CẢM (Cần bảo vệ tuyệt đối)
    # Regex này bắt: Số + khoảng trắng (tùy chọn) + đơn vị
    # Ví dụ bắt: "5mg", "5 mg", "120/80 mmHg", "38.5 độ", "95%"
    sensitive_pattern = r'\d+\.?\d*\s*(mg|g|kg|ml|l|mmol|mol|iu|ui|mmhg|cmh2o|bpm|%|độ|degree|viên|tablets?|capsules?|liều|gói|ống|chai|lọ|mcg|µg|ng|pg|meq|units?)'
    
    if re.search(sensitive_pattern, text_lower):
        # CASE 1: Có số liệu lâm sàng -> Bảo vệ kỹ
        # Giữ lại nếu khác nhau dù chỉ một chút (vd: 5mg vs 50mg)
        return 0.95
    else:
        # CASE 2: Không có số HOẶC chỉ có số hành chính (tuổi, ngày, năm...)
        # Ví dụ: "Bệnh nhân 65 tuổi", "Điều trị 5 ngày"
        # -> Lọc mạnh tay để model học văn phong đa dạng, tránh lặp template
        return 0.85


def char_shingles(text: str, k: int = 5) -> Set[str]:
    """Generate character k-shingles from text."""
    text = normalize_text(text)
    if len(text) < k:
        return {text}
    return {text[i:i+k] for i in range(len(text) - k + 1)}


def build_minhash(text: str, num_perm: int = 128, k: int = 5) -> MinHash:
    """Build MinHash signature for text."""
    if not text.strip() or not text :
        return MinHash(num_perm=num_perm)
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



def _compute_minhash_worker(args):
    """Worker function for parallel MinHash computation."""
    idx, text, num_perm, k = args
    m = build_minhash(text, num_perm=num_perm, k=k)
    return idx, m


def deduplicate_minhash(
    pairs: List[Tuple[str, str]],
    threshold: float = 0.8,
    num_perm: int = 128,
    k: int = 5,
    dedup_by: str = "both",
    rep_strategy: str = "longest",
    num_workers: int = 12,
    use_smart_threshold: bool = True
) -> Tuple[List[int], dict]:
    """
    Deduplicate pairs using MinHash LSH with smart threshold.
    
    Smart threshold:
    - 0.95 cho câu có số liệu y khoa (mg, ml, mmHg...) -> bảo vệ kỹ
    - 0.85 cho câu thường -> lọc mạnh tay
    """
    import multiprocessing
    
    n = len(pairs)
    
    # Prepare texts for hashing
    print(f"Preparing texts for hashing...")
    texts = []
    for src, tgt in pairs:
        if dedup_by == "src":
            texts.append(src)
        elif dedup_by == "tgt":
            texts.append(tgt)
        else:  # both
            texts.append(src + " ||| " + tgt)
    
    # Compute smart thresholds for each text
    if use_smart_threshold:
        print("Computing smart thresholds based on medical content...")
        thresholds = [get_smart_threshold(text) for text in texts]
        high_thresh_count = sum(1 for t in thresholds if t >= 0.95)
        low_thresh_count = n - high_thresh_count
        print(f"  High threshold (0.95 - medical): {high_thresh_count:,} ({high_thresh_count/n:.1%})")
        print(f"  Low threshold (0.85 - normal):   {low_thresh_count:,} ({low_thresh_count/n:.1%})")
        # Use lower threshold for LSH index (to catch all potential duplicates)
        lsh_threshold = 0.85
    else:
        thresholds = [threshold] * n
        lsh_threshold = threshold
    
    # Build MinHash signatures (parallel for large datasets)
    print(f"Building MinHash signatures (LSH threshold={lsh_threshold}, k={k})...")
    minhashes = [None] * n
    
    if n > 10000 and num_workers > 1:
        # Use multiprocessing for large datasets
        num_workers = min(num_workers, multiprocessing.cpu_count())
        print(f"Using {num_workers} workers for parallel processing...")
        
        work_items = [(i, texts[i], num_perm, k) for i in range(n)]
        
        # Use Pool.imap_unordered for better tqdm compatibility
        with multiprocessing.Pool(processes=num_workers) as pool:
            for idx, m in tqdm(
                pool.imap_unordered(_compute_minhash_worker, work_items, chunksize=1000),
                total=n,
                desc="Computing MinHash"
            ):
                minhashes[idx] = m
    else:
        # Sequential for small datasets (less overhead)
        for i, text in enumerate(tqdm(texts, desc="Computing MinHash")):
            minhashes[i] = build_minhash(text, num_perm=num_perm, k=k)
    
    # Build LSH index with lower threshold to catch all candidates
    print("Building LSH index...")
    lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
    for i, m in enumerate(tqdm(minhashes, desc="Indexing")):
        lsh.insert(f"m{i}", m)
    
    # Find clusters using Union-Find with smart threshold
    print("Finding duplicate clusters (with smart threshold filtering)...")
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])  # Path compression
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Query and union similar items - apply smart threshold
    skipped_by_smart = 0
    for i in tqdm(range(n), desc="Clustering"):
        candidates = lsh.query(minhashes[i])
        for c in candidates:
            j = int(c[1:])
            if i != j:
                # Compute actual Jaccard similarity
                similarity = minhashes[i].jaccard(minhashes[j])
                # Use the higher threshold of the two (more protective)
                effective_threshold = max(thresholds[i], thresholds[j])
                if similarity >= effective_threshold:
                    union(i, j)
                else:
                    skipped_by_smart += 1
    
    if use_smart_threshold:
        print(f"  Pairs skipped by smart threshold: {skipped_by_smart:,}")
    
    # Group by root
    from collections import defaultdict
    clusters_dict = defaultdict(list)
    for i in range(n):
        root = find(i)
        clusters_dict[root].append(i)
    
    clusters = list(clusters_dict.values())
    
    # Select representatives
    keep_indices = []
    for cluster in clusters:
        rep = select_representative(cluster, pairs, strategy=rep_strategy)
        keep_indices.append(rep)
    
    # Sort to maintain original order
    keep_indices = sorted(keep_indices)
    
    # Clear minhashes to free memory
    del minhashes
    
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
    p.add_argument("--smart_threshold", action="store_true", default=True,
                   help="Use smart threshold: 0.95 for medical numbers (mg, ml...), 0.85 for normal text")
    p.add_argument("--no_smart_threshold", dest="smart_threshold", action="store_false",
                   help="Disable smart threshold, use fixed --threshold value")
    
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
        rep_strategy=args.rep_strategy,
        use_smart_threshold=args.smart_threshold
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
            "smart_threshold": args.smart_threshold,
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
