import argparse
import os
import re
import unicodedata
import random
from collections import Counter
from typing import List, Tuple, Optional


def normalize_unicode(text: str) -> str:
    """Normalize unicode to NFC form."""
    return unicodedata.normalize("NFC", text)


def clean_text(text: str) -> str:
    """Clean a single text string."""
    # Normalize unicode
    text = normalize_unicode(text)
    
    # Remove HTML entities
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    text = re.sub(r'&#\d+;', ' ', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'www\.\S+', ' ', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip
    text = text.strip()
    
    return text


def is_valid_pair(
    src: str, 
    tgt: str, 
    min_len: int = 3,
    max_len: int = 256,
    max_ratio: float = 3.0,
    min_alpha_ratio: float = 0.5
) -> bool:
    """
    Check if a source-target pair is valid.
    
    Args:
        src: Source sentence
        tgt: Target sentence
        min_len: Minimum word count
        max_len: Maximum word count
        max_ratio: Maximum length ratio between src and tgt
        min_alpha_ratio: Minimum ratio of alphabetic characters
    """
    src_words = src.split()
    tgt_words = tgt.split()
    
    src_len = len(src_words)
    tgt_len = len(tgt_words)
    
    # Check minimum length
    if src_len < min_len or tgt_len < min_len:
        return False
    
    # Check maximum length
    if src_len > max_len or tgt_len > max_len:
        return False
    
    # Check length ratio (avoid misaligned pairs)
    if src_len > 0 and tgt_len > 0:
        ratio = max(src_len / tgt_len, tgt_len / src_len)
        if ratio > max_ratio:
            return False
    
    # Check if mostly alphabetic (not just numbers/symbols)
    def alpha_ratio(text):
        alpha_count = sum(1 for c in text if c.isalpha())
        return alpha_count / max(len(text), 1)
    
    if alpha_ratio(src) < min_alpha_ratio or alpha_ratio(tgt) < min_alpha_ratio:
        return False
    
    # Check for empty after cleaning
    if not src.strip() or not tgt.strip():
        return False
    
    return True



def load_parallel_corpus(src_path: str, tgt_path: str) -> List[Tuple[str, str]]:
    """Load parallel corpus from two files."""
    with open(src_path, encoding="utf8") as f:
        src_lines = [line.strip() for line in f]
    with open(tgt_path, encoding="utf8") as f:
        tgt_lines = [line.strip() for line in f]
    
    if len(src_lines) != len(tgt_lines):
        raise ValueError(f"Line count mismatch: {len(src_lines)} src vs {len(tgt_lines)} tgt")
    
    return list(zip(src_lines, tgt_lines))


def deduplicate(pairs: List[Tuple[str, str]], by: str = "both") -> List[Tuple[str, str]]:
    """
    Remove duplicate pairs.
    
    Args:
        pairs: List of (src, tgt) tuples
        by: "src", "tgt", or "both"
    """
    seen = set()
    unique = []
    
    for src, tgt in pairs:
        if by == "src":
            key = src.lower()
        elif by == "tgt":
            key = tgt.lower()
        else:  # both
            key = (src.lower(), tgt.lower())
        
        if key not in seen:
            seen.add(key)
            unique.append((src, tgt))
    
    return unique


def split_data(
    pairs: List[Tuple[str, str]], 
    dev_size: int = 1000,
    test_size: int = 1000,
    seed: int = 42
) -> Tuple[List, List, List]:
    """Split data into train/dev/test sets."""
    random.seed(seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)
    
    test = shuffled[:test_size]
    dev = shuffled[test_size:test_size + dev_size]
    train = shuffled[test_size + dev_size:]
    
    return train, dev, test


def save_parallel(pairs: List[Tuple[str, str]], src_path: str, tgt_path: str):
    """Save parallel corpus to two files."""
    with open(src_path, "w", encoding="utf8") as f_src, \
         open(tgt_path, "w", encoding="utf8") as f_tgt:
        for src, tgt in pairs:
            f_src.write(src + "\n")
            f_tgt.write(tgt + "\n")


def print_stats(name: str, pairs: List[Tuple[str, str]]):
    """Print statistics about a dataset."""
    if not pairs:
        print(f"{name}: 0 pairs")
        return
    
    src_lens = [len(s.split()) for s, _ in pairs]
    tgt_lens = [len(t.split()) for _, t in pairs]
    
    print(f"\n{name}:")
    print(f"  Pairs: {len(pairs):,}")
    print(f"  Src length: min={min(src_lens)}, max={max(src_lens)}, avg={sum(src_lens)/len(src_lens):.1f}")
    print(f"  Tgt length: min={min(tgt_lens)}, max={max(tgt_lens)}, avg={sum(tgt_lens)/len(tgt_lens):.1f}")


def main():
    p = argparse.ArgumentParser(description="Preprocess VLSP Medical Translation data")
    
    # Input
    p.add_argument("--src_in", required=True, help="Source language input file")
    p.add_argument("--tgt_in", required=True, help="Target language input file")
    p.add_argument("--src_lang", default="en", help="Source language code")
    p.add_argument("--tgt_lang", default="vi", help="Target language code")
    
    # Output
    p.add_argument("--out_dir", default="data/clean", help="Output directory")
    
    # Filtering options
    p.add_argument("--min_len", type=int, default=3, help="Minimum words per sentence")
    p.add_argument("--max_len", type=int, default=256, help="Maximum words per sentence")
    p.add_argument("--max_ratio", type=float, default=3.0, help="Maximum src/tgt length ratio")
    p.add_argument("--min_alpha", type=float, default=0.5, help="Minimum alphabetic char ratio")
    
    # Deduplication
    p.add_argument("--dedup", choices=["none", "src", "tgt", "both"], default="both",
                   help="Deduplication strategy")
    
    # Split options
    p.add_argument("--dev_size", type=int, default=1000, help="Dev set size")
    p.add_argument("--test_size", type=int, default=1000, help="Test set size")
    p.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    
    # Other
    p.add_argument("--no_split", action="store_true", help="Don't split, just clean")
    
    args = p.parse_args()

    print("="*60)
    print("VLSP Medical Translation Preprocessing")
    print("="*60)

    # Load data
    print(f"\nLoading data from:")
    print(f"  Source: {args.src_in}")
    print(f"  Target: {args.tgt_in}")
    
    pairs = load_parallel_corpus(args.src_in, args.tgt_in)
    print(f"Loaded {len(pairs):,} pairs")

    # Clean
    print("\nCleaning...")
    cleaned = []
    for src, tgt in pairs:
        src_clean = clean_text(src)
        tgt_clean = clean_text(tgt)
        if src_clean and tgt_clean:
            cleaned.append((src_clean, tgt_clean))
    print(f"After cleaning: {len(cleaned):,} pairs")

    # Filter
    print("\nFiltering...")
    filtered = [
        (src, tgt) for src, tgt in cleaned
        if is_valid_pair(
            src, tgt,
            min_len=args.min_len,
            max_len=args.max_len,
            max_ratio=args.max_ratio,
            min_alpha_ratio=args.min_alpha
        )
    ]
    print(f"After filtering: {len(filtered):,} pairs")

    # Deduplicate
    if args.dedup != "none":
        print(f"\nDeduplicating by '{args.dedup}'...")
        deduped = deduplicate(filtered, by=args.dedup)
        print(f"After dedup: {len(deduped):,} pairs")
    else:
        deduped = filtered

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)


    # Split or save all
    if args.no_split:
        # Just save cleaned data
        src_out = os.path.join(args.out_dir, f"all.{args.src_lang}")
        tgt_out = os.path.join(args.out_dir, f"all.{args.tgt_lang}")
        save_parallel(deduped, src_out, tgt_out)
        print_stats("All data", deduped)
    else:
        # Split into train/dev/test
        print(f"\nSplitting (dev={args.dev_size}, test={args.test_size})...")
        
        if len(deduped) < args.dev_size + args.test_size + 100:
            print(f"WARNING: Not enough data for requested split sizes!")
            print(f"  Available: {len(deduped)}")
            print(f"  Requested: {args.dev_size} dev + {args.test_size} test + train")
            # Adjust sizes
            total = len(deduped)
            args.test_size = min(args.test_size, total // 10)
            args.dev_size = min(args.dev_size, total // 10)
            print(f"  Adjusted to: {args.dev_size} dev + {args.test_size} test")
        
        train, dev, test = split_data(
            deduped, 
            dev_size=args.dev_size, 
            test_size=args.test_size,
            seed=args.seed
        )
        
        # Save splits
        for split_name, split_pairs in [("train", train), ("dev", dev), ("test", test)]:
            src_out = os.path.join(args.out_dir, f"{split_name}.{args.src_lang}")
            tgt_out = os.path.join(args.out_dir, f"{split_name}.{args.tgt_lang}")
            save_parallel(split_pairs, src_out, tgt_out)
            print_stats(split_name.capitalize(), split_pairs)

    # Summary
    print("\n" + "="*60)
    print("Preprocessing complete!")
    print(f"Output saved to: {args.out_dir}")
    print("="*60)
    
    # Save preprocessing config
    config = vars(args)
    config["original_count"] = len(pairs)
    config["final_count"] = len(deduped)
    
    import json
    config_path = os.path.join(args.out_dir, "preprocess_config.json")
    with open(config_path, "w", encoding="utf8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Config saved to: {config_path}")


if __name__ == "__main__":
    main()
