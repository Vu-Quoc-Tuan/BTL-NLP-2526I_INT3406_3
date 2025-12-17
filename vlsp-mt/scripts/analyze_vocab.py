import argparse
import re
from collections import Counter


def tokenize_simple(text):
    """Simple word tokenization for Vietnamese/English."""
    # Lowercase and split on whitespace/punctuation
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    return words


def load_vocab(file_path):
    """Load file and extract vocabulary with frequency."""
    vocab = Counter()
    with open(file_path, encoding='utf8') as f:
        for line in f:
            words = tokenize_simple(line.strip())
            vocab.update(words)
    return vocab


def main():
    p = argparse.ArgumentParser(description="Analyze vocabulary overlap")
    p.add_argument("--train", required=True, help="Training file")
    p.add_argument("--test", required=True, help="Test file")
    p.add_argument("--top_oov", type=int, default=50, help="Show top N OOV words")
    p.add_argument("--save_oov", default=None, help="Save OOV words to file")
    args = p.parse_args()

    print(f"Loading train vocab from: {args.train}")
    train_vocab = load_vocab(args.train)
    print(f"  Train vocab size: {len(train_vocab):,} unique words")
    print(f"  Train total tokens: {sum(train_vocab.values()):,}")

    print(f"\nLoading test vocab from: {args.test}")
    test_vocab = load_vocab(args.test)
    print(f"  Test vocab size: {len(test_vocab):,} unique words")
    print(f"  Test total tokens: {sum(test_vocab.values()):,}")

    # Find OOV words
    oov_words = {}
    for word, count in test_vocab.items():
        if word not in train_vocab:
            oov_words[word] = count

    oov_count = len(oov_words)
    oov_tokens = sum(oov_words.values())
    test_tokens = sum(test_vocab.values())
    
    print(f"\n{'='*50}")
    print("OOV ANALYSIS")
    print(f"{'='*50}")
    print(f"  OOV word types: {oov_count:,} ({100*oov_count/len(test_vocab):.1f}% of test vocab)")
    print(f"  OOV tokens: {oov_tokens:,} ({100*oov_tokens/test_tokens:.1f}% of test tokens)")
    
    sorted_oov = sorted(oov_words.items(), key=lambda x: -x[1])
    
    print(f"\n  Top {args.top_oov} OOV words (by frequency in test):")
    for i, (word, count) in enumerate(sorted_oov[:args.top_oov], 1):
        print(f"    {i:3}. {word:30} (count: {count})")

    if args.save_oov:
        with open(args.save_oov, 'w', encoding='utf8') as f:
            for word, count in sorted_oov:
                f.write(f"{word}\t{count}\n")
        print(f"\nOOV words saved to: {args.save_oov}")

    covered_tokens = test_tokens - oov_tokens
    print(f"\n{'='*50}")
    print("COVERAGE SUMMARY")
    print(f"{'='*50}")
    print(f"  Token coverage: {100*covered_tokens/test_tokens:.2f}%")

if __name__ == "__main__":
    main()
