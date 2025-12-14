# scripts/eval_bleu.py
"""
Evaluate machine translation quality using multiple metrics:
- BLEU (SacreBLEU)
- chrF / chrF++
- TER (Translation Edit Rate)
- COMET (optional, neural metric)
"""
import argparse
import json
import os
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF, TER


def load_file(path: str) -> list:
    """Load file and return list of lines."""
    with open(path, encoding="utf8") as f:
        lines = [line.strip() for line in f]
    return lines


def compute_metrics(hyp: list, ref: list, src: list = None) -> dict:
    """
    Compute translation metrics.
    
    Args:
        hyp: List of hypothesis translations
        ref: List of reference translations
        src: List of source sentences (optional, for COMET)
    
    Returns:
        Dictionary of metric scores
    """
    results = {}
    
    # BLEU
    bleu = BLEU()
    bleu_result = bleu.corpus_score(hyp, [ref])
    results["bleu"] = {
        "score": round(bleu_result.score, 2),
        "signature": bleu_result.format(signature=bleu.get_signature()),
    }
    
    # chrF++
    chrf = CHRF(word_order=2)  # chrF++ with word order
    chrf_result = chrf.corpus_score(hyp, [ref])
    results["chrf++"] = {
        "score": round(chrf_result.score, 2),
    }
    
    # chrF (without word order)
    chrf_basic = CHRF(word_order=0)
    chrf_basic_result = chrf_basic.corpus_score(hyp, [ref])
    results["chrf"] = {
        "score": round(chrf_basic_result.score, 2),
    }
    
    # TER
    ter = TER()
    ter_result = ter.corpus_score(hyp, [ref])
    results["ter"] = {
        "score": round(ter_result.score, 2),
    }
    
    return results


def compute_sentence_scores(hyp: list, ref: list) -> list:
    """Compute per-sentence BLEU and chrF scores."""
    scores = []
    for h, r in zip(hyp, ref):
        sent_bleu = sacrebleu.sentence_bleu(h, [r]).score
        sent_chrf = sacrebleu.sentence_chrf(h, [r]).score
        scores.append({
            "bleu": round(sent_bleu, 2),
            "chrf": round(sent_chrf, 2),
        })
    return scores


def find_worst_translations(hyp: list, ref: list, src: list = None, n: int = 10) -> list:
    """Find the worst translations by BLEU score."""
    scores = compute_sentence_scores(hyp, ref)
    
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1]["bleu"])
    
    worst = []
    for idx, score in indexed[:n]:
        item = {
            "index": idx,
            "bleu": score["bleu"],
            "chrf": score["chrf"],
            "hyp": hyp[idx],
            "ref": ref[idx],
        }
        if src:
            item["src"] = src[idx]
        worst.append(item)
    
    return worst



def main():
    p = argparse.ArgumentParser(description="Evaluate MT quality with multiple metrics")
    p.add_argument("--hyp", required=True, help="Hypothesis file")
    p.add_argument("--ref", required=True, help="Reference file")
    p.add_argument("--src", default=None, help="Source file (optional, for analysis)")
    p.add_argument("--output", default=None, help="Save results to JSON file")
    p.add_argument("--show_worst", type=int, default=0, 
                   help="Show N worst translations (0 to disable)")
    p.add_argument("--save_sentence_scores", default=None,
                   help="Save per-sentence scores to file")
    args = p.parse_args()

    # Load files
    print(f"Loading hypothesis: {args.hyp}")
    hyp = load_file(args.hyp)
    
    print(f"Loading reference: {args.ref}")
    ref = load_file(args.ref)
    
    src = None
    if args.src:
        print(f"Loading source: {args.src}")
        src = load_file(args.src)

    # Check alignment
    if len(hyp) != len(ref):
        print(f"ERROR: Line count mismatch!")
        print(f"  Hypothesis: {len(hyp)} lines")
        print(f"  Reference: {len(ref)} lines")
        return
    
    if src and len(src) != len(hyp):
        print(f"WARNING: Source file has different line count ({len(src)} vs {len(hyp)})")
        src = None

    print(f"\nEvaluating {len(hyp)} sentence pairs...")

    # Compute metrics
    results = compute_metrics(hyp, ref, src)

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"  BLEU:    {results['bleu']['score']:.2f}")
    print(f"  chrF++:  {results['chrf++']['score']:.2f}")
    print(f"  chrF:    {results['chrf']['score']:.2f}")
    print(f"  TER:     {results['ter']['score']:.2f} (lower is better)")
    print("="*50)

    # Show worst translations
    if args.show_worst > 0:
        print(f"\n{args.show_worst} WORST TRANSLATIONS (by BLEU):")
        print("-"*50)
        worst = find_worst_translations(hyp, ref, src, n=args.show_worst)
        for i, item in enumerate(worst, 1):
            print(f"\n[{i}] Line {item['index']+1} | BLEU: {item['bleu']:.1f} | chrF: {item['chrf']:.1f}")
            if "src" in item:
                print(f"  SRC: {item['src']}")
            print(f"  HYP: {item['hyp']}")
            print(f"  REF: {item['ref']}")

    # Save results
    if args.output:
        output_data = {
            "files": {
                "hypothesis": args.hyp,
                "reference": args.ref,
                "source": args.src,
            },
            "num_sentences": len(hyp),
            "metrics": results,
        }
        with open(args.output, "w", encoding="utf8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")

    # Save sentence scores
    if args.save_sentence_scores:
        print(f"\nComputing per-sentence scores...")
        sent_scores = compute_sentence_scores(hyp, ref)
        with open(args.save_sentence_scores, "w", encoding="utf8") as f:
            json.dump(sent_scores, f, indent=2)
        print(f"Sentence scores saved to: {args.save_sentence_scores}")


if __name__ == "__main__":
    main()
