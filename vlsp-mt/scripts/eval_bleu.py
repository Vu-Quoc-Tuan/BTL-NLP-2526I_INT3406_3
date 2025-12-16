# scripts/eval_bleu.py
"""
Evaluate machine translation quality using multiple metrics:
- BLEU (SacreBLEU)
- chrF / chrF++
- TER (Translation Edit Rate)
- METEOR
- Gemini Score (LLM-as-judge, optional)
"""
import argparse
import json
import os
import random
import time
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF, TER

# METEOR requires nltk
try:
    import nltk
    from nltk.translate.meteor_score import meteor_score
    from nltk.tokenize import word_tokenize
    METEOR_AVAILABLE = True
except ImportError:
    METEOR_AVAILABLE = False

# Gemini requires google-generativeai
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def load_file(path: str) -> list:
    """Load file and return list of lines."""
    with open(path, encoding="utf8") as f:
        lines = [line.strip() for line in f]
    return lines


def compute_meteor(hyp: list, ref: list) -> float:
    """Compute corpus-level METEOR score."""
    if not METEOR_AVAILABLE:
        return None
    
    # Ensure NLTK data is available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4', quiet=True)
    
    scores = []
    for h, r in zip(hyp, ref):
        # Tokenize
        hyp_tokens = word_tokenize(h.lower())
        ref_tokens = word_tokenize(r.lower())
        # METEOR expects reference as list of tokens
        score = meteor_score([ref_tokens], hyp_tokens)
        scores.append(score)
    
    return sum(scores) / len(scores) if scores else 0.0


def compute_gemini_score(
    hyp: list, 
    ref: list, 
    src: list,
    api_key: str,
    sample_size: int = 100,
    direction: str = "en2vi",
    verbose: bool = False,
    batch_size: int = 10
) -> dict:
    """
    Use Gemini as judge to evaluate translation quality.
    
    Uses BATCHING to reduce API calls:
    - Instead of 1 request per sentence, sends batch_size sentences per request
    - 100 samples with batch_size=10 = only 10 API calls (vs 100 without batching)
    
    Returns average score (1-5) and detailed breakdown.
    """
    if not GEMINI_AVAILABLE:
        return None
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Sample if dataset is large
    indices = list(range(len(hyp)))
    if len(indices) > sample_size:
        random.seed(42)
        indices = random.sample(indices, sample_size)
    
    src_lang = "English" if direction == "en2vi" else "Vietnamese"
    tgt_lang = "Vietnamese" if direction == "en2vi" else "English"
    
    scores = []
    detailed_results = []
    errors = 0
    
    # Split indices into batches
    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
    total_batches = len(batches)
    
    print(f"  Using batching: {len(indices)} samples / {batch_size} per batch = {total_batches} API calls")
    
    for batch_num, batch_indices in enumerate(batches):
        # Build batch prompt
        samples_text = ""
        for i, idx in enumerate(batch_indices):
            samples_text += f"""
--- Sample {i+1} ---
Source ({src_lang}): {src[idx]}
Reference ({tgt_lang}): {ref[idx]}
Translation ({tgt_lang}): {hyp[idx]}
"""
        
        prompt = f"""STRICT {src_lang}->{tgt_lang} medical translation evaluator.

Score 1-2: Hallucination (added info), Omission (missing info), Wrong medical terms, Truncated
Score 3: Minor errors, awkward phrasing
Score 4: Accurate, minor style issues
Score 5: Perfect, no errors

Compare Translation to Source ONLY. Penalize hallucinations and omissions strictly.

{samples_text}

JSON array with {len(batch_indices)} objects: [{{"score": 1-5, "reason": "brief"}}, ...]"""
        
        try:
            # Retry with exponential backoff for rate limits
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = model.generate_content(prompt)
                    break
                except Exception as retry_e:
                    if "429" in str(retry_e):
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 30  # 30s, 60s, 90s, 120s
                            print(f"\n  Rate limited, waiting {wait_time}s... (attempt {attempt+1}/{max_retries})")
                            time.sleep(wait_time)
                        else:
                            print(f"\n  Rate limit exceeded after {max_retries} retries. Waiting 120s before continuing...")
                            time.sleep(120)
                            raise retry_e
                    else:
                        raise retry_e
            
            text = response.text.strip()
            # Parse JSON from response - handle various formats
            if "```" in text:
                parts = text.split("```")
                if len(parts) >= 2:
                    text = parts[1]
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.strip()
            
            # Try to find JSON array in text
            import re
            if not text.startswith("["):
                json_match = re.search(r'\[[\s\S]*\]', text)
                if json_match:
                    text = json_match.group()
            
            batch_results = json.loads(text)
            
            # Process each result in batch
            for i, idx in enumerate(batch_indices):
                if i < len(batch_results):
                    result = batch_results[i]
                    score = result.get("score", 3)
                    reason = result.get("reason", "")
                    
                    # Validate score is in range
                    if not isinstance(score, (int, float)) or score < 1 or score > 5:
                        score = 3
                    score = int(score)
                else:
                    score = 3
                    reason = "Missing from batch response"
                    errors += 1
                
                scores.append(score)
                detailed_results.append({
                    "idx": idx,
                    "score": score,
                    "reason": reason,
                    "src": src[idx][:100] + "..." if len(src[idx]) > 100 else src[idx],
                    "hyp": hyp[idx][:100] + "..." if len(hyp[idx]) > 100 else hyp[idx],
                    "ref": ref[idx][:100] + "..." if len(ref[idx]) > 100 else ref[idx],
                })
            
        except Exception as e:
            # On batch error, assign default scores to all items in batch
            for idx in batch_indices:
                errors += 1
                scores.append(3)
                detailed_results.append({
                    "idx": idx,
                    "score": 3,
                    "reason": f"BATCH ERROR: {str(e)}",
                    "src": src[idx][:100],
                    "hyp": hyp[idx][:100],
                    "ref": ref[idx][:100],
                })
            if verbose:
                print(f"\n  DEBUG: Batch {batch_num+1} error: {str(e)}")
        
        # Progress
        processed = min((batch_num + 1) * batch_size, len(indices))
        print(f"  Gemini eval: {processed}/{len(indices)} (batch {batch_num+1}/{total_batches})")
        
        # Rate limit: wait between batches (4s per batch is safe for free tier)
        if batch_num < total_batches - 1:
            time.sleep(4)
    
    avg_score = sum(scores) / len(scores) if scores else 0
    
    # Print low scores for debugging
    if verbose:
        low_scores = [r for r in detailed_results if r["score"] <= 2]
        if low_scores:
            print(f"\n  === LOW SCORE SAMPLES (score <= 2): {len(low_scores)} ===")
            for r in low_scores[:5]:
                print(f"\n  [Line {r['idx']+1}] Score: {r['score']}")
                print(f"    Reason: {r['reason']}")
                print(f"    SRC: {r['src']}")
                print(f"    HYP: {r['hyp']}")
                print(f"    REF: {r['ref']}")
    
    return {
        "score": round(avg_score, 2),
        "samples_evaluated": len(indices),
        "errors": errors,
        "api_calls": total_batches,
        "batch_size": batch_size,
        "score_distribution": {
            str(i): scores.count(i) for i in range(1, 6)
        },
        "detailed_results": detailed_results if verbose else None
    }


def compute_metrics(hyp: list, ref: list, src: list = None, use_meteor: bool = True) -> dict:
    """
    Compute translation metrics.
    
    Args:
        hyp: List of hypothesis translations
        ref: List of reference translations
        src: List of source sentences (optional, for COMET)
        use_meteor: Whether to compute METEOR score
    
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
    
    # METEOR
    if use_meteor:
        if METEOR_AVAILABLE:
            meteor = compute_meteor(hyp, ref)
            results["meteor"] = {
                "score": round(meteor * 100, 2),  # Scale to 0-100 like other metrics
            }
        else:
            print("  METEOR not available (install nltk: pip install nltk)")
    
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
    p.add_argument("--no_meteor", action="store_true",
                   help="Skip METEOR computation (faster)")
    
    # Gemini evaluation
    p.add_argument("--gemini", action="store_true",
                   help="Use Gemini as judge for evaluation")
    p.add_argument("--gemini_api_key", default=None,
                   help="Gemini API key (or set GEMINI_API_KEY env var)")
    p.add_argument("--gemini_samples", type=int, default=100,
                   help="Number of samples for Gemini evaluation")
    p.add_argument("--gemini_batch_size", type=int, default=10,
                   help="Number of samples per API call (batching to reduce rate limits)")
    p.add_argument("--direction", default="en2vi", choices=["en2vi", "vi2en"],
                   help="Translation direction for Gemini prompt")
    p.add_argument("--gemini_verbose", action="store_true",
                   help="Show detailed Gemini evaluation (low score samples)")
    
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
    results = compute_metrics(hyp, ref, src, use_meteor=not args.no_meteor)

    # Gemini evaluation
    if args.gemini:
        if not src:
            print("ERROR: --src is required for Gemini evaluation")
        else:
            api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                print("ERROR: Gemini API key required (--gemini_api_key or GEMINI_API_KEY env)")
            elif not GEMINI_AVAILABLE:
                print("ERROR: google-generativeai not installed (pip install google-generativeai)")
            else:
                print(f"\nRunning Gemini evaluation ({args.gemini_samples} samples, batch_size={args.gemini_batch_size})...")
                gemini_result = compute_gemini_score(
                    hyp, ref, src,
                    api_key=api_key,
                    sample_size=args.gemini_samples,
                    direction=args.direction,
                    verbose=args.gemini_verbose,
                    batch_size=args.gemini_batch_size
                )
                if gemini_result:
                    results["gemini"] = gemini_result

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"  BLEU:    {results['bleu']['score']:.2f}")
    print(f"  chrF++:  {results['chrf++']['score']:.2f}")
    print(f"  chrF:    {results['chrf']['score']:.2f}")
    print(f"  TER:     {results['ter']['score']:.2f} (lower is better)")
    if "meteor" in results:
        print(f"  METEOR:  {results['meteor']['score']:.2f}")
    if "gemini" in results:
        print(f"  Gemini:  {results['gemini']['score']:.2f}/5.0 ({results['gemini']['samples_evaluated']} samples)")
        # Always show score distribution
        dist = results['gemini']['score_distribution']
        print(f"    Distribution: 1={dist.get('1',0)}, 2={dist.get('2',0)}, 3={dist.get('3',0)}, 4={dist.get('4',0)}, 5={dist.get('5',0)}")
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
