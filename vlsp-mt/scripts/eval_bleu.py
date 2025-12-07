# scripts/eval_bleu.py
import argparse, sacrebleu
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--hyp", required=True); p.add_argument("--ref", required=True)
    args=p.parse_args()
    hyp=open(args.hyp, encoding="utf8").read().strip().splitlines()
    ref=open(args.ref, encoding="utf8").read().strip().splitlines()
    bleu = sacrebleu.corpus_bleu(hyp, [ref])
    chrf = sacrebleu.corpus_chrf(hyp, [ref])
    print("SacreBLEU:", bleu.score, "ChrF++:", chrf.score)
