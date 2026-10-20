"""Predict style changes with a trained Sentence-Transformer and write PAN solution files.

    python scripts/predict_sbert.py --model runs/sbert/all-mpnet-base-v2_consecutive_hard/best \
        --problems data/pan24-multi-author-analysis/hard/validation --out results/sbert/hard
"""
import argparse
import json
from pathlib import Path

from style_change.metrics import evaluate
from style_change.pan import read_problem_dir, write_solutions
from style_change.sbert import StyleEncoder, scores_to_changes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--problems", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--chunk-tokens", type=int, default=256)
    args = ap.parse_args()

    docs = read_problem_dir(args.problems, strict=False)
    encoder = StyleEncoder(args.model, args.chunk_tokens)
    predictions = scores_to_changes(encoder.document_scores(docs), args.threshold)
    write_solutions(predictions, args.out)
    print(f"wrote {len(predictions)} solutions to {args.out}")
    if all(d.changes is not None for d in docs):
        print(json.dumps(evaluate(docs, predictions), indent=2))


if __name__ == "__main__":
    main()
