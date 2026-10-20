"""Score PAN solution folders against the ground truth (macro F1 per difficulty).

    python scripts/evaluate.py --truth data/pan24-multi-author-analysis --split validation \
        --solutions results/sbert            # expects results/sbert/{easy,medium,hard}/solution-problem-*.json
"""
import argparse
import json
from pathlib import Path

from style_change.metrics import evaluate
from style_change.pan import DIFFICULTIES, read_problem_dir, read_solutions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--solutions", type=Path, required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--difficulties", nargs="+", default=list(DIFFICULTIES))
    args = ap.parse_args()

    results = {}
    for difficulty in args.difficulties:
        docs = read_problem_dir(args.truth / difficulty / args.split, difficulty, args.split)
        sol_dir = args.solutions / difficulty
        if (sol_dir / args.split).is_dir():
            sol_dir = sol_dir / args.split
        results[difficulty] = evaluate(docs, read_solutions(sol_dir))
    print(json.dumps(results, indent=2))
    print(" | ".join(f"{d}: {r['f1_macro']:.4f}" for d, r in results.items()))


if __name__ == "__main__":
    main()
