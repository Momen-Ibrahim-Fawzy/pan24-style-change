"""Print document, pair and paragraph-length statistics for every difficulty and split.

    python scripts/dataset_stats.py --data data/pan24-multi-author-analysis --out results/dataset_stats.csv
"""
import argparse
from pathlib import Path

from style_change.pan import load_pan24
from style_change.stats import summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    table = summary(load_pan24(args.data))
    print(table.round(1).to_string(index=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
