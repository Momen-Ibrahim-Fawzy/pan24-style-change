"""Build training/validation pair tables with approach one (consecutive) or two (grouped).

    python scripts/prepare_pairs.py --data data/pan24-multi-author-analysis --approach grouped --out data/pairs
"""
import argparse
from pathlib import Path

from style_change.pairs import build_pairs, pairs_frame
from style_change.pan import DIFFICULTIES, load_pan24


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--approach", choices=["consecutive", "grouped"], default="consecutive")
    ap.add_argument("--difficulties", nargs="+", default=list(DIFFICULTIES))
    ap.add_argument("--out", type=Path, default=Path("data/pairs"))
    args = ap.parse_args()

    datasets = load_pan24(args.data, args.difficulties)
    args.out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        # validation is always consecutive: that is what the task is scored on
        approach = args.approach if split == "train" else "consecutive"
        docs = [d for (diff, s), ds in datasets.items() if s == split for d in ds]
        df = pairs_frame(build_pairs(docs, approach))
        path = args.out / f"{split}_{approach}.csv"
        df.to_csv(path, index=False)
        counts = df.groupby("difficulty")["change"].value_counts().unstack(fill_value=0)
        print(f"{path}: {len(df)} pairs\n{counts.rename(columns={0: 'same', 1: 'change'})}\n")


if __name__ == "__main__":
    main()
