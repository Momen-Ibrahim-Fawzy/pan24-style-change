"""Fine-tune a Sentence-Transformer on paragraph pairs (one model per difficulty).

    python scripts/train_sbert.py --config configs/sbert.yaml --data data/pan24-multi-author-analysis \
        --difficulty hard --approach consecutive --model sentence-transformers/all-mpnet-base-v2
"""
import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path

import yaml

from style_change.pairs import build_pairs
from style_change.pan import load_pan24
from style_change.sbert import SBertConfig, train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/sbert.yaml"))
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--difficulty", choices=["easy", "medium", "hard"], required=True)
    ap.add_argument("--approach", choices=["consecutive", "grouped"], default="consecutive")
    ap.add_argument("--model")
    ap.add_argument("--output-dir")
    args = ap.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    cfg = SBertConfig(**{k: v for k, v in raw.items() if k in {f.name for f in fields(SBertConfig)}})
    if args.model:
        cfg.model = args.model
    cfg.output_dir = args.output_dir or f"runs/sbert/{Path(cfg.model).name}_{args.approach}_{args.difficulty}"

    data = load_pan24(args.data, [args.difficulty])
    train_pairs = build_pairs(data[(args.difficulty, "train")], args.approach)
    val_docs = data[(args.difficulty, "validation")]
    print(f"{len(train_pairs)} training pairs ({args.approach}), {len(val_docs)} validation documents")

    result = train(cfg, train_pairs, val_docs)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.output_dir, "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    Path(cfg.output_dir, "history.json").write_text(json.dumps(result, indent=2))
    print(f"best validation macro F1: {result['best_f1_macro']:.4f}")


if __name__ == "__main__":
    main()
