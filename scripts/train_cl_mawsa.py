"""Train the CL-MAWSA baseline: CoSENT encoder, then a classifier on frozen embeddings.

    python scripts/train_cl_mawsa.py --config configs/cl_mawsa.yaml --data data/pan24-multi-author-analysis \
        --difficulty medium
"""
import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path

import torch
import yaml

from style_change.cosent import CoSENTConfig, train_classifier, train_encoder
from style_change.pairs import build_pairs
from style_change.pan import load_pan24


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/cl_mawsa.yaml"))
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--difficulty", choices=["easy", "medium", "hard"], required=True)
    ap.add_argument("--approach", choices=["consecutive", "grouped"], default="grouped")
    args = ap.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    cfg = CoSENTConfig(**{k: v for k, v in raw.items() if k in {f.name for f in fields(CoSENTConfig)}})
    cfg.output_dir = f"{cfg.output_dir}/{args.difficulty}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_pan24(args.data, [args.difficulty])
    train_pairs = build_pairs(data[(args.difficulty, "train")], args.approach)
    val_docs = data[(args.difficulty, "validation")]

    encoder = train_encoder(cfg, train_pairs, device)
    best = train_classifier(cfg, encoder, build_pairs(data[(args.difficulty, "train")], "consecutive"),
                            val_docs, device)
    encoder.model.save_pretrained(f"{cfg.output_dir}/encoder")
    encoder.tokenizer.save_pretrained(f"{cfg.output_dir}/encoder")
    Path(cfg.output_dir, "result.json").write_text(json.dumps({"config": asdict(cfg), "best": best}, indent=2))
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
