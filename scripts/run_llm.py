"""Annotate PAN documents with a chat LLM and score the result.

    export OPENAI_API_KEY=...            # or PERPLEXITY_API_KEY for --provider perplexity
    python scripts/run_llm.py --data data/pan24-multi-author-analysis --model gpt-3.5-turbo \
        --split validation --out results/llm/gpt-3.5-turbo

    # prompt development on the first 50 training documents of each difficulty
    python scripts/run_llm.py --data data/pan24-multi-author-analysis --model gpt-3.5-turbo --split train --limit 50

Answers are cached (JSONL, including the raw text) so interrupted runs resume and re-scoring costs nothing.
"""
import argparse
import json
from pathlib import Path

from tqdm import tqdm

from style_change.llm import PROVIDERS, LLMAnnotator
from style_change.metrics import evaluate
from style_change.pan import DIFFICULTIES, load_pan24, write_solutions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", required=True, help="e.g. gpt-3.5-turbo, llama-3-70b-instruct, mixtral-8x7b-instruct")
    ap.add_argument("--provider", choices=list(PROVIDERS), default="openai")
    ap.add_argument("--base-url")
    ap.add_argument("--split", choices=["train", "validation"], default="validation")
    ap.add_argument("--difficulties", nargs="+", default=list(DIFFICULTIES))
    ap.add_argument("--limit", type=int, help="first N documents per difficulty")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    out = args.out or Path("results/llm") / args.model
    annotator = LLMAnnotator(args.model, args.provider, args.base_url, cache_path=str(out / "cache.jsonl"))
    data = load_pan24(args.data, args.difficulties, [args.split])
    scores = {}
    for (difficulty, split), docs in sorted(data.items()):
        docs = docs[: args.limit] if args.limit else docs
        predictions = {}
        for doc in tqdm(docs, desc=f"{args.model} {difficulty}/{split}"):
            changes = annotator.annotate(doc)
            if changes is not None:
                predictions[doc.id] = changes
        write_solutions(predictions, out / difficulty / split)
        scores[difficulty] = evaluate(docs, predictions)
        print(difficulty, json.dumps(scores[difficulty]))
    (out / f"scores_{args.split}.json").write_text(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
