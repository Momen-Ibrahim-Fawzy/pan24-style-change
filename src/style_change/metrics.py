"""PAN 2024 evaluation: macro F1 over all paragraph pairs of a dataset."""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

import numpy as np

from .pan import Document


def f1_per_class(y_true: Sequence[int], y_pred: Sequence[int]) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    scores = []
    for c in (0, 1):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return np.array(scores, dtype=float)


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return float(f1_per_class(y_true, y_pred).mean())


def flatten(docs: Sequence[Document], predictions: Mapping[str, Sequence[int]],
            missing: int = 1) -> Dict[str, List[int]]:
    """Align predictions with the truth of each document.

    A document without a (valid) prediction gets ``missing`` for every pair. 1 ("change") is the majority
    class in PAN 2024, so this is the conservative fallback for LLM outputs that cannot be parsed.
    """
    y_true: List[int] = []
    y_pred: List[int] = []
    for doc in docs:
        pred = predictions.get(doc.id)
        if pred is None or len(pred) != len(doc.changes):
            pred = [missing] * len(doc.changes)
        y_true += list(doc.changes)
        y_pred += [int(p) for p in pred]
    return {"true": y_true, "pred": y_pred}


def evaluate(docs: Sequence[Document], predictions: Mapping[str, Sequence[int]], missing: int = 1) -> Dict[str, float]:
    flat = flatten(docs, predictions, missing)
    covered = sum(1 for d in docs if d.id in predictions and predictions[d.id] is not None
                  and len(predictions[d.id]) == len(d.changes))
    per_class = f1_per_class(flat["true"], flat["pred"])
    return {"f1_macro": float(per_class.mean()), "f1_same": float(per_class[0]), "f1_change": float(per_class[1]),
            "pairs": len(flat["true"]), "documents": len(docs), "coverage": covered / max(len(docs), 1)}
