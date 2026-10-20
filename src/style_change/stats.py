"""Dataset statistics: paragraph lengths and label balance per difficulty and split."""
from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from .pan import Document

_WORD = re.compile(r"\w+")
_TOKEN = re.compile(r"\S+")


def length_stats(paragraphs: Sequence[str]) -> Dict[str, float]:
    words = [len(_WORD.findall(p)) for p in paragraphs]
    tokens = [len(_TOKEN.findall(p)) for p in paragraphs]
    n = max(len(paragraphs), 1)
    return {"paragraphs": len(paragraphs), "avg_words": sum(words) / n, "max_words": max(words, default=0),
            "avg_tokens": sum(tokens) / n, "max_tokens": max(tokens, default=0)}


def summary(datasets: Dict[Tuple[str, str], List[Document]]) -> pd.DataFrame:
    rows = []
    for (difficulty, split), docs in sorted(datasets.items()):
        unique = list({p for d in docs for p in d.paragraphs})
        changes = [c for d in docs for c in (d.changes or [])]
        rows.append({"difficulty": difficulty, "split": split, "documents": len(docs),
                     "pairs": len(changes), "same_author": changes.count(0), "change": changes.count(1),
                     **length_stats(unique)})
    return pd.DataFrame(rows)
