"""Turning documents into paragraph pairs for pairwise training.

The ground truth only labels *consecutive* paragraphs, so there is no author id per paragraph. Two ways
of building training pairs are compared:

* **Approach one (consecutive)**: the ``n - 1`` neighbouring pairs of a document with their labels. This
  is exactly what the model is evaluated on.
* **Approach two (grouped)**: paragraphs are grouped into runs written by one author. Every pair inside a
  run is a positive (same author). Every pair taken across two *adjacent* runs is a negative, since
  adjacent runs are separated by a change. Non-adjacent runs are never paired, because the same author
  may come back later in the document.

Labels follow PAN: ``change = 1`` for different authors and 0 for the same author.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Dict, Iterable, List

import pandas as pd

from .pan import Document, author_segments


@dataclass(frozen=True)
class Pair:
    text1: str
    text2: str
    change: int
    doc_id: str = ""
    difficulty: str = ""

    @property
    def same_author(self) -> int:
        return 1 - self.change


def consecutive_pairs(doc: Document) -> List[Pair]:
    if doc.changes is None:
        raise ValueError(f"document {doc.id} has no ground truth")
    return [Pair(doc.paragraphs[i], doc.paragraphs[i + 1], int(c), doc.id, doc.difficulty)
            for i, c in enumerate(doc.changes)]


def grouped_pairs(doc: Document) -> List[Pair]:
    if doc.changes is None:
        raise ValueError(f"document {doc.id} has no ground truth")
    segments = author_segments(doc.changes)
    groups: Dict[int, List[str]] = {}
    for paragraph, seg in zip(doc.paragraphs, segments):
        groups.setdefault(seg, []).append(paragraph)

    pairs: List[Pair] = []
    for seg in sorted(groups):
        pairs += [Pair(a, b, 0, doc.id, doc.difficulty) for a, b in combinations(groups[seg], 2)]
        if seg + 1 in groups:
            pairs += [Pair(a, b, 1, doc.id, doc.difficulty) for a, b in product(groups[seg], groups[seg + 1])]
    return pairs


APPROACHES = {"consecutive": consecutive_pairs, "grouped": grouped_pairs}


def build_pairs(docs: Iterable[Document], approach: str = "consecutive") -> List[Pair]:
    fn = APPROACHES[approach]
    return [pair for doc in docs for pair in fn(doc)]


def pairs_frame(pairs: Iterable[Pair]) -> pd.DataFrame:
    return pd.DataFrame([{"text1": p.text1, "text2": p.text2, "change": p.change,
                          "doc_id": p.doc_id, "difficulty": p.difficulty} for p in pairs])


def frame_to_pairs(df: pd.DataFrame) -> List[Pair]:
    return [Pair(str(r.text1), str(r.text2), int(r.change), str(getattr(r, "doc_id", "")),
                 str(getattr(r, "difficulty", ""))) for r in df.itertuples(index=False)]
