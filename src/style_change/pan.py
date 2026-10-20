"""Reading and writing the PAN 2024 multi-author writing style analysis format.

Layout of the released data::

    pan24-multi-author-analysis/
      easy|medium|hard/
        train|validation/
          problem-<id>.txt          one paragraph per line
          truth-problem-<id>.json   {"authors": k, "changes": [0/1, ...]}   (n_paragraphs - 1 labels)

``changes[i] == 1`` means paragraphs ``i`` and ``i + 1`` were written by different authors.
Predictions are written as ``solution-problem-<id>.json`` with a single ``changes`` key.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

DIFFICULTIES = ("easy", "medium", "hard")
SPLITS = ("train", "validation")
_PROBLEM = re.compile(r"^problem-(\d+)\.txt$")


@dataclass
class Document:
    id: str
    paragraphs: List[str]
    changes: Optional[List[int]] = None
    authors: Optional[int] = None
    difficulty: str = ""
    split: str = ""
    meta: Dict = field(default_factory=dict)

    @property
    def n_pairs(self) -> int:
        return len(self.paragraphs) - 1


def read_paragraphs(path: Union[str, Path]) -> List[str]:
    text = Path(path).read_text(encoding="utf-8")
    return [line.strip() for line in text.split("\n") if line.strip()]


def read_problem_dir(directory: Union[str, Path], difficulty: str = "", split: str = "",
                     strict: bool = True) -> List[Document]:
    """Read every problem in one folder, attaching ground truth when it exists.

    With ``strict`` a document whose label count does not match its paragraph count raises; otherwise the
    document is skipped.
    """
    directory = Path(directory)
    ids = sorted((m.group(1) for p in directory.iterdir() if (m := _PROBLEM.match(p.name))), key=int)
    docs: List[Document] = []
    for pid in ids:
        paragraphs = read_paragraphs(directory / f"problem-{pid}.txt")
        truth_path = directory / f"truth-problem-{pid}.json"
        changes = authors = None
        if truth_path.exists():
            truth = json.loads(truth_path.read_text(encoding="utf-8"))
            changes, authors = [int(c) for c in truth["changes"]], truth.get("authors")
            if len(changes) != len(paragraphs) - 1:
                if strict:
                    raise ValueError(f"{directory}/problem-{pid}: {len(paragraphs)} paragraphs, "
                                     f"{len(changes)} labels")
                continue
        docs.append(Document(pid, paragraphs, changes, authors, difficulty, split))
    return docs


def load_pan24(root: Union[str, Path], difficulties: Sequence[str] = DIFFICULTIES,
               splits: Sequence[str] = SPLITS) -> Dict[Tuple[str, str], List[Document]]:
    root = Path(root)
    return {(d, s): read_problem_dir(root / d / s, d, s)
            for d in difficulties for s in splits if (root / d / s).is_dir()}


def write_solutions(predictions: Dict[str, Sequence[int]], out_dir: Union[str, Path]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for pid, changes in predictions.items():
        (out_dir / f"solution-problem-{pid}.json").write_text(
            json.dumps({"changes": [int(c) for c in changes]}), encoding="utf-8")


def read_solutions(directory: Union[str, Path]) -> Dict[str, List[int]]:
    out = {}
    for path in Path(directory).glob("solution-problem-*.json"):
        pid = path.stem.split("-")[-1]
        out[pid] = [int(c) for c in json.loads(path.read_text(encoding="utf-8"))["changes"]]
    return out


def author_segments(changes: Iterable[int]) -> List[int]:
    """Segment index of every paragraph: consecutive paragraphs with no change share a segment.

    ``[0, 1, 0, 1]`` -> ``[0, 0, 1, 1, 2]``. Segments only identify runs; two non-adjacent segments may
    still belong to the same author, which the ground truth does not reveal.
    """
    segments = [0]
    for change in changes:
        segments.append(segments[-1] + int(change))
    return segments
