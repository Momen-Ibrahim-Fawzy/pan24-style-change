"""Sentence-Transformer style-change detector.

A bi-encoder (all-MiniLM-L12-v2 or all-mpnet-base-v2) is fine-tuned with a contrastive loss on paragraph
pairs so that paragraphs by the same author are close in embedding space. At inference each pair of
consecutive paragraphs is scored by cosine similarity, and a change is predicted when the similarity falls
below a threshold (0.5 by default, or tuned on validation).

Long paragraphs are split into chunks of ``chunk_tokens`` word pieces. The chunks of the two paragraphs
are aligned cyclically (the shorter list wraps around), and the pair score is the mean cosine similarity
over the aligned chunk pairs. Nothing beyond the encoder's window is lost.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .metrics import evaluate
from .pairs import Pair
from .pan import Document


@dataclass
class SBertConfig:
    model: str = "sentence-transformers/all-mpnet-base-v2"
    output_dir: str = "runs/sbert"
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 2e-5
    warmup_steps: int = 0
    margin: float = 0.5
    subsample: float = 0.7      # fraction of training pairs drawn fresh every epoch
    eval_every: int = 2         # epochs between validation runs
    chunk_tokens: int = 256
    threshold: float = 0.5
    seed: int = 42


def chunk_text(text: str, tokenize: Callable[[str], List[str]], detokenize: Callable[[List[str]], str],
               chunk_tokens: int) -> List[str]:
    tokens = tokenize(text)
    if len(tokens) <= chunk_tokens:
        return [text]
    return [detokenize(tokens[i:i + chunk_tokens]) for i in range(0, len(tokens), chunk_tokens)]


def align_chunks(chunks1: Sequence[str], chunks2: Sequence[str]) -> List[Tuple[str, str]]:
    """Pair chunk i of one paragraph with chunk i of the other; the shorter list wraps around."""
    n = max(len(chunks1), len(chunks2))
    return [(chunks1[i % len(chunks1)], chunks2[i % len(chunks2)]) for i in range(n)]


class StyleEncoder:
    """Wraps a SentenceTransformer with chunked pair scoring."""

    def __init__(self, model_or_path: str, chunk_tokens: int = 256, device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_or_path, device=device)
        self.chunk_tokens = chunk_tokens
        tok = self.model.tokenizer
        self._tokenize = lambda t: tok.encode(t, add_special_tokens=False)
        self._detokenize = lambda ids: tok.decode(ids)

    def chunks(self, text: str) -> List[str]:
        return chunk_text(text, self._tokenize, self._detokenize, self.chunk_tokens)

    def pair_scores(self, pairs: Sequence[Tuple[str, str]], batch_size: int = 64) -> np.ndarray:
        """Mean chunk-level cosine similarity for each (paragraph, paragraph) pair."""
        aligned, owner = [], []
        for k, (a, b) in enumerate(pairs):
            for x, y in align_chunks(self.chunks(a), self.chunks(b)):
                aligned.append((x, y))
                owner.append(k)
        if not aligned:
            return np.zeros(0)
        left = self.model.encode([x for x, _ in aligned], batch_size=batch_size, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False)
        right = self.model.encode([y for _, y in aligned], batch_size=batch_size, convert_to_numpy=True,
                                  normalize_embeddings=True, show_progress_bar=False)
        sims = (left * right).sum(axis=1)
        owner = np.asarray(owner)
        return np.array([sims[owner == k].mean() for k in range(len(pairs))])

    def document_scores(self, docs: Sequence[Document], batch_size: int = 64) -> Dict[str, np.ndarray]:
        pairs, index = [], []
        for doc in docs:
            for i in range(doc.n_pairs):
                pairs.append((doc.paragraphs[i], doc.paragraphs[i + 1]))
                index.append(doc.id)
        scores = self.pair_scores(pairs, batch_size)
        out: Dict[str, List[float]] = {}
        for pid, s in zip(index, scores):
            out.setdefault(pid, []).append(float(s))
        return {pid: np.asarray(v) for pid, v in out.items()}


def scores_to_changes(scores: Dict[str, np.ndarray], threshold: float) -> Dict[str, List[int]]:
    return {pid: (s < threshold).astype(int).tolist() for pid, s in scores.items()}


def tune_threshold(docs: Sequence[Document], scores: Dict[str, np.ndarray],
                   grid: Sequence[float] = tuple(np.round(np.arange(0.05, 1.0, 0.05), 2))) -> Tuple[float, float]:
    """Similarity threshold that maximises macro F1 on ``docs``."""
    best = max(((t, evaluate(docs, scores_to_changes(scores, t))["f1_macro"]) for t in grid), key=lambda x: x[1])
    return float(best[0]), float(best[1])


def train(cfg: SBertConfig, train_pairs: Sequence[Pair], val_docs: Sequence[Document]) -> Dict:
    """Contrastive fine-tuning; the checkpoint with the best validation macro F1 is saved to ``best/``."""
    import torch
    from sentence_transformers import InputExample, losses
    from torch.utils.data import DataLoader

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    encoder = StyleEncoder(cfg.model, cfg.chunk_tokens)
    loss = losses.ContrastiveLoss(model=encoder.model, margin=cfg.margin)
    # ContrastiveLoss expects 1 for similar pairs, i.e. same author.
    examples = [InputExample(texts=[p.text1, p.text2], label=p.same_author) for p in train_pairs]

    out = Path(cfg.output_dir)
    history, best_f1 = [], -1.0
    for epoch in range(cfg.epochs):
        sample = random.sample(examples, int(len(examples) * cfg.subsample))
        loader = DataLoader(sample, shuffle=True, batch_size=cfg.batch_size)
        encoder.model.fit(train_objectives=[(loader, loss)], epochs=1, warmup_steps=cfg.warmup_steps,
                          optimizer_params={"lr": cfg.learning_rate}, show_progress_bar=True)
        if (epoch + 1) % cfg.eval_every and epoch + 1 != cfg.epochs:
            continue
        scores = encoder.document_scores(val_docs, cfg.batch_size)
        metrics = evaluate(val_docs, scores_to_changes(scores, cfg.threshold))
        threshold, tuned_f1 = tune_threshold(val_docs, scores)
        record = {"epoch": epoch + 1, **metrics, "tuned_threshold": threshold, "tuned_f1_macro": tuned_f1}
        history.append(record)
        print(record)
        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            encoder.model.save(str(out / "best"))
    return {"best_f1_macro": best_f1, "history": history}
