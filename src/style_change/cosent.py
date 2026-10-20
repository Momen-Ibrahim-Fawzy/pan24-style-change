"""CL-MAWSA baseline (Chen et al., PAN 2023): CoSENT encoder training, then a frozen-encoder classifier.

Stage 1 trains a transformer encoder (DeBERTa-base) with the CoSENT ranking loss

    L = log(1 + sum_{(i,j) pos, (k,l) neg} exp(lambda * (cos(u_k, u_l) - cos(u_i, u_j))))

which pushes every same-author pair in a batch above every different-author pair in cosine similarity.
Stage 2 freezes the encoder, embeds each paragraph by its [CLS] vector, and trains a tanh-activated
classifier on the pair features [u; v; |u - v|] with cross-entropy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import evaluate
from .pairs import Pair
from .pan import Document


def cosent_loss(cos_sim: torch.Tensor, same_author: torch.Tensor, scale: float = 20.0) -> torch.Tensor:
    """CoSENT loss over a batch of pair similarities; ``same_author`` is 1 for positive pairs."""
    diff = scale * (cos_sim[None, :] - cos_sim[:, None])  # diff[i, j] = s_j - s_i
    mask = (same_author[:, None] > same_author[None, :]).float()  # i positive, j negative
    diff = diff - (1 - mask) * 1e12
    diff = torch.cat([torch.zeros(1, device=diff.device, dtype=diff.dtype), diff.flatten()])
    return torch.logsumexp(diff, dim=0)


class PairClassifier(nn.Module):
    def __init__(self, dim: int, hidden: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        hidden = hidden or dim
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(3 * dim, hidden), nn.Tanh(),
                                 nn.Dropout(dropout), nn.Linear(hidden, 2))

    @staticmethod
    def features(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.cat([u, v, (u - v).abs()], dim=-1)

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.net(self.features(u, v))


@dataclass
class CoSENTConfig:
    model: str = "microsoft/deberta-base"
    output_dir: str = "runs/cl_mawsa"
    max_length: int = 256
    encoder_batch_size: int = 16
    encoder_epochs: int = 3
    encoder_lr: float = 2e-5
    scale: float = 20.0
    classifier_batch_size: int = 128
    classifier_epochs: int = 10
    classifier_lr: float = 1e-3
    seed: int = 42


class CLSEncoder(nn.Module):
    def __init__(self, name: str, max_length: int = 256):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name)
        self.max_length = max_length

    @property
    def dim(self) -> int:
        return self.model.config.hidden_size

    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        batch = self.tokenizer(list(texts), truncation=True, padding=True, max_length=self.max_length,
                               return_tensors="pt").to(device)
        return self.model(**batch).last_hidden_state[:, 0]

    @torch.no_grad()
    def embed(self, texts: Sequence[str], batch_size: int = 64) -> torch.Tensor:
        self.eval()
        out = [self(texts[i:i + batch_size]).float().cpu() for i in range(0, len(texts), batch_size)]
        return torch.cat(out) if out else torch.zeros(0, self.dim)


def _batches(items: Sequence, size: int, rng: np.random.Generator):
    order = rng.permutation(len(items))
    for i in range(0, len(order), size):
        yield [items[j] for j in order[i:i + size]]


def train_encoder(cfg: CoSENTConfig, pairs: Sequence[Pair], device: str) -> CLSEncoder:
    encoder = CLSEncoder(cfg.model, cfg.max_length).to(device)
    optim = torch.optim.AdamW(encoder.parameters(), lr=cfg.encoder_lr)
    rng = np.random.default_rng(cfg.seed)
    for epoch in range(cfg.encoder_epochs):
        encoder.train()
        losses = []
        for batch in _batches(pairs, cfg.encoder_batch_size, rng):
            u = encoder([p.text1 for p in batch])
            v = encoder([p.text2 for p in batch])
            labels = torch.tensor([p.same_author for p in batch], device=device)
            loss = cosent_loss(F.cosine_similarity(u, v), labels, cfg.scale)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())
        print(f"encoder epoch {epoch + 1}: CoSENT loss {np.mean(losses):.4f}")
    return encoder


def _embed_pairs(encoder: CLSEncoder, pairs: Sequence[Pair], batch_size: int):
    texts = sorted({t for p in pairs for t in (p.text1, p.text2)})
    index = {t: i for i, t in enumerate(texts)}
    emb = encoder.embed(texts, batch_size)
    u = emb[[index[p.text1] for p in pairs]]
    v = emb[[index[p.text2] for p in pairs]]
    return u, v


def train_classifier(cfg: CoSENTConfig, encoder: CLSEncoder, train_pairs: Sequence[Pair],
                     val_docs: Sequence[Document], device: str) -> Dict:
    """Stage 2 on frozen embeddings; keeps the epoch with the best validation macro F1."""
    from .pairs import build_pairs

    torch.manual_seed(cfg.seed)
    u_tr, v_tr = _embed_pairs(encoder, train_pairs, cfg.encoder_batch_size * 4)
    y_tr = torch.tensor([p.change for p in train_pairs])
    val_pairs = build_pairs(val_docs, "consecutive")
    u_va, v_va = _embed_pairs(encoder, val_pairs, cfg.encoder_batch_size * 4)

    head = PairClassifier(encoder.dim).to(device)
    optim = torch.optim.Adam(head.parameters(), lr=cfg.classifier_lr)
    best = {"f1_macro": -1.0}
    for epoch in range(cfg.classifier_epochs):
        head.train()
        order = torch.randperm(len(y_tr))
        for i in range(0, len(order), cfg.classifier_batch_size):
            idx = order[i:i + cfg.classifier_batch_size]
            logits = head(u_tr[idx].to(device), v_tr[idx].to(device))
            loss = F.cross_entropy(logits, y_tr[idx].to(device))
            optim.zero_grad()
            loss.backward()
            optim.step()
        head.eval()
        with torch.no_grad():
            pred = head(u_va.to(device), v_va.to(device)).argmax(-1).cpu().tolist()
        predictions: Dict[str, List[int]] = {}
        for p, y in zip(val_pairs, pred):
            predictions.setdefault(p.doc_id, []).append(int(y))
        metrics = evaluate(val_docs, predictions)
        print(f"classifier epoch {epoch + 1}: validation F1 {metrics['f1_macro']:.4f}")
        if metrics["f1_macro"] > best["f1_macro"]:
            best = {"epoch": epoch + 1, **metrics}
            out = Path(cfg.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            torch.save(head.state_dict(), out / "classifier.pt")
    return best
