"""Zero-shot style-change detection with chat LLMs.

The model receives the whole document as a list of paragraphs and must answer with
``{"changes": [...]}`` holding exactly ``n_paragraphs - 1`` zeros and ones. The prompt was developed on
50 training documents per difficulty and then run unchanged on the validation sets.

LLM answers are not always well-formed: extra prose around the JSON, a second JSON object, labels as
strings, or a list of the wrong length. ``parse_changes`` recovers what can be recovered and returns
``None`` otherwise. Unanswered documents are counted (coverage) and scored with the fallback label.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .pan import Document

SYSTEM_PROMPT = """You are a natural language expert who detects changes of writing style within a list of paragraphs.

Instructions:
1. Identify the positions in a multi-author list of paragraphs where the author's writing style changes.
2. For each pair of consecutive paragraphs, decide whether the writing style changes between them.
3. Answer in JSON with a single key, "changes". Its value is a list of length num_paragraphs - 1 where each element is
   - 0 if there is no change in writing style between the two paragraphs, or
   - 1 if there is a change in writing style between the two paragraphs.
4. The list must contain only zeros and ones, and its length must be exactly num_paragraphs - 1.
5. Output format: {"changes": [0/1, 0/1, ...]}"""

USER_TEMPLATE = """Annotate the following document with its style-change labels.
The list of labels must have exactly {n_labels} elements.
Document paragraphs: {paragraphs}"""

# OpenAI-compatible endpoints. The open models were served through Perplexity's API during the project;
# any OpenAI-compatible server (vLLM, Together, a local gateway) works with --base-url.
PROVIDERS = {
    "openai": {"base_url": None, "api_key_env": "OPENAI_API_KEY"},
    "perplexity": {"base_url": "https://api.perplexity.ai", "api_key_env": "PERPLEXITY_API_KEY"},
}


def build_messages(paragraphs: Sequence[str]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(n_labels=len(paragraphs) - 1,
                                                         paragraphs=json.dumps(list(paragraphs), ensure_ascii=False))},
    ]


_LIST = re.compile(r"""['"]?changes['"]?\s*[:=]\s*\[([^\]]*)\]""", re.IGNORECASE)


def _to_binary(values) -> Optional[List[int]]:
    out = []
    for v in values:
        if isinstance(v, bool):
            v = int(v)
        if isinstance(v, str):
            v = v.strip().strip("'\"")
        try:
            v = int(v)
        except (TypeError, ValueError):
            return None
        if v not in (0, 1):
            return None
        out.append(v)
    return out


def parse_changes(text: str, n_labels: int) -> Optional[List[int]]:
    """Extract a valid ``changes`` list of length ``n_labels`` from an LLM answer, or return None."""
    if not text:
        return None
    decoder = json.JSONDecoder()
    for start in (m.start() for m in re.finditer(r"\{", text)):
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("changes"), list):
            values = _to_binary(obj["changes"])
            if values is not None and len(values) == n_labels:
                return values
    for m in _LIST.finditer(text):  # JSON-like but not valid JSON
        items = [x for x in m.group(1).split(",") if x.strip()]
        values = _to_binary(items)
        if values is not None and len(values) == n_labels:
            return values
    return None


@dataclass
class LLMAnnotator:
    model: str
    provider: str = "openai"
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_retries: int = 3
    cache_path: Optional[str] = None

    def __post_init__(self):
        self.cache: Dict[str, Optional[List[int]]] = {}
        if self.cache_path and Path(self.cache_path).exists():
            for line in Path(self.cache_path).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.cache[rec["key"]] = rec["changes"]
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            spec = PROVIDERS.get(self.provider, {"base_url": None, "api_key_env": "OPENAI_API_KEY"})
            api_key = os.environ.get(spec["api_key_env"])
            if not api_key:
                raise RuntimeError(f"set {spec['api_key_env']} to call {self.provider}")
            self._client = OpenAI(api_key=api_key, base_url=self.base_url or spec["base_url"])
        return self._client

    def _key(self, doc: Document) -> str:
        return f"{self.model}|{doc.difficulty}|{doc.split}|{doc.id}"

    def complete(self, messages: List[Dict[str, str]]) -> str:
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(model=self.model, messages=messages,
                                                               temperature=self.temperature)
                return response.choices[0].message.content or ""
            except Exception:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2
        return ""

    def annotate(self, doc: Document) -> Optional[List[int]]:
        key = self._key(doc)
        if key in self.cache:
            return self.cache[key]
        try:
            answer = self.complete(build_messages(doc.paragraphs))
            changes = parse_changes(answer, doc.n_pairs)
        except Exception as exc:  # API failure after retries: record as unanswered
            print(f"{key}: {exc}")
            answer, changes = "", None
        self.cache[key] = changes
        if self.cache_path:
            Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "changes": changes, "raw": answer}, ensure_ascii=False) + "\n")
        return changes
