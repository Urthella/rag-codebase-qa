"""Deterministic local embedding provider -- zero keys, zero network, pure numpy.

Feature-hashing ("hashing trick") embedder:

* word/identifier tokens (camelCase and snake_case split into subtokens),
  weighted by ``log(1 + tf)``
* character 3-grams of the lowercased text (robust to typos and partial
  identifiers), at half weight

Each feature is hashed with BLAKE2b (stable across processes and platforms --
unlike Python's salted ``hash()``) onto one of ``dim`` buckets with a +/-1 sign,
then the vector is L2-normalized. Cosine similarity between such vectors
approximates weighted token-overlap similarity, which works well for code
search where identifier overlap dominates. It is NOT a semantic embedding --
use the openai/sbert providers for that -- but it makes the whole pipeline run
offline and is deterministic for tests and evals.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter

import numpy as np

from ..retrieval.tokenize import code_tokens
from .base import EmbeddingProvider

_NGRAM = 3
_NGRAM_WEIGHT = 0.5


def _stable_hash(feature: str) -> int:
    return int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big")


class HashedNGramEmbedder(EmbeddingProvider):
    name = "local-hash"

    def __init__(self, dim: int = 384):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _features(self, text: str) -> Counter:
        feats: Counter = Counter()
        for tok in code_tokens(text):
            feats[f"w:{tok}"] += 1.0
        squashed = "".join(ch if ch.isalnum() else " " for ch in text.lower())
        squashed = " ".join(squashed.split())
        for i in range(len(squashed) - _NGRAM + 1):
            gram = squashed[i : i + _NGRAM]
            if gram.strip():
                feats[f"g:{gram}"] += _NGRAM_WEIGHT
        return feats

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature, tf in self._features(text).items():
                h = _stable_hash(feature)
                idx = h % self.dim
                sign = 1.0 if (h >> 63) & 1 else -1.0
                out[row, idx] += sign * math.log1p(tf)
            norm = float(np.linalg.norm(out[row]))
            if norm > 0:
                out[row] /= norm
        return out
