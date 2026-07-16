"""OpenAI embedding provider (optional; requires `pip install .[openai]` and
OPENAI_API_KEY). Selected via RAGQA_EMBEDDINGS=openai."""

from __future__ import annotations

import numpy as np

from .base import EmbeddingProvider

_BATCH = 128


class OpenAIEmbedder(EmbeddingProvider):
    name = "openai"

    def __init__(self, model: str = "text-embedding-3-small", dim: int = 1536):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "OpenAI embeddings require the openai package: pip install 'ragqa[openai]' "
                "and set OPENAI_API_KEY"
            ) from exc
        self._client = OpenAI()
        self.model = model
        self.dim = dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - needs API key
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for start in range(0, len(texts), _BATCH):
            batch = [t[:32000] if t.strip() else " " for t in texts[start : start + _BATCH]]
            resp = self._client.embeddings.create(model=self.model, input=batch, dimensions=self.dim)
            for i, item in enumerate(resp.data):
                out[start + i] = np.asarray(item.embedding, dtype=np.float32)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms
