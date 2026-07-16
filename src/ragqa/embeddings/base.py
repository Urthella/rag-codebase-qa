"""EmbeddingProvider interface + provider factory."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    """Turns texts into L2-normalized float32 vectors of a fixed dimension."""

    name: str = "base"
    dim: int = 0

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array, rows L2-normalized."""

    def embed_query(self, text: str) -> np.ndarray:
        """Default: queries are embedded like documents."""
        return self.embed_texts([text])[0]


def get_embedder(provider: str | None = None) -> EmbeddingProvider:
    """Build the embedding provider selected by RAGQA_EMBEDDINGS (default: local).

    - ``local``  : deterministic hashed n-gram embedder, zero dependencies/keys
    - ``openai`` : text-embedding-3-small (requires OPENAI_API_KEY + `pip install .[openai]`)
    - ``sbert``  : sentence-transformers (requires `pip install .[sbert]`, pulls torch)
    """
    from .. import config

    provider = (provider or config.embeddings_provider()).lower()
    if provider == "local":
        from .local_hash import HashedNGramEmbedder

        return HashedNGramEmbedder(dim=config.embedding_dim())
    if provider == "openai":
        from .openai_embed import OpenAIEmbedder

        return OpenAIEmbedder(model=config.openai_embedding_model())
    if provider == "sbert":
        from .sbert_embed import SbertEmbedder

        return SbertEmbedder(model_name=config.sbert_model())
    raise ValueError(f"Unknown embeddings provider: {provider!r} (expected local|openai|sbert)")
