"""sentence-transformers embedding provider (optional extra; pulls torch).

Selected via RAGQA_EMBEDDINGS=sbert. Not installed by default -- install with
`pip install 'ragqa[sbert]'`.
"""

from __future__ import annotations

import numpy as np

from .base import EmbeddingProvider


class SbertEmbedder(EmbeddingProvider):
    name = "sbert"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers embeddings require: pip install 'ragqa[sbert]' "
                "(installs torch)"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - needs torch
        vecs = self._model.encode(
            [t if t.strip() else " " for t in texts],
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)
