"""VectorStore interface + store factory (SQLite default, pgvector optional)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..models import Chunk


class VectorStore(ABC):
    """Persistence + nearest-neighbor search for chunks and their embeddings."""

    @abstractmethod
    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> list[int]:
        """Persist chunks with row-aligned embeddings; returns assigned ids."""

    @abstractmethod
    def get_chunks(self, repo: str) -> list[Chunk]:
        """All chunks of a repo (ids populated), in insertion order."""

    @abstractmethod
    def search(self, repo: str, query_vec: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        """Top-k chunks by cosine similarity against the (normalized) query."""

    @abstractmethod
    def delete_repo(self, repo: str) -> None: ...

    @abstractmethod
    def upsert_repo_meta(self, name: str, meta: dict[str, Any]) -> None: ...

    @abstractmethod
    def list_repos(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    def repo_version(self, repo: str) -> tuple:
        """Cheap change token: differs whenever a repo's chunks change."""

    def close(self) -> None:  # pragma: no cover - trivial default
        pass


def get_store() -> VectorStore:
    """DATABASE_URL set (postgres://...) -> pgvector store, else SQLite."""
    from .. import config

    url = config.database_url()
    if url.startswith(("postgres://", "postgresql://")):
        from .pgvector_store import PgVectorStore

        return PgVectorStore(url)
    from .sqlite_store import SqliteVectorStore

    return SqliteVectorStore(config.db_path())
