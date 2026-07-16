"""pgvector-backed vector store (optional; selected via DATABASE_URL).

Code-complete adapter implementing the same VectorStore interface against
Postgres + the pgvector extension. Requires `pip install 'ragqa[pg]'` and a
database where `CREATE EXTENSION vector` has been (or can be) run.

NOTE: this adapter is not exercised by the offline test suite (no Postgres in
the default environment); the SQLite store is the reference implementation.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from ..models import Chunk
from .base import VectorStore

_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS repos (
    name TEXT PRIMARY KEY,
    meta JSONB NOT NULL DEFAULT '{{}}'
);
CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    repo TEXT NOT NULL,
    path TEXT NOT NULL,
    language TEXT NOT NULL,
    kind TEXT NOT NULL,
    symbol TEXT NOT NULL DEFAULT '',
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    doc TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL,
    embedding vector({dim}) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_repo ON chunks(repo);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks
    USING hnsw (embedding vector_cosine_ops);
"""


def _row_to_chunk(row) -> Chunk:
    (cid, repo, path, language, kind, symbol, start_line, end_line, doc, text) = row
    return Chunk(
        id=cid,
        repo=repo,
        path=path,
        language=language,
        kind=kind,
        symbol=symbol,
        start_line=start_line,
        end_line=end_line,
        doc=doc,
        text=text,
    )


_CHUNK_COLS = "id, repo, path, language, kind, symbol, start_line, end_line, doc, text"


class PgVectorStore(VectorStore):
    def __init__(self, database_url: str, dim: int | None = None):
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "pgvector store requires: pip install 'ragqa[pg]' "
                "(psycopg[binary] + pgvector)"
            ) from exc
        from .. import config

        self.dim = dim or config.embedding_dim()
        self._conn = psycopg.connect(database_url, autocommit=True)
        with self._conn.cursor() as cur:
            cur.execute(_SCHEMA.format(dim=self.dim))
        register_vector(self._conn)

    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> list[int]:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be row-aligned")
        ids: list[int] = []
        with self._conn.cursor() as cur:
            for chunk, vec in zip(chunks, embeddings):
                cur.execute(
                    """INSERT INTO chunks
                       (repo, path, language, kind, symbol, start_line, end_line, doc, text, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                    (
                        chunk.repo,
                        chunk.path,
                        chunk.language,
                        chunk.kind,
                        chunk.symbol,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.doc,
                        chunk.text,
                        np.asarray(vec, dtype=np.float32),
                    ),
                )
                new_id = cur.fetchone()[0]
                chunk.id = new_id
                ids.append(new_id)
        return ids

    def get_chunks(self, repo: str) -> list[Chunk]:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT {_CHUNK_COLS} FROM chunks WHERE repo = %s ORDER BY id", (repo,)
            )
            return [_row_to_chunk(r) for r in cur.fetchall()]

    def search(self, repo: str, query_vec: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        vec = np.asarray(query_vec, dtype=np.float32)
        with self._conn.cursor() as cur:
            cur.execute(
                f"""SELECT {_CHUNK_COLS}, 1 - (embedding <=> %s) AS score
                    FROM chunks WHERE repo = %s
                    ORDER BY embedding <=> %s LIMIT %s""",
                (vec, repo, vec, top_k),
            )
            return [(_row_to_chunk(r[:-1]), float(r[-1])) for r in cur.fetchall()]

    def delete_repo(self, repo: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE repo = %s", (repo,))
            cur.execute("DELETE FROM repos WHERE name = %s", (repo,))

    def upsert_repo_meta(self, name: str, meta: dict[str, Any]) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """INSERT INTO repos(name, meta) VALUES(%s, %s)
                   ON CONFLICT (name) DO UPDATE SET meta = EXCLUDED.meta""",
                (name, json.dumps(meta)),
            )

    def list_repos(self) -> list[dict[str, Any]]:
        with self._conn.cursor() as cur:
            cur.execute(
                """SELECT r.name, r.meta, COUNT(c.id)
                   FROM repos r LEFT JOIN chunks c ON c.repo = r.name
                   GROUP BY r.name, r.meta ORDER BY r.name"""
            )
            out = []
            for name, meta, n_chunks in cur.fetchall():
                info = meta if isinstance(meta, dict) else json.loads(meta)
                info.update({"name": name, "num_chunks": n_chunks})
                out.append(info)
            return out

    def repo_version(self, repo: str) -> tuple:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*), COALESCE(MAX(id), 0) FROM chunks WHERE repo = %s", (repo,)
            )
            return tuple(cur.fetchone())

    def close(self) -> None:
        self._conn.close()
