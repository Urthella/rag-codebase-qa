"""SQLite-backed vector store.

Embeddings are stored as float32 BLOBs; similarity search is brute-force
cosine via a numpy matrix-vector product. The embedding matrix per repo is
cached in memory and invalidated with a cheap (count, max_id) version token,
so repeated queries don't re-read BLOBs. Fine for corpora up to ~100k chunks;
beyond that, switch to the pgvector store (DATABASE_URL).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

import numpy as np

from ..models import Chunk
from .base import VectorStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS repos (
    name TEXT PRIMARY KEY,
    meta TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo TEXT NOT NULL,
    path TEXT NOT NULL,
    language TEXT NOT NULL,
    kind TEXT NOT NULL,
    symbol TEXT NOT NULL DEFAULT '',
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    doc TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_repo ON chunks(repo);
"""


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        id=row["id"],
        repo=row["repo"],
        path=row["path"],
        language=row["language"],
        kind=row["kind"],
        symbol=row["symbol"],
        start_line=row["start_line"],
        end_line=row["end_line"],
        doc=row["doc"],
        text=row["text"],
    )


class SqliteVectorStore(VectorStore):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        # repo -> (version, ids array, matrix)
        self._matrix_cache: dict[str, tuple[tuple, np.ndarray, np.ndarray]] = {}

    # -- writes --------------------------------------------------------------
    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> list[int]:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be row-aligned")
        ids: list[int] = []
        with self._lock:
            cur = self._conn.cursor()
            for chunk, vec in zip(chunks, embeddings):
                cur.execute(
                    """INSERT INTO chunks
                       (repo, path, language, kind, symbol, start_line, end_line, doc, text, embedding)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                        np.asarray(vec, dtype=np.float32).tobytes(),
                    ),
                )
                chunk.id = cur.lastrowid
                ids.append(cur.lastrowid)
            self._conn.commit()
        return ids

    def delete_repo(self, repo: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM chunks WHERE repo = ?", (repo,))
            self._conn.execute("DELETE FROM repos WHERE name = ?", (repo,))
            self._conn.commit()
        self._matrix_cache.pop(repo, None)

    def upsert_repo_meta(self, name: str, meta: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO repos(name, meta) VALUES(?, ?) "
                "ON CONFLICT(name) DO UPDATE SET meta = excluded.meta",
                (name, json.dumps(meta)),
            )
            self._conn.commit()

    # -- reads ---------------------------------------------------------------
    def list_repos(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT r.name, r.meta, COUNT(c.id) AS n_chunks
                   FROM repos r LEFT JOIN chunks c ON c.repo = r.name
                   GROUP BY r.name ORDER BY r.name"""
            ).fetchall()
        out = []
        for row in rows:
            meta = json.loads(row["meta"])
            meta.update({"name": row["name"], "num_chunks": row["n_chunks"]})
            out.append(meta)
        return out

    def get_chunks(self, repo: str) -> list[Chunk]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM chunks WHERE repo = ? ORDER BY id", (repo,)
            ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def repo_version(self, repo: str) -> tuple:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(MAX(id), 0) AS m FROM chunks WHERE repo = ?",
                (repo,),
            ).fetchone()
        return (row["n"], row["m"])

    # -- similarity search -----------------------------------------------------
    def _matrix(self, repo: str) -> tuple[np.ndarray, np.ndarray]:
        version = self.repo_version(repo)
        cached = self._matrix_cache.get(repo)
        if cached is not None and cached[0] == version:
            return cached[1], cached[2]
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, embedding FROM chunks WHERE repo = ? ORDER BY id", (repo,)
            ).fetchall()
        if not rows:
            ids = np.zeros(0, dtype=np.int64)
            matrix = np.zeros((0, 1), dtype=np.float32)
        else:
            ids = np.array([r["id"] for r in rows], dtype=np.int64)
            matrix = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        self._matrix_cache[repo] = (version, ids, matrix)
        return ids, matrix

    def search(self, repo: str, query_vec: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        ids, matrix = self._matrix(repo)
        if len(ids) == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        sims = matrix @ q  # rows and q are L2-normalized -> cosine similarity
        top_k = min(top_k, len(ids))
        idx = np.argpartition(-sims, top_k - 1)[:top_k]
        idx = idx[np.argsort(-sims[idx])]
        wanted = [int(ids[i]) for i in idx]
        placeholders = ",".join("?" * len(wanted))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM chunks WHERE id IN ({placeholders})", wanted
            ).fetchall()
        by_id = {r["id"]: _row_to_chunk(r) for r in rows}
        return [(by_id[ids[i]], float(sims[i])) for i in idx]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
