"""Hybrid retrieval: dense vector similarity + BM25, fused with RRF.

Reciprocal Rank Fusion (RRF) merges the two rankings without having to
calibrate score scales against each other:

    fused(d) = sum over rankings r of  1 / (K + rank_r(d))     (K = 60)

A weighted-score fusion (min-max normalized, alpha * vector + (1-alpha) * bm25)
is also provided for comparison via ``fusion="weighted"``.
"""

from __future__ import annotations

from ..embeddings.base import EmbeddingProvider
from ..models import Chunk, ScoredChunk
from ..store.base import VectorStore
from .bm25 import BM25Index
from .tokenize import code_tokens

RRF_K = 60


def rrf_fuse(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Fuse rankings (lists of doc ids, best first) into {doc_id: rrf_score}."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


def _minmax(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-12:
        return {d: 1.0 for d in scores}
    return {d: (s - lo) / (hi - lo) for d, s in scores.items()}


class HybridSearcher:
    """Per-repo searcher. Caches the BM25 index and chunk list; call
    ``refresh_if_stale()`` (cheap) before searching to pick up re-indexes."""

    def __init__(self, store: VectorStore, embedder: EmbeddingProvider, repo: str):
        self.store = store
        self.embedder = embedder
        self.repo = repo
        self._version: tuple | None = None
        self._chunks: list[Chunk] = []
        self._by_id: dict[int, int] = {}
        self._bm25: BM25Index | None = None

    # -- index maintenance -------------------------------------------------
    def refresh_if_stale(self) -> None:
        version = self.store.repo_version(self.repo)
        if version == self._version and self._bm25 is not None:
            return
        self._chunks = self.store.get_chunks(self.repo)
        self._by_id = {c.id: i for i, c in enumerate(self._chunks)}
        docs = [self._bm25_doc(c) for c in self._chunks]
        self._bm25 = BM25Index(docs)
        self._version = version

    @staticmethod
    def _bm25_doc(chunk: Chunk) -> list[str]:
        parts = [
            chunk.text,
            chunk.symbol,
            chunk.doc,
            chunk.path.replace("/", " ").replace(".", " "),
        ]
        return code_tokens("\n".join(p for p in parts if p))

    # -- search ------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 8,
        mode: str = "hybrid",
        fusion: str = "rrf",
        alpha: float = 0.5,
    ) -> list[ScoredChunk]:
        self.refresh_if_stale()
        if not self._chunks:
            return []
        n_cand = max(top_k * 5, 50)

        vector_scores: dict[int, float] = {}
        bm25_scores: dict[int, float] = {}

        if mode in ("vector", "hybrid"):
            qvec = self.embedder.embed_query(query)
            for chunk, score in self.store.search(self.repo, qvec, n_cand):
                vector_scores[chunk.id] = score
        if mode in ("bm25", "hybrid"):
            assert self._bm25 is not None
            for doc_idx, score in self._bm25.top_n(code_tokens(query), n_cand):
                bm25_scores[self._chunks[doc_idx].id] = score

        if mode == "vector":
            order = sorted(vector_scores, key=vector_scores.get, reverse=True)
            fused = dict(vector_scores)
        elif mode == "bm25":
            order = sorted(bm25_scores, key=bm25_scores.get, reverse=True)
            fused = dict(bm25_scores)
        elif mode == "hybrid":
            vec_rank = sorted(vector_scores, key=vector_scores.get, reverse=True)
            bm_rank = sorted(bm25_scores, key=bm25_scores.get, reverse=True)
            if fusion == "weighted":
                nv, nb = _minmax(vector_scores), _minmax(bm25_scores)
                fused = {
                    d: alpha * nv.get(d, 0.0) + (1 - alpha) * nb.get(d, 0.0)
                    for d in set(nv) | set(nb)
                }
            else:
                fused = rrf_fuse([vec_rank, bm_rank])
            order = sorted(fused, key=fused.get, reverse=True)
        else:
            raise ValueError(f"Unknown search mode: {mode!r} (expected vector|bm25|hybrid)")

        results: list[ScoredChunk] = []
        for rank, chunk_id in enumerate(order[:top_k], start=1):
            chunk = self._chunks[self._by_id[chunk_id]]
            results.append(
                ScoredChunk(
                    chunk=chunk,
                    vector_score=vector_scores.get(chunk_id, 0.0),
                    bm25_score=bm25_scores.get(chunk_id, 0.0),
                    fused_score=fused.get(chunk_id, 0.0),
                    rank=rank,
                )
            )
        return results


class SearcherRegistry:
    """Cache of HybridSearcher per repo (BM25 index build is not free)."""

    def __init__(self, store: VectorStore, embedder: EmbeddingProvider):
        self.store = store
        self.embedder = embedder
        self._searchers: dict[str, HybridSearcher] = {}

    def get(self, repo: str) -> HybridSearcher:
        if repo not in self._searchers:
            self._searchers[repo] = HybridSearcher(self.store, self.embedder, repo)
        return self._searchers[repo]
