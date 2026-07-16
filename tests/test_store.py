"""SQLite vector store round-trip + search tests."""

import numpy as np

from ragqa.models import Chunk
from ragqa.retrieval.hybrid import HybridSearcher


def make_chunk(repo, path, text, symbol="", start=1, end=5, kind="function"):
    return Chunk(
        repo=repo,
        path=path,
        language="python",
        kind=kind,
        symbol=symbol,
        start_line=start,
        end_line=end,
        text=text,
        doc="",
    )


class TestSqliteRoundTrip:
    def test_add_and_get_preserves_metadata(self, tmp_store, embedder):
        chunks = [
            make_chunk("r1", "a.py", "def alpha():\n    return 1", symbol="alpha", start=3, end=4),
            make_chunk("r1", "b.py", "def beta():\n    return 2", symbol="beta", start=10, end=11),
        ]
        vecs = embedder.embed_texts([c.text for c in chunks])
        ids = tmp_store.add_chunks(chunks, vecs)
        assert len(ids) == 2 and ids[0] != ids[1]
        assert chunks[0].id == ids[0]

        loaded = tmp_store.get_chunks("r1")
        assert len(loaded) == 2
        first = loaded[0]
        assert (first.path, first.symbol, first.start_line, first.end_line) == ("a.py", "alpha", 3, 4)
        assert first.text == "def alpha():\n    return 1"

    def test_embedding_roundtrip_bitexact(self, tmp_store, embedder):
        chunk = make_chunk("r1", "a.py", "def gamma(): pass", symbol="gamma")
        vec = embedder.embed_texts([chunk.text])
        tmp_store.add_chunks([chunk], vec)
        # search with the stored vector itself must return similarity ~1.0
        results = tmp_store.search("r1", vec[0], top_k=1)
        assert len(results) == 1
        top_chunk, score = results[0]
        assert top_chunk.symbol == "gamma"
        assert abs(score - 1.0) < 1e-5

    def test_search_ranks_similar_text_first(self, tmp_store, embedder):
        chunks = [
            make_chunk("r1", "bucket.py", "class TokenBucket: def refill(self): tokens += rate", "TokenBucket"),
            make_chunk("r1", "window.py", "def fixed_window(count, limit): return count <= limit", "fixed_window"),
            make_chunk("r1", "proxy.py", "def proxy_request(req): forward(req)", "proxy_request"),
        ]
        vecs = embedder.embed_texts([c.text for c in chunks])
        tmp_store.add_chunks(chunks, vecs)
        qvec = embedder.embed_query("token bucket refill rate")
        results = tmp_store.search("r1", qvec, top_k=3)
        assert results[0][0].path == "bucket.py"
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_delete_repo_and_isolation(self, tmp_store, embedder):
        c1 = make_chunk("r1", "a.py", "def a(): pass", "a")
        c2 = make_chunk("r2", "b.py", "def b(): pass", "b")
        vecs = embedder.embed_texts([c1.text, c2.text])
        tmp_store.add_chunks([c1], vecs[:1])
        tmp_store.add_chunks([c2], vecs[1:])
        tmp_store.delete_repo("r1")
        assert tmp_store.get_chunks("r1") == []
        assert len(tmp_store.get_chunks("r2")) == 1

    def test_repo_meta_and_version(self, tmp_store, embedder):
        v0 = tmp_store.repo_version("r1")
        chunk = make_chunk("r1", "a.py", "x = 1", kind="module")
        tmp_store.add_chunks([chunk], embedder.embed_texts([chunk.text]))
        assert tmp_store.repo_version("r1") != v0
        tmp_store.upsert_repo_meta("r1", {"source": "/tmp/x", "num_files": 1})
        repos = tmp_store.list_repos()
        assert repos[0]["name"] == "r1"
        assert repos[0]["num_chunks"] == 1
        assert repos[0]["source"] == "/tmp/x"


class TestHybridSearcherOverStore:
    def test_hybrid_beats_or_matches_singles_on_obvious_query(self, tmp_store, embedder):
        chunks = [
            make_chunk(
                "r1",
                "src/tokenBucket.ts",
                "export class TokenBucket { check() { this.tokens += elapsedMs * this.refillPerSecond / 1000 } }",
                "TokenBucket",
            ),
            make_chunk(
                "r1",
                "src/fixedWindow.ts",
                "export class FixedWindow { check() { return this.count <= this.limit } }",
                "FixedWindow",
            ),
            make_chunk(
                "r1",
                "src/proxy.ts",
                "export function proxyRequest(req, res) { forward(req, res) }",
                "proxyRequest",
            ),
        ]
        vecs = embedder.embed_texts([c.text for c in chunks])
        tmp_store.add_chunks(chunks, vecs)

        searcher = HybridSearcher(tmp_store, embedder, "r1")
        for mode in ("vector", "bm25", "hybrid"):
            results = searcher.search("where is the token bucket refill", top_k=3, mode=mode)
            assert results, f"no results for mode {mode}"
            assert results[0].chunk.path == "src/tokenBucket.ts", f"mode {mode} missed"

        hybrid = searcher.search("token bucket refill", top_k=3, mode="hybrid")
        top = hybrid[0]
        assert top.fused_score > 0
        assert top.vector_score > 0 and top.bm25_score > 0  # found by both signals

    def test_refresh_picks_up_new_chunks(self, tmp_store, embedder):
        searcher = HybridSearcher(tmp_store, embedder, "r1")
        assert searcher.search("anything", mode="hybrid") == []
        chunk = make_chunk("r1", "new.py", "def brand_new_symbol(): pass", "brand_new_symbol")
        tmp_store.add_chunks([chunk], embedder.embed_texts([chunk.text]))
        results = searcher.search("brand new symbol", mode="hybrid")
        assert results and results[0].chunk.path == "new.py"

    def test_weighted_fusion_mode(self, tmp_store, embedder):
        chunk = make_chunk("r1", "a.py", "def weighted_target(): pass", "weighted_target")
        tmp_store.add_chunks([chunk], embedder.embed_texts([chunk.text]))
        searcher = HybridSearcher(tmp_store, embedder, "r1")
        results = searcher.search("weighted target", mode="hybrid", fusion="weighted")
        assert results and results[0].chunk.symbol == "weighted_target"
