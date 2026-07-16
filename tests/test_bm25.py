"""BM25 scoring + hybrid fusion sanity tests."""

import numpy as np

from ragqa.retrieval.bm25 import BM25Index
from ragqa.retrieval.hybrid import rrf_fuse
from ragqa.retrieval.tokenize import code_tokens, split_identifier


class TestTokenizer:
    def test_camel_case_split(self):
        assert split_identifier("refillPerSecond") == ["refill", "per", "second"]

    def test_snake_case_split(self):
        assert split_identifier("token_bucket_limiter") == ["token", "bucket", "limiter"]

    def test_code_tokens_keeps_full_identifier_and_parts(self):
        toks = code_tokens("const refillPerSecond = 5;")
        assert "refillpersecond" in toks
        assert "refill" in toks and "second" in toks

    def test_stopwords_removed(self):
        toks = code_tokens("where is the refill implemented")
        assert "the" not in toks and "is" not in toks
        assert "refill" in toks


class TestBM25:
    DOCS = [
        code_tokens("class TokenBucket { refill(elapsed) { tokens += elapsed * refillPerSecond } }"),
        code_tokens("function fixedWindow(count, limit) { return count <= limit }"),
        code_tokens("const slidingWindowLog = timestamps.filter(t => t > cutoff)"),
        code_tokens("export function proxyRequest(req, res) { /* forward upstream */ }"),
    ]

    def test_rare_term_ranks_target_doc_first(self):
        index = BM25Index(self.DOCS)
        top = index.top_n(code_tokens("token bucket refill"), 4)
        assert top, "expected non-empty results"
        assert top[0][0] == 0  # TokenBucket doc

    def test_query_without_matches_scores_zero(self):
        index = BM25Index(self.DOCS)
        assert index.top_n(code_tokens("zebra quantum"), 4) == []

    def test_idf_prefers_rarer_terms(self):
        # 'function' appears in 2 docs, 'sliding' in 1 -> sliding query should
        # score doc 2 higher than a 'function' query scores any doc.
        index = BM25Index(self.DOCS)
        sliding = index.scores(["sliding"])
        func = index.scores(["function"])
        assert sliding.max() > func.max()

    def test_scores_shape_and_dtype(self):
        index = BM25Index(self.DOCS)
        scores = index.scores(code_tokens("limit"))
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (4,)

    def test_empty_corpus(self):
        index = BM25Index([])
        assert index.top_n(["anything"], 5) == []


class TestRRFFusion:
    def test_doc_in_both_rankings_beats_single_ranking_docs(self):
        vector_rank = [10, 20, 30]
        bm25_rank = [10, 40, 50]
        fused = rrf_fuse([vector_rank, bm25_rank])
        assert max(fused, key=fused.get) == 10
        # every doc from either list is present
        assert set(fused) == {10, 20, 30, 40, 50}

    def test_rank_position_matters(self):
        fused = rrf_fuse([[1, 2], [2, 1]])
        # symmetric ranks -> equal scores
        assert abs(fused[1] - fused[2]) < 1e-12
        fused2 = rrf_fuse([[1, 2], [1, 2]])
        assert fused2[1] > fused2[2]

    def test_rrf_formula(self):
        fused = rrf_fuse([[7]], k=60)
        assert abs(fused[7] - 1.0 / 61.0) < 1e-12
