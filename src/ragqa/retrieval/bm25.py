"""Okapi BM25 implemented from scratch (no rank-bm25 dependency).

Standard parameters k1=1.5, b=0.75 and the "plus 1" IDF variant
``ln(1 + (N - df + 0.5) / (df + 0.5))`` so IDF is never negative.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np


class BM25Index:
    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = len(documents)
        self.doc_freqs: list[Counter] = [Counter(doc) for doc in documents]
        self.doc_lens = np.array([len(doc) for doc in documents], dtype=np.float64)
        self.avg_len = float(self.doc_lens.mean()) if self.n_docs else 0.0
        df: Counter = Counter()
        for freqs in self.doc_freqs:
            df.update(freqs.keys())
        self.idf: dict[str, float] = {
            term: math.log(1.0 + (self.n_docs - d + 0.5) / (d + 0.5)) for term, d in df.items()
        }
        # inverted index: term -> list of doc ids (for sparse scoring)
        self.postings: dict[str, list[int]] = {}
        for doc_id, freqs in enumerate(self.doc_freqs):
            for term in freqs:
                self.postings.setdefault(term, []).append(doc_id)

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        """BM25 score of every document for the query. O(query * postings)."""
        scores = np.zeros(self.n_docs, dtype=np.float64)
        if self.n_docs == 0 or self.avg_len == 0:
            return scores
        for term, qtf in Counter(query_tokens).items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id in self.postings[term]:
                tf = self.doc_freqs[doc_id][term]
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_lens[doc_id] / self.avg_len)
                scores[doc_id] += idf * (tf * (self.k1 + 1) / denom) * min(qtf, 2)
        return scores

    def top_n(self, query_tokens: list[str], n: int) -> list[tuple[int, float]]:
        scores = self.scores(query_tokens)
        if self.n_docs == 0:
            return []
        n = min(n, self.n_docs)
        idx = np.argpartition(-scores, n - 1)[:n]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > 0.0]
