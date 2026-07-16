"""Extractive answer provider: citations present, 'I don't know' behavior."""

import re

from ragqa.answer.extractive import ExtractiveProvider
from ragqa.answer.prompts import I_DONT_KNOW, extract_citations
from ragqa.models import Chunk, ScoredChunk


def scored(path, text, symbol="", start=10, end=20, doc="", **scores):
    chunk = Chunk(
        repo="r",
        path=path,
        language="typescript",
        kind="method",
        symbol=symbol,
        start_line=start,
        end_line=end,
        text=text,
        doc=doc,
    )
    return ScoredChunk(chunk=chunk, **scores)


class TestExtractiveProvider:
    def test_answer_contains_citation_and_matching_lines(self):
        provider = ExtractiveProvider()
        chunks = [
            scored(
                "src/limiters/memory/tokenBucket.ts",
                "async check(key) {\n  bucket.tokens = Math.min(capacity, bucket.tokens + elapsedMs * refillPerSecond / 1000);\n}",
                symbol="TokenBucketMemoryLimiter.check",
                start=28,
                end=57,
                doc="Refill is fractional.",
                fused_score=0.03,
                vector_score=0.4,
                bm25_score=7.0,
            )
        ]
        answer = provider.answer("Where is the token bucket refill implemented?", chunks)
        assert answer.grounded
        assert "[src/limiters/memory/tokenBucket.ts:28-57]" in answer.text
        assert len(answer.citations) == 1
        cite = answer.citations[0]
        assert (cite.path, cite.start_line, cite.end_line) == (
            "src/limiters/memory/tokenBucket.ts", 28, 57,
        )
        # the refill line is surfaced in the excerpt
        assert "refillPerSecond" in answer.text

    def test_i_dont_know_when_nothing_matches(self):
        provider = ExtractiveProvider()
        chunks = [scored("src/unrelated.ts", "const zebra = 1;", symbol="zebra")]
        answer = provider.answer("Where is quantum entanglement frobnicated?", chunks)
        assert not answer.grounded
        assert answer.text.startswith(I_DONT_KNOW)
        assert answer.citations == []

    def test_i_dont_know_when_no_chunks(self):
        provider = ExtractiveProvider()
        answer = provider.answer("anything at all", [])
        assert not answer.grounded
        assert answer.text.startswith(I_DONT_KNOW)

    def test_deduplicates_identical_spans_and_caps_chunks(self):
        provider = ExtractiveProvider()
        same = [
            scored("a.ts", "function alpha() { return alpha; }", symbol="alpha", start=1, end=3)
            for _ in range(3)
        ]
        others = [
            scored("b.ts", "function alpha_helper() { return 2; }", symbol="alpha_helper", start=5, end=7),
            scored("c.ts", "const alphaMax = 3;", symbol="alphaMax", start=9, end=9),
            scored("d.ts", "let alphaMin = 0;", symbol="alphaMin", start=11, end=11),
        ]
        answer = provider.answer("alpha", same + others)
        # duplicates collapse; at most 3 citations total
        labels = [c.label for c in answer.citations]
        assert len(labels) == len(set(labels))
        assert len(labels) <= 3


class TestCitationParsing:
    def test_extract_citations_from_llm_style_answer(self):
        text = (
            "The refill happens in check() [src/limiters/memory/tokenBucket.ts:28-57] and "
            "the Lua variant lives in [src/limiters/redis/scripts.ts:36-66]. "
            "Repeated: [src/limiters/memory/tokenBucket.ts:28-57]"
        )
        cites = extract_citations(text)
        assert len(cites) == 2  # de-duplicated
        assert cites[0].path == "src/limiters/memory/tokenBucket.ts"
        assert (cites[0].start_line, cites[0].end_line) == (28, 57)

    def test_citation_label_round_trips(self):
        cites = extract_citations("see [a/b/c.py:1-9]")
        assert cites[0].label == "[a/b/c.py:1-9]"
        assert re.match(r"\[[\w./-]+:\d+-\d+\]", cites[0].label)
