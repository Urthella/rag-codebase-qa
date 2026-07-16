"""Offline extractive answer provider -- the zero-key default.

Composes an answer directly from the top retrieved chunks: for each chunk it
shows the citation, the symbol, its docstring/leading comment, and the lines
that actually match the query terms. Not generative -- but fully grounded by
construction, and it makes the demo work with no API key configured.
"""

from __future__ import annotations

from ..models import Answer, Citation, ScoredChunk
from ..retrieval.tokenize import code_tokens
from .base import AnswerProvider
from .prompts import I_DONT_KNOW

_MAX_CHUNKS = 3
_MAX_EXCERPT_LINES = 6


class ExtractiveProvider(AnswerProvider):
    name = "extractive"

    def answer(self, question: str, chunks: list[ScoredChunk]) -> Answer:
        query_tokens = set(code_tokens(question))
        relevant = [sc for sc in chunks if self._is_relevant(sc, query_tokens)]
        if not relevant:
            return Answer(
                text=I_DONT_KNOW + " No indexed chunk matched the terms of the question.",
                citations=[],
                provider=self.name,
                grounded=False,
            )

        used: list[ScoredChunk] = []
        seen_spans: set[tuple[str, int]] = set()
        for sc in relevant:
            key = (sc.chunk.path, sc.chunk.start_line)
            if key in seen_spans:
                continue
            seen_spans.add(key)
            used.append(sc)
            if len(used) >= _MAX_CHUNKS:
                break

        parts = ["Based on the indexed code, the most relevant locations are:", ""]
        citations: list[Citation] = []
        for sc in used:
            c = sc.chunk
            title = c.symbol or c.path.rsplit("/", 1)[-1]
            parts.append(f"- {c.citation} `{title}` ({c.kind}, {c.language})")
            if c.doc:
                doc_line = c.doc.strip().split("\n")[0]
                parts.append(f"  {doc_line}")
            excerpt = self._matching_lines(sc, query_tokens)
            if excerpt:
                parts.append("  ```")
                parts.extend(f"  {line}" for line in excerpt)
                parts.append("  ```")
            citations.append(
                Citation(path=c.path, start_line=c.start_line, end_line=c.end_line)
            )
        parts.append("")
        parts.append(
            "(Extractive offline answer -- set RAGQA_ANSWER_PROVIDER=anthropic or openai "
            "for a synthesized explanation.)"
        )
        return Answer(text="\n".join(parts), citations=citations, provider=self.name)

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _is_relevant(sc: ScoredChunk, query_tokens: set[str]) -> bool:
        if not query_tokens:
            return False
        chunk_tokens = set(
            code_tokens(f"{sc.chunk.text}\n{sc.chunk.symbol}\n{sc.chunk.doc}\n{sc.chunk.path}")
        )
        return bool(query_tokens & chunk_tokens)

    @staticmethod
    def _matching_lines(sc: ScoredChunk, query_tokens: set[str]) -> list[str]:
        """Lines of the chunk that contain query terms (plus the signature)."""
        lines = sc.chunk.text.split("\n")
        picked: list[str] = []
        if lines and lines[0].strip():
            picked.append(lines[0].rstrip())
        for line in lines[1:]:
            if len(picked) >= _MAX_EXCERPT_LINES:
                break
            line_tokens = set(code_tokens(line))
            if line_tokens & query_tokens and line.strip():
                picked.append(line.rstrip())
        return picked
