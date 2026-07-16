"""Grounded prompting for LLM answer providers.

The system prompt enforces strict grounding: the model may only use the
provided chunks, must cite every claim as [file:start-end], and must say
"I don't know" when the context is insufficient.
"""

from __future__ import annotations

import re

from ..models import Citation, ScoredChunk

SYSTEM_PROMPT = """\
You are a codebase Q&A assistant. You answer questions about a code repository
using ONLY the code chunks provided in the user message.

Strict rules:
1. Ground every statement in the provided chunks. Do not use outside knowledge
   about libraries, frameworks, or "typical" implementations.
2. Cite the chunks you used inline, in exactly this format: [path:start-end],
   e.g. [src/limiters/memory/tokenBucket.ts:28-57]. Every paragraph that makes
   a claim about the code must contain at least one citation.
3. If the provided chunks do not contain enough information to answer, reply
   with exactly: "I don't know based on the indexed code." optionally followed
   by one sentence about what is missing. Do not guess.
4. Quote small, relevant code snippets when they help, but keep the answer
   focused and under ~250 words.
5. Refer to symbols by their real names as they appear in the chunks.
"""

_CITATION_RE = re.compile(r"\[([\w./\\ -]+\.\w+):(\d+)-(\d+)\]")

I_DONT_KNOW = "I don't know based on the indexed code."


def format_context(chunks: list[ScoredChunk], max_chars_per_chunk: int = 3000) -> str:
    blocks = []
    for i, sc in enumerate(chunks, start=1):
        c = sc.chunk
        header = f"### Chunk {i}: [{c.path}:{c.start_line}-{c.end_line}]"
        meta = f"kind={c.kind}" + (f" symbol={c.symbol}" if c.symbol else "") + f" lang={c.language}"
        body = c.text[:max_chars_per_chunk]
        blocks.append(f"{header}\n{meta}\n```\n{body}\n```")
    return "\n\n".join(blocks)


def build_user_message(question: str, chunks: list[ScoredChunk]) -> str:
    return (
        "Here are the retrieved code chunks (most relevant first):\n\n"
        f"{format_context(chunks)}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the chunks above, with [path:start-end] citations."
    )


def extract_citations(answer_text: str) -> list[Citation]:
    """Parse [path:start-end] citations out of an LLM answer, de-duplicated."""
    seen: set[tuple[str, int, int]] = set()
    out: list[Citation] = []
    for m in _CITATION_RE.finditer(answer_text):
        key = (m.group(1), int(m.group(2)), int(m.group(3)))
        if key not in seen:
            seen.add(key)
            out.append(Citation(path=key[0], start_line=key[1], end_line=key[2]))
    return out
