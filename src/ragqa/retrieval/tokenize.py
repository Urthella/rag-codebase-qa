"""Code-aware tokenizer shared by BM25 and the local hashed embedder.

Splits identifiers on snake_case / camelCase / digits boundaries and keeps the
original identifier too, so a query for "token bucket refill" matches
``refillPerSecond`` and ``TokenBucketMemoryLimiter``.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "in", "of", "to", "and", "or", "for",
    "on", "at", "by", "it", "this", "that", "be", "as", "do", "does", "how",
    "what", "where", "which", "when", "who", "why", "i", "you", "we", "did",
}


def split_identifier(ident: str) -> list[str]:
    parts = []
    for piece in ident.split("_"):
        if not piece:
            continue
        parts.extend(m.group(0) for m in _CAMEL_RE.finditer(piece))
    return [p.lower() for p in parts if p]


def code_tokens(text: str) -> list[str]:
    """Tokenize code or a natural-language query into lowercase subtokens."""
    out: list[str] = []
    for m in _WORD_RE.finditer(text):
        word = m.group(0)
        lower = word.lower()
        sub = split_identifier(word)
        if len(sub) > 1:
            # keep the whole identifier as a token too (exact-match boost)
            if len(lower) >= 2 and lower not in _STOPWORDS:
                out.append(lower)
            out.extend(s for s in sub if len(s) >= 2 and s not in _STOPWORDS)
        elif len(lower) >= 2 and lower not in _STOPWORDS:
            out.append(lower)
    return out
