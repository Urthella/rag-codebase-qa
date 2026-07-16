"""Shared data models used across chunking, storage, retrieval and answering."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Chunk:
    """One retrievable unit of code or text.

    Line numbers are 1-based and inclusive on both ends, so a citation renders
    as ``[path:start_line-end_line]``.
    """

    repo: str
    path: str  # repo-relative, posix separators
    language: str  # "python", "typescript", "javascript", "markdown", ...
    kind: str  # function | method | class | interface | type | enum | const | module | section | file
    symbol: str  # qualified symbol name ("TokenBucket.check") or "" for module chunks
    start_line: int
    end_line: int
    text: str
    doc: str = ""  # docstring or leading comment, if any
    id: Optional[int] = None  # assigned by the vector store

    @property
    def citation(self) -> str:
        return f"[{self.path}:{self.start_line}-{self.end_line}]"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScoredChunk:
    """A chunk plus its retrieval scores for one query."""

    chunk: Chunk
    vector_score: float = 0.0
    bm25_score: float = 0.0
    fused_score: float = 0.0
    rank: int = 0

    def to_dict(self, max_text_chars: int = 1600) -> dict:
        text = self.chunk.text
        truncated = False
        if len(text) > max_text_chars:
            text = text[:max_text_chars]
            truncated = True
        return {
            "id": self.chunk.id,
            "path": self.chunk.path,
            "language": self.chunk.language,
            "kind": self.chunk.kind,
            "symbol": self.chunk.symbol,
            "start_line": self.chunk.start_line,
            "end_line": self.chunk.end_line,
            "doc": self.chunk.doc,
            "text": text,
            "text_truncated": truncated,
            "vector_score": round(self.vector_score, 4),
            "bm25_score": round(self.bm25_score, 4),
            "fused_score": round(self.fused_score, 4),
            "rank": self.rank,
            "citation": self.chunk.citation,
        }


@dataclass
class Citation:
    path: str
    start_line: int
    end_line: int

    @property
    def label(self) -> str:
        return f"[{self.path}:{self.start_line}-{self.end_line}]"

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "label": self.label,
        }


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    provider: str = "extractive"
    grounded: bool = True  # False when the provider said "I don't know"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "citations": [c.to_dict() for c in self.citations],
            "provider": self.provider,
            "grounded": self.grounded,
        }
