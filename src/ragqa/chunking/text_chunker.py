"""Chunkers for non-code text: markdown (by heading) and config files (whole)."""

from __future__ import annotations

import re

from ..models import Chunk
from .common import finalize, slice_lines

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")


def chunk_markdown(repo: str, path: str, source: str) -> list[Chunk]:
    lines = source.split("\n")
    sections: list[tuple[int, str]] = []  # (0-based start idx, title)
    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if m:
            sections.append((i, m.group(2).strip()))
    chunks: list[Chunk] = []
    if not sections or sections[0][0] > 0:
        end0 = (sections[0][0] - 1) if sections else len(lines) - 1
        chunks.append(
            Chunk(
                repo=repo,
                path=path,
                language="markdown",
                kind="section",
                symbol="(preamble)",
                start_line=1,
                end_line=end0 + 1,
                text=slice_lines(lines, 1, end0 + 1),
            )
        )
    for idx, (start, title) in enumerate(sections):
        end = sections[idx + 1][0] - 1 if idx + 1 < len(sections) else len(lines) - 1
        chunks.append(
            Chunk(
                repo=repo,
                path=path,
                language="markdown",
                kind="section",
                symbol=title,
                start_line=start + 1,
                end_line=end + 1,
                text=slice_lines(lines, start + 1, end + 1),
                doc=title,
            )
        )
    return finalize(chunks)


def chunk_whole_file(repo: str, path: str, source: str, language: str) -> list[Chunk]:
    lines = source.split("\n")
    return finalize(
        [
            Chunk(
                repo=repo,
                path=path,
                language=language,
                kind="file",
                symbol="",
                start_line=1,
                end_line=len(lines),
                text=source,
            )
        ]
    )
