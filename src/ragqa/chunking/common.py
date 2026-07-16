"""Helpers shared by all chunkers."""

from __future__ import annotations

from ..models import Chunk

# Chunks longer than this many lines get split into overlapping windows so a
# single 800-line function cannot dominate context windows or embeddings.
MAX_CHUNK_LINES = 160
WINDOW_LINES = 120
WINDOW_OVERLAP = 20


def slice_lines(lines: list[str], start: int, end: int) -> str:
    """Return source text for 1-based inclusive [start, end]."""
    return "\n".join(lines[start - 1 : end])


def split_oversized(chunk: Chunk) -> list[Chunk]:
    """Split a chunk that exceeds MAX_CHUNK_LINES into overlapping windows."""
    n_lines = chunk.end_line - chunk.start_line + 1
    if n_lines <= MAX_CHUNK_LINES:
        return [chunk]
    lines = chunk.text.split("\n")
    out: list[Chunk] = []
    part = 1
    offset = 0
    while offset < len(lines):
        window = lines[offset : offset + WINDOW_LINES]
        start_line = chunk.start_line + offset
        end_line = start_line + len(window) - 1
        out.append(
            Chunk(
                repo=chunk.repo,
                path=chunk.path,
                language=chunk.language,
                kind=chunk.kind,
                symbol=f"{chunk.symbol}[part {part}]" if chunk.symbol else f"[part {part}]",
                start_line=start_line,
                end_line=end_line,
                text="\n".join(window),
                doc=chunk.doc if part == 1 else "",
            )
        )
        if offset + WINDOW_LINES >= len(lines):
            break
        offset += WINDOW_LINES - WINDOW_OVERLAP
        part += 1
    return out


def finalize(chunks: list[Chunk]) -> list[Chunk]:
    """Split oversized chunks, drop empty ones, keep source order."""
    out: list[Chunk] = []
    for c in chunks:
        if not c.text.strip():
            continue
        out.extend(split_oversized(c))
    out.sort(key=lambda c: (c.start_line, c.end_line))
    return out


def gap_chunks(
    repo: str,
    path: str,
    language: str,
    lines: list[str],
    covered: list[tuple[int, int]],
) -> list[Chunk]:
    """Build 'module' chunks from lines not covered by any symbol chunk.

    ``covered`` is a list of 1-based inclusive (start, end) ranges. Contiguous
    uncovered regions with non-trivial content (imports, constants, top-level
    statements) become module-level chunks.
    """
    n = len(lines)
    mask = [False] * (n + 2)
    for s, e in covered:
        for i in range(max(1, s), min(n, e) + 1):
            mask[i] = True

    chunks: list[Chunk] = []
    i = 1
    while i <= n:
        if mask[i]:
            i += 1
            continue
        j = i
        while j <= n and not mask[j]:
            j += 1
        seg_end = j - 1
        text = slice_lines(lines, i, seg_end)
        # Require at least one substantive line before emitting a module chunk.
        substantive = [
            ln for ln in text.split("\n") if ln.strip() and not ln.strip().startswith(("#", "//"))
        ]
        if substantive:
            chunks.append(
                Chunk(
                    repo=repo,
                    path=path,
                    language=language,
                    kind="module",
                    symbol="",
                    start_line=i,
                    end_line=seg_end,
                    text=text,
                )
            )
        i = j
    return chunks
