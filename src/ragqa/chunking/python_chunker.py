"""AST-aware Python chunker built on the stdlib ``ast`` module.

Emits one chunk per top-level function, one per class (whole class body) and
one per method inside a class, plus module-level chunks for uncovered code
(imports, constants, top-level statements). Decorators are included in the
chunk span. Docstrings are extracted into ``Chunk.doc``.
"""

from __future__ import annotations

import ast

from ..models import Chunk
from .common import finalize, gap_chunks, slice_lines


def _node_span(node: ast.AST) -> tuple[int, int]:
    """1-based inclusive span including decorators."""
    start = node.lineno
    if getattr(node, "decorator_list", None):
        start = min(d.lineno for d in node.decorator_list)
    end = node.end_lineno or node.lineno
    return start, end


def _docstring(node: ast.AST) -> str:
    try:
        return ast.get_docstring(node) or ""
    except TypeError:
        return ""


def chunk_python(repo: str, path: str, source: str) -> list[Chunk]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Unparseable file: fall back to a single file-level chunk.
        lines = source.split("\n")
        return finalize(
            [
                Chunk(
                    repo=repo,
                    path=path,
                    language="python",
                    kind="file",
                    symbol="",
                    start_line=1,
                    end_line=len(lines),
                    text=source,
                )
            ]
        )

    lines = source.split("\n")
    chunks: list[Chunk] = []
    covered: list[tuple[int, int]] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start, end = _node_span(node)
            covered.append((start, end))
            chunks.append(
                Chunk(
                    repo=repo,
                    path=path,
                    language="python",
                    kind="function",
                    symbol=node.name,
                    start_line=start,
                    end_line=end,
                    text=slice_lines(lines, start, end),
                    doc=_docstring(node),
                )
            )
        elif isinstance(node, ast.ClassDef):
            cls_start, cls_end = _node_span(node)
            covered.append((cls_start, cls_end))
            chunks.append(
                Chunk(
                    repo=repo,
                    path=path,
                    language="python",
                    kind="class",
                    symbol=node.name,
                    start_line=cls_start,
                    end_line=cls_end,
                    text=slice_lines(lines, cls_start, cls_end),
                    doc=_docstring(node),
                )
            )
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    m_start, m_end = _node_span(sub)
                    chunks.append(
                        Chunk(
                            repo=repo,
                            path=path,
                            language="python",
                            kind="method",
                            symbol=f"{node.name}.{sub.name}",
                            start_line=m_start,
                            end_line=m_end,
                            text=slice_lines(lines, m_start, m_end),
                            doc=_docstring(sub),
                        )
                    )

    chunks.extend(gap_chunks(repo, path, "python", lines, covered))
    return finalize(chunks)
