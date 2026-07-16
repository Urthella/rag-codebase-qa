"""Regex + brace-balance chunker for JavaScript / TypeScript.

Fallback used when tree-sitter is unavailable. Recognizes top-level (optionally
exported) declarations: functions, classes (with per-method sub-chunks),
interfaces, enums, type aliases, and const arrow functions / object literals.
End of a declaration is found by balancing braces with a scanner that skips
strings, template literals and comments. Leading ``//`` or ``/** */`` comments
directly above a declaration are captured as the chunk doc.
"""

from __future__ import annotations

import re

from ..models import Chunk
from .common import finalize, gap_chunks, slice_lines

_DECL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("function", re.compile(r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*(\w+)")),
    ("class", re.compile(r"^(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(\w+)")),
    ("interface", re.compile(r"^(?:export\s+)?interface\s+(\w+)")),
    ("enum", re.compile(r"^(?:export\s+)?(?:const\s+)?enum\s+(\w+)")),
    ("type", re.compile(r"^(?:export\s+)?type\s+(\w+)\s*(?:<[^=]*>)?\s*=")),
    (
        "const",
        re.compile(r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*(?::[^=]+)?="),
    ),
]

_METHOD_RE = re.compile(
    r"^\s+(?:public\s+|private\s+|protected\s+|readonly\s+|static\s+|override\s+)*"
    r"(?:async\s+)?(?:get\s+|set\s+)?\*?\s*([A-Za-z_$][\w$]*)\s*(?:<[^>]*>)?\s*\([^;]*$"
)
_METHOD_SKIP = {"if", "for", "while", "switch", "catch", "return", "new", "function", "constructor"}


def _scan_block_end(lines: list[str], start_idx: int) -> int:
    """Given the 0-based line index where a declaration starts, return the
    0-based index of the line where its top-level braces close.

    Skips string literals, template literals and comments while balancing
    ``{}``. If no ``{`` is found before a ``;`` at depth 0, the statement end
    (line with the ``;``) is returned instead (e.g. ``type X = Y;``).
    """
    depth = 0
    seen_brace = False
    in_block_comment = False
    string_ch: str | None = None  # "'", '"', or "`"

    for i in range(start_idx, len(lines)):
        line = lines[i]
        j = 0
        while j < len(line):
            ch = line[j]
            nxt = line[j + 1] if j + 1 < len(line) else ""
            if in_block_comment:
                if ch == "*" and nxt == "/":
                    in_block_comment = False
                    j += 2
                    continue
                j += 1
                continue
            if string_ch is not None:
                if ch == "\\":
                    j += 2
                    continue
                if ch == string_ch:
                    string_ch = None
                j += 1
                continue
            # normal state
            if ch == "/" and nxt == "/":
                break  # rest of line is a comment
            if ch == "/" and nxt == "*":
                in_block_comment = True
                j += 2
                continue
            if ch in ("'", '"', "`"):
                string_ch = ch
                j += 1
                continue
            if ch == "{":
                depth += 1
                seen_brace = True
            elif ch == "}":
                depth -= 1
                if seen_brace and depth == 0:
                    return i
            elif ch == ";" and not seen_brace and depth == 0:
                return i
            j += 1
        # template literals / block comments may span lines; strings ('"') do not
        if string_ch in ("'", '"'):
            string_ch = None
    return len(lines) - 1


def _leading_comment(lines: list[str], decl_idx: int) -> tuple[str, int]:
    """Collect the comment block immediately above ``decl_idx`` (0-based).

    Returns (comment_text, comment_start_idx). comment_start_idx == decl_idx
    when there is no leading comment.
    """
    i = decl_idx - 1
    collected: list[str] = []
    # /** ... */ or /* ... */ block ending on the previous line
    if i >= 0 and lines[i].strip().endswith("*/"):
        j = i
        while j >= 0:
            collected.insert(0, lines[j])
            if lines[j].strip().startswith("/*"):
                break
            j -= 1
        else:
            return "", decl_idx
        text = "\n".join(collected)
        text = re.sub(r"^\s*/\*+|\*+/\s*$", "", text.strip())
        text = "\n".join(re.sub(r"^\s*\*\s?", "", ln) for ln in text.split("\n")).strip()
        return text, j
    # contiguous // lines
    while i >= 0 and lines[i].strip().startswith("//"):
        collected.insert(0, re.sub(r"^\s*//\s?", "", lines[i]))
        i -= 1
    if collected:
        return "\n".join(collected).strip(), i + 1
    return "", decl_idx


def _method_chunks(
    repo: str, path: str, language: str, lines: list[str], cls_name: str, cls_start: int, cls_end: int
) -> list[Chunk]:
    """Extract method sub-chunks inside a class body (0-based cls indices)."""
    out: list[Chunk] = []
    i = cls_start + 1
    while i < cls_end:
        line = lines[i]
        m = _METHOD_RE.match(line)
        name = None
        if m and m.group(1) not in _METHOD_SKIP:
            name = m.group(1)
        elif re.match(r"^\s+constructor\s*\(", line):
            name = "constructor"
        if name and "{" in "".join(lines[i : min(i + 4, cls_end)]):
            end = _scan_block_end(lines, i)
            if end > cls_end:
                end = cls_end
            doc, _ = _leading_comment(lines, i)
            out.append(
                Chunk(
                    repo=repo,
                    path=path,
                    language=language,
                    kind="method",
                    symbol=f"{cls_name}.{name}",
                    start_line=i + 1,
                    end_line=end + 1,
                    text=slice_lines(lines, i + 1, end + 1),
                    doc=doc,
                )
            )
            i = end + 1
        else:
            i += 1
    return out


def chunk_js_ts(repo: str, path: str, source: str, language: str) -> list[Chunk]:
    lines = source.split("\n")
    chunks: list[Chunk] = []
    covered: list[tuple[int, int]] = []

    i = 0
    while i < len(lines):
        stripped_match = None
        kind = None
        symbol = None
        for k, pat in _DECL_PATTERNS:
            m = pat.match(lines[i])
            if m:
                stripped_match, kind, symbol = m, k, m.group(1)
                break
        if stripped_match is None:
            i += 1
            continue
        # A bare `const X = 3;` is module-level state; only treat as its own
        # chunk when it opens a block/arrow function or spans multiple lines.
        end_idx = _scan_block_end(lines, i)
        if kind == "const":
            decl_text = " ".join(lines[i : end_idx + 1])
            if "=>" in decl_text:
                kind = "function"
            elif end_idx == i:
                i += 1
                continue  # single-line constant -> leave to the module chunk
        doc, doc_start = _leading_comment(lines, i)
        start_line = doc_start + 1  # include leading comment in the span
        end_line = end_idx + 1
        covered.append((start_line, end_line))
        chunks.append(
            Chunk(
                repo=repo,
                path=path,
                language=language,
                kind=kind,
                symbol=symbol or "",
                start_line=start_line,
                end_line=end_line,
                text=slice_lines(lines, start_line, end_line),
                doc=doc,
            )
        )
        if kind == "class":
            chunks.extend(_method_chunks(repo, path, language, lines, symbol or "", i, end_idx))
        i = end_idx + 1

    chunks.extend(gap_chunks(repo, path, language, lines, covered))
    return finalize(chunks)
