"""tree-sitter based multi-language chunker (preferred backend).

Uses ``tree-sitter-language-pack`` grammars for Python / JavaScript /
TypeScript / TSX. If the packages are not importable the dispatcher falls back
to the stdlib-``ast`` Python chunker and the regex JS/TS chunker.
"""

from __future__ import annotations

from functools import lru_cache

from ..models import Chunk
from .common import finalize, gap_chunks, slice_lines

_TS_LANG_BY_LANGUAGE = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "tsx": "tsx",
}

_PY_DECLS = {"function_definition": "function", "class_definition": "class"}
_JS_DECLS = {
    "function_declaration": "function",
    "generator_function_declaration": "function",
    "class_declaration": "class",
    "abstract_class_declaration": "class",
    "interface_declaration": "interface",
    "type_alias_declaration": "type",
    "enum_declaration": "enum",
}


def available() -> bool:
    try:
        import tree_sitter  # noqa: F401
        import tree_sitter_language_pack  # noqa: F401

        return True
    except ImportError:
        return False


@lru_cache(maxsize=8)
def _get_parser(ts_lang: str):
    from tree_sitter_language_pack import get_parser

    return get_parser(ts_lang)


def _span(node) -> tuple[int, int]:
    """1-based inclusive (start_line, end_line) for a node."""
    start = node.start_point[0] + 1
    end_row, end_col = node.end_point
    end = end_row + 1 if end_col > 0 else end_row  # exclusive end at col 0 -> previous line
    return start, max(start, end)


def _node_name(node) -> str:
    name = node.child_by_field_name("name")
    if name is not None:
        return name.text.decode("utf-8", "replace")
    return ""


def _leading_comment(node, source_bytes: bytes) -> tuple[str, int]:
    """Collect contiguous comment siblings directly above the node.

    Returns (comment_text, start_line_of_comment_block). start line equals the
    node's own start line when no comment is attached.
    """
    import re

    comments: list[str] = []
    start_line, _ = _span(node)
    prev = node.prev_named_sibling
    expected_row = node.start_point[0] - 1
    while prev is not None and prev.type == "comment" and prev.end_point[0] >= expected_row - 1:
        comments.insert(0, prev.text.decode("utf-8", "replace"))
        start_line, _ = _span(prev)
        expected_row = prev.start_point[0] - 1
        prev = prev.prev_named_sibling
    if not comments:
        return "", _span(node)[0]
    text = "\n".join(comments)
    text = re.sub(r"/\*+|\*+/", "", text)
    cleaned = []
    for ln in text.split("\n"):
        cleaned.append(re.sub(r"^\s*(\*|//|#)\s?", "", ln))
    return "\n".join(cleaned).strip(), start_line


def _py_docstring(node) -> str:
    body = node.child_by_field_name("body")
    if body is None or body.named_child_count == 0:
        return ""
    first = body.named_child(0)
    if first.type == "expression_statement" and first.named_child_count > 0:
        expr = first.named_child(0)
        if expr.type == "string":
            raw = expr.text.decode("utf-8", "replace")
            return raw.strip("\"' \n\r")
    return ""


def _make_chunk(repo, path, language, lines, kind, symbol, start, end, doc="") -> Chunk:
    return Chunk(
        repo=repo,
        path=path,
        language=language,
        kind=kind,
        symbol=symbol,
        start_line=start,
        end_line=end,
        text=slice_lines(lines, start, end),
        doc=doc,
    )


def _chunk_python_ts(repo: str, path: str, lines: list[str], root, source_bytes: bytes) -> list[Chunk]:
    chunks: list[Chunk] = []
    covered: list[tuple[int, int]] = []

    for child in root.named_children:
        node = child
        if node.type == "decorated_definition":
            inner = node.child_by_field_name("definition")
            if inner is None:
                continue
            kind = _PY_DECLS.get(inner.type)
            name_node = inner
        else:
            kind = _PY_DECLS.get(node.type)
            name_node = node
        if kind is None:
            continue
        start, end = _span(node)
        covered.append((start, end))
        symbol = _node_name(name_node)
        chunks.append(
            _make_chunk(repo, path, "python", lines, kind, symbol, start, end, _py_docstring(name_node))
        )
        if kind == "class":
            body = name_node.child_by_field_name("body")
            if body is not None:
                for sub in body.named_children:
                    inner_sub = sub
                    if sub.type == "decorated_definition":
                        inner_sub = sub.child_by_field_name("definition") or sub
                    if inner_sub.type == "function_definition":
                        m_start, m_end = _span(sub)
                        chunks.append(
                            _make_chunk(
                                repo,
                                path,
                                "python",
                                lines,
                                "method",
                                f"{symbol}.{_node_name(inner_sub)}",
                                m_start,
                                m_end,
                                _py_docstring(inner_sub),
                            )
                        )

    chunks.extend(gap_chunks(repo, path, "python", lines, covered))
    return finalize(chunks)


def _arrow_function_name(node) -> str | None:
    """For lexical_declaration nodes: name of a declarator whose value is a function."""
    for decl in node.named_children:
        if decl.type != "variable_declarator":
            continue
        value = decl.child_by_field_name("value")
        if value is not None and value.type in ("arrow_function", "function_expression", "function"):
            name = decl.child_by_field_name("name")
            if name is not None:
                return name.text.decode("utf-8", "replace")
    return None


def _first_declarator_name(node) -> str | None:
    for decl in node.named_children:
        if decl.type == "variable_declarator":
            name = decl.child_by_field_name("name")
            if name is not None:
                return name.text.decode("utf-8", "replace")
    return None


def _chunk_js_ts_ts(
    repo: str, path: str, language: str, lines: list[str], root, source_bytes: bytes
) -> list[Chunk]:
    chunks: list[Chunk] = []
    covered: list[tuple[int, int]] = []

    def handle_decl(outer, decl) -> None:
        kind = _JS_DECLS.get(decl.type)
        symbol = ""
        if kind is None:
            if decl.type in ("lexical_declaration", "variable_declaration"):
                name = _arrow_function_name(decl)
                if name is not None:
                    kind, symbol = "function", name
                else:
                    # Multi-line constants (template-literal scripts, config
                    # objects, lookup tables) become their own symbol chunks;
                    # single-line constants stay in the module chunk.
                    d_start, d_end = _span(decl)
                    name = _first_declarator_name(decl)
                    if name is None or d_end - d_start < 1:
                        return
                    kind, symbol = "const", name
            else:
                return
        else:
            symbol = _node_name(decl)
        start, end = _span(outer)
        doc, doc_start = _leading_comment(outer, source_bytes)
        start = min(start, doc_start)
        covered.append((start, end))
        chunks.append(_make_chunk(repo, path, language, lines, kind, symbol, start, end, doc))
        if kind == "class":
            body = decl.child_by_field_name("body")
            if body is not None:
                for sub in body.named_children:
                    if sub.type == "method_definition":
                        m_start, m_end = _span(sub)
                        m_doc, m_doc_start = _leading_comment(sub, source_bytes)
                        chunks.append(
                            _make_chunk(
                                repo,
                                path,
                                language,
                                lines,
                                "method",
                                f"{symbol}.{_node_name(sub)}",
                                min(m_start, m_doc_start),
                                m_end,
                                m_doc,
                            )
                        )

    for child in root.named_children:
        if child.type == "export_statement":
            decl = child.child_by_field_name("declaration")
            if decl is not None:
                handle_decl(child, decl)
        else:
            handle_decl(child, child)

    chunks.extend(gap_chunks(repo, path, language, lines, covered))
    return finalize(chunks)


def chunk_with_treesitter(repo: str, path: str, source: str, language: str) -> list[Chunk]:
    ts_lang = _TS_LANG_BY_LANGUAGE[language]
    parser = _get_parser(ts_lang)
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    lines = source.split("\n")
    if language == "python":
        return _chunk_python_ts(repo, path, lines, tree.root_node, source_bytes)
    return _chunk_js_ts_ts(repo, path, language, lines, tree.root_node, source_bytes)
