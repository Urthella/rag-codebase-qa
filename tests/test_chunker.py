"""Chunker tests: AST boundaries and line numbers must be exact on fixtures."""

from pathlib import Path

import pytest

from ragqa.chunking import chunk_text
from ragqa.chunking.js_chunker import chunk_js_ts
from ragqa.chunking.python_chunker import chunk_python
from ragqa.chunking import treesitter_chunker

FIXTURES = Path(__file__).parent / "fixtures"
PY_SOURCE = (FIXTURES / "sample.py").read_text(encoding="utf-8")
TS_SOURCE = (FIXTURES / "sample.ts").read_text(encoding="utf-8")


def by_symbol(chunks, symbol):
    matches = [c for c in chunks if c.symbol == symbol]
    assert matches, f"no chunk with symbol {symbol!r}; got {[c.symbol for c in chunks]}"
    return matches[0]


# ---------------------------------------------------------------- python (ast)
class TestPythonAstChunker:
    def test_function_boundaries_and_doc(self):
        chunks = chunk_python("r", "sample.py", PY_SOURCE)
        fn = by_symbol(chunks, "top_level_function")
        assert (fn.start_line, fn.end_line) == (8, 11)
        assert fn.kind == "function"
        assert fn.doc == "Adds two numbers."
        # text matches the exact source lines
        assert fn.text.split("\n")[0] == "def top_level_function(a, b):"
        assert fn.text.split("\n")[-1] == "    return result"

    def test_decorator_included_in_span(self):
        chunks = chunk_python("r", "sample.py", PY_SOURCE)
        fn = by_symbol(chunks, "decorated_function")
        assert (fn.start_line, fn.end_line) == (14, 17)
        assert fn.text.startswith("@staticmethod")

    def test_class_and_method_chunks(self):
        chunks = chunk_python("r", "sample.py", PY_SOURCE)
        cls = by_symbol(chunks, "TokenBucket")
        assert cls.kind == "class"
        assert (cls.start_line, cls.end_line) == (20, 36)
        assert cls.doc == "A tiny token bucket."

        refill = by_symbol(chunks, "TokenBucket.refill")
        assert refill.kind == "method"
        assert (refill.start_line, refill.end_line) == (28, 30)
        assert refill.doc == "Refill tokens based on elapsed time."

        consume = by_symbol(chunks, "TokenBucket.consume")
        assert (consume.start_line, consume.end_line) == (32, 36)

    def test_module_chunk_covers_imports_and_constants(self):
        chunks = chunk_python("r", "sample.py", PY_SOURCE)
        modules = [c for c in chunks if c.kind == "module"]
        assert modules, "expected a module-level chunk for imports/constants"
        head = modules[0]
        assert head.start_line == 1
        assert "import math" in head.text
        assert "MAX_TOKENS = 100" in head.text

    def test_no_overlap_between_symbol_chunks_of_same_kind(self):
        chunks = chunk_python("r", "sample.py", PY_SOURCE)
        funcs = sorted(
            (c for c in chunks if c.kind == "function"), key=lambda c: c.start_line
        )
        for a, b in zip(funcs, funcs[1:]):
            assert a.end_line < b.start_line

    def test_syntax_error_falls_back_to_file_chunk(self):
        chunks = chunk_python("r", "broken.py", "def broken(:\n    pass\n")
        assert len(chunks) == 1
        assert chunks[0].kind == "file"
        assert chunks[0].start_line == 1


# ---------------------------------------------------------------- js/ts (regex)
class TestJsTsRegexChunker:
    def test_interface_boundaries(self):
        chunks = chunk_js_ts("r", "sample.ts", TS_SOURCE, "typescript")
        iface = by_symbol(chunks, "RateLimitDecision")
        assert iface.kind == "interface"
        assert (iface.start_line, iface.end_line) == (4, 7)

    def test_function_with_leading_block_comment(self):
        chunks = chunk_js_ts("r", "sample.ts", TS_SOURCE, "typescript")
        fn = by_symbol(chunks, "checkLimit")
        assert fn.kind == "function"
        # leading /** */ comment is part of the span; code ends at closing brace
        assert (fn.start_line, fn.end_line) == (9, 15)
        assert "Checks whether a request is allowed." in fn.doc

    def test_arrow_function_const(self):
        chunks = chunk_js_ts("r", "sample.ts", TS_SOURCE, "typescript")
        fn = by_symbol(chunks, "refillTokens")
        assert fn.kind == "function"
        assert (fn.start_line, fn.end_line) == (17, 19)

    def test_class_with_methods(self):
        chunks = chunk_js_ts("r", "sample.ts", TS_SOURCE, "typescript")
        cls = by_symbol(chunks, "Bucket")
        assert cls.kind == "class"
        assert (cls.start_line, cls.end_line) == (21, 37)  # includes leading // comment
        assert "tiny in-memory bucket" in cls.doc

        ctor = by_symbol(chunks, "Bucket.constructor")
        assert (ctor.start_line, ctor.end_line) == (25, 27)

        consume = by_symbol(chunks, "Bucket.consume")
        assert consume.kind == "method"
        assert (consume.start_line, consume.end_line) == (30, 36)

    def test_type_alias_single_line(self):
        chunks = chunk_js_ts("r", "sample.ts", TS_SOURCE, "typescript")
        alias = by_symbol(chunks, "Algorithm")
        assert alias.kind == "type"
        assert (alias.start_line, alias.end_line) == (39, 39)

    def test_module_chunk_for_imports(self):
        chunks = chunk_js_ts("r", "sample.ts", TS_SOURCE, "typescript")
        modules = [c for c in chunks if c.kind == "module"]
        assert any("import { thing }" in m.text for m in modules)


# ---------------------------------------------------------------- tree-sitter
@pytest.mark.skipif(not treesitter_chunker.available(), reason="tree-sitter not installed")
class TestTreeSitterChunker:
    def test_python_boundaries_match_ast(self):
        chunks = treesitter_chunker.chunk_with_treesitter("r", "sample.py", PY_SOURCE, "python")
        fn = by_symbol(chunks, "top_level_function")
        assert (fn.start_line, fn.end_line) == (8, 11)
        assert fn.doc == "Adds two numbers."
        cls = by_symbol(chunks, "TokenBucket")
        assert (cls.start_line, cls.end_line) == (20, 36)
        refill = by_symbol(chunks, "TokenBucket.refill")
        assert (refill.start_line, refill.end_line) == (28, 30)

    def test_typescript_declarations(self):
        chunks = treesitter_chunker.chunk_with_treesitter("r", "sample.ts", TS_SOURCE, "typescript")
        fn = by_symbol(chunks, "checkLimit")
        assert fn.kind == "function"
        assert fn.start_line in (9, 12)  # with or without the attached doc comment
        assert fn.end_line == 15
        cls = by_symbol(chunks, "Bucket")
        assert cls.end_line == 37
        consume = by_symbol(chunks, "Bucket.consume")
        assert consume.end_line == 36
        iface = by_symbol(chunks, "RateLimitDecision")
        assert (iface.start_line, iface.end_line)[1] == 7

    def test_dispatcher_uses_some_backend_consistently(self):
        # Whatever backend the dispatcher picks, key symbols must be present.
        chunks = chunk_text("r", "sample.ts", TS_SOURCE, "typescript")
        symbols = {c.symbol for c in chunks}
        assert {"checkLimit", "Bucket", "RateLimitDecision"} <= symbols


# ---------------------------------------------------------------- misc
class TestDispatcherMisc:
    def test_markdown_sections(self):
        md = "# Title\n\nintro\n\n## Setup\n\nsteps here\n"
        chunks = chunk_text("r", "README.md", md, "markdown")
        assert [c.symbol for c in chunks] == ["Title", "Setup"]
        assert chunks[0].start_line == 1
        assert chunks[1].start_line == 5

    def test_oversized_chunk_is_split_with_overlap(self):
        body = "\n".join(f"    x{i} = {i}" for i in range(400))
        source = f"def huge():\n{body}\n"
        chunks = chunk_python("r", "huge.py", source)
        parts = [c for c in chunks if c.symbol.startswith("huge")]
        assert len(parts) > 1
        # windows must be contiguous-with-overlap and line-accurate
        for a, b in zip(parts, parts[1:]):
            assert b.start_line < a.end_line  # overlap
            assert b.start_line > a.start_line
        first_text_lines = parts[0].text.split("\n")
        assert len(first_text_lines) == parts[0].end_line - parts[0].start_line + 1
