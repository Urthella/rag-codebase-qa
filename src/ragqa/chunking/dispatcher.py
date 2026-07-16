"""Route files to the right chunker by extension and available backends."""

from __future__ import annotations

import logging
from pathlib import Path

from .. import config
from ..models import Chunk
from . import treesitter_chunker
from .js_chunker import chunk_js_ts
from .python_chunker import chunk_python
from .text_chunker import chunk_markdown, chunk_whole_file

log = logging.getLogger(__name__)

LANGUAGE_BY_EXT = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    ".md": "markdown",
    ".json": "json",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
    ".lua": "lua",
    ".sql": "sql",
    ".sh": "shell",
    ".conf": "config",
}

SUPPORTED_EXTENSIONS = set(LANGUAGE_BY_EXT)

_TREESITTER_LANGS = {"python", "javascript", "typescript", "tsx"}
_FALLBACK_JS_LANGS = {"javascript", "typescript", "tsx"}


def language_for_path(path: str | Path) -> str | None:
    return LANGUAGE_BY_EXT.get(Path(path).suffix.lower())


def _use_treesitter() -> bool:
    return config.chunker_backend() == "auto" and treesitter_chunker.available()


def chunk_text(repo: str, rel_path: str, source: str, language: str) -> list[Chunk]:
    """Chunk already-read source text. ``rel_path`` should use posix separators."""
    if language in _TREESITTER_LANGS and _use_treesitter():
        try:
            return treesitter_chunker.chunk_with_treesitter(repo, rel_path, source, language)
        except Exception:  # pragma: no cover - defensive: never fail ingestion
            log.exception("tree-sitter chunking failed for %s; using fallback", rel_path)
    if language == "python":
        return chunk_python(repo, rel_path, source)
    if language in _FALLBACK_JS_LANGS:
        return chunk_js_ts(repo, rel_path, source, language)
    if language == "markdown":
        return chunk_markdown(repo, rel_path, source)
    return chunk_whole_file(repo, rel_path, source, language)


def chunk_file(repo: str, root: Path, file_path: Path) -> list[Chunk]:
    """Read and chunk one file. Returns [] for unsupported or unreadable files."""
    language = language_for_path(file_path)
    if language is None:
        return []
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    rel_path = file_path.relative_to(root).as_posix()
    return chunk_text(repo, rel_path, source, language)
