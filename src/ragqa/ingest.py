"""Ingestion pipeline + CLI.

Takes a local repo path or a git clone URL, walks source files, chunks them
AST-aware, embeds each chunk and persists everything to the vector store.

CLI:
    python -m ragqa.ingest <path-or-url> [--name NAME] [--db PATH]
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .chunking import chunk_file, language_for_path
from .embeddings.base import EmbeddingProvider, get_embedder
from .models import Chunk
from .store.base import VectorStore, get_store

log = logging.getLogger(__name__)

IGNORED_DIRS = {
    "node_modules", ".git", ".hg", ".svn", ".venv", "venv", "env", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "dist", "build", "out",
    ".next", ".nuxt", "coverage", ".idea", ".vscode", "target", "vendor",
    ".tox", "site-packages", ".cache", "eggs", ".eggs",
}
MAX_FILE_BYTES = 1_000_000
_EMBED_BATCH = 256

_GIT_URL_RE = re.compile(r"^(https?://|git@|ssh://)|\.git$")


@dataclass
class IngestStats:
    repo: str
    source: str
    num_files: int = 0
    num_chunks: int = 0
    seconds: float = 0.0
    skipped_files: int = 0
    languages: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "repo": self.repo,
            "source": self.source,
            "num_files": self.num_files,
            "num_chunks": self.num_chunks,
            "seconds": round(self.seconds, 2),
            "skipped_files": self.skipped_files,
            "languages": self.languages,
        }


def looks_like_git_url(source: str) -> bool:
    return bool(_GIT_URL_RE.search(source.strip()))


def repo_name_from_source(source: str) -> str:
    name = source.rstrip("/\\").split("/")[-1].split("\\")[-1]
    if name.endswith(".git"):
        name = name[:-4]
    return name or "repo"


def resolve_source(source: str, name: str | None = None) -> tuple[str, Path]:
    """Return (repo_name, local_path). Clones git URLs with --depth 1."""
    if looks_like_git_url(source):
        repo = name or repo_name_from_source(source)
        dest = config.repos_dir() / repo
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            log.info("Repo %s already cloned at %s -- pulling latest", repo, dest)
            subprocess.run(
                ["git", "-C", str(dest), "pull", "--ff-only"],
                check=False,
                capture_output=True,
                timeout=300,
            )
        else:
            log.info("Cloning %s -> %s", source, dest)
            subprocess.run(
                ["git", "clone", "--depth", "1", source, str(dest)],
                check=True,
                capture_output=True,
                timeout=600,
            )
        return repo, dest
    path = Path(source).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source}")
    return name or path.name, path


def iter_source_files(root: Path):
    """Yield supported source files under root, skipping vendored/derived dirs."""
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir(), key=lambda p: p.name)
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir():
                if entry.name in IGNORED_DIRS or entry.name.startswith("."):
                    continue
                stack.append(entry)
            elif entry.is_file():
                if language_for_path(entry) is None:
                    continue
                try:
                    if entry.stat().st_size > MAX_FILE_BYTES:
                        continue
                except OSError:
                    continue
                yield entry


def ingest(
    source: str,
    name: str | None = None,
    store: VectorStore | None = None,
    embedder: EmbeddingProvider | None = None,
) -> IngestStats:
    """Full pipeline: resolve -> walk -> chunk -> embed -> persist.

    Re-ingesting the same repo name replaces its previous index.
    """
    t0 = time.time()
    owns_store = store is None
    store = store or get_store()
    embedder = embedder or get_embedder()

    repo, root = resolve_source(source, name)
    stats = IngestStats(repo=repo, source=source)

    all_chunks: list[Chunk] = []
    for file_path in iter_source_files(root):
        chunks = chunk_file(repo, root, file_path)
        if not chunks:
            stats.skipped_files += 1
            continue
        stats.num_files += 1
        lang = chunks[0].language
        stats.languages[lang] = stats.languages.get(lang, 0) + 1
        all_chunks.extend(chunks)

    store.delete_repo(repo)
    for start in range(0, len(all_chunks), _EMBED_BATCH):
        batch = all_chunks[start : start + _EMBED_BATCH]
        vectors = embedder.embed_texts([_embedding_text(c) for c in batch])
        store.add_chunks(batch, vectors)

    stats.num_chunks = len(all_chunks)
    stats.seconds = time.time() - t0
    store.upsert_repo_meta(
        repo,
        {
            "source": source,
            "path": str(root),
            "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "num_files": stats.num_files,
            "num_chunks": stats.num_chunks,
            "embedder": embedder.name,
            "dim": embedder.dim,
        },
    )
    if owns_store:
        store.close()
    return stats


def _embedding_text(chunk: Chunk) -> str:
    """What gets embedded: path + symbol + doc + code, so file names and
    docstrings contribute to dense retrieval."""
    head = f"{chunk.path} {chunk.symbol} {chunk.kind}".strip()
    parts = [head]
    if chunk.doc:
        parts.append(chunk.doc)
    parts.append(chunk.text)
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ragqa.ingest",
        description="Index a local repo path or a git URL into the RAG store.",
    )
    parser.add_argument("source", help="local directory or git clone URL")
    parser.add_argument("--name", help="repo name to index under (default: derived)")
    parser.add_argument("--db", help="override SQLite db path (RAGQA_DB_PATH)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.db:
        import os

        os.environ["RAGQA_DB_PATH"] = args.db

    stats = ingest(args.source, name=args.name)
    print(
        f"Indexed repo '{stats.repo}': {stats.num_files} files -> {stats.num_chunks} chunks "
        f"in {stats.seconds:.1f}s (languages: {stats.languages})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
