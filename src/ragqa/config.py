"""Environment-driven configuration.

Everything works offline with zero keys by default; API-backed providers are
opt-in via environment variables (see .env.example).
"""

from __future__ import annotations

import os
from pathlib import Path

try:  # optional nicety -- load .env from CWD if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def db_path() -> Path:
    return Path(os.environ.get("RAGQA_DB_PATH", str(PROJECT_ROOT / "data" / "index.db")))


def database_url() -> str:
    """Non-empty when a Postgres/pgvector store should be used."""
    return os.environ.get("DATABASE_URL", "")


def repos_dir() -> Path:
    return Path(os.environ.get("RAGQA_REPOS_DIR", str(PROJECT_ROOT / "data" / "repos")))


def embeddings_provider() -> str:
    return os.environ.get("RAGQA_EMBEDDINGS", "local").lower()


def answer_provider() -> str:
    return os.environ.get("RAGQA_ANSWER_PROVIDER", "extractive").lower()


def chunker_backend() -> str:
    """'auto' (tree-sitter when importable, else fallback) or 'fallback'."""
    return os.environ.get("RAGQA_CHUNKER", "auto").lower()


def embedding_dim() -> int:
    return int(os.environ.get("RAGQA_EMBED_DIM", "384"))


def anthropic_model() -> str:
    return os.environ.get("RAGQA_ANTHROPIC_MODEL", "claude-sonnet-5")


def openai_answer_model() -> str:
    return os.environ.get("RAGQA_OPENAI_MODEL", "gpt-4o-mini")


def openai_embedding_model() -> str:
    return os.environ.get("RAGQA_OPENAI_EMBED_MODEL", "text-embedding-3-small")


def sbert_model() -> str:
    return os.environ.get("RAGQA_SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
