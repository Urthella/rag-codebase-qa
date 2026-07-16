import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def mini_repo() -> Path:
    return FIXTURES / "mini_repo"


@pytest.fixture
def tmp_store(tmp_path):
    from ragqa.store.sqlite_store import SqliteVectorStore

    store = SqliteVectorStore(tmp_path / "test.db")
    yield store
    store.close()


@pytest.fixture
def embedder():
    from ragqa.embeddings.local_hash import HashedNGramEmbedder

    return HashedNGramEmbedder(dim=128)
