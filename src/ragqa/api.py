"""FastAPI backend + static chat UI.

Run with:  uvicorn ragqa.api:app --reload
Endpoints: POST /api/index, GET /api/status, POST /api/ask, GET /api/repos
The chat UI is served at /.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__, config
from .answer.base import get_answer_provider
from .embeddings.base import get_embedder
from .ingest import ingest
from .retrieval.hybrid import SearcherRegistry
from .store.base import get_store

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


class IndexRequest(BaseModel):
    source: str = Field(..., description="Local repo path or git clone URL")
    name: Optional[str] = Field(None, description="Repo name to index under")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2)
    repo: Optional[str] = Field(None, description="Repo to search (default: most recent)")
    top_k: int = Field(8, ge=1, le=50)
    mode: str = Field("hybrid", pattern="^(hybrid|vector|bm25)$")
    provider: Optional[str] = Field(
        None, description="Answer provider override: extractive|anthropic|openai"
    )


class AppState:
    def __init__(self):
        self.store = get_store()
        self.embedder = get_embedder()
        self.searchers = SearcherRegistry(self.store, self.embedder)
        self.lock = threading.Lock()
        self.job: dict = {"state": "idle"}

    def close(self):
        self.store.close()


def _state(request: Request) -> AppState:
    return request.app.state.ragqa


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ragqa = AppState()
    log.info(
        "ragqa ready (store=%s, embedder=%s)",
        type(app.state.ragqa.store).__name__,
        app.state.ragqa.embedder.name,
    )
    yield
    app.state.ragqa.close()


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Codebase Q&A", version=__version__, lifespan=lifespan)

    @app.post("/api/index")
    def index_repo(req: IndexRequest, request: Request):
        state = _state(request)
        if not state.lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="An indexing job is already running")
        try:
            state.job = {"state": "indexing", "source": req.source, "started_at": time.time()}
            stats = ingest(req.source, name=req.name, store=state.store, embedder=state.embedder)
            state.job = {"state": "done", **stats.to_dict()}
            return {"ok": True, "stats": stats.to_dict()}
        except FileNotFoundError as exc:
            state.job = {"state": "error", "error": str(exc)}
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("indexing failed")
            state.job = {"state": "error", "error": str(exc)}
            raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}")
        finally:
            state.lock.release()

    @app.get("/api/status")
    def status(request: Request):
        state = _state(request)
        repos = state.store.list_repos()
        return {
            "status": state.job.get("state", "idle"),
            "last_job": state.job,
            "num_repos": len(repos),
            "embedder": state.embedder.name,
            "answer_provider_default": config.answer_provider(),
            "version": __version__,
        }

    @app.get("/api/repos")
    def repos(request: Request):
        return {"repos": _state(request).store.list_repos()}

    @app.post("/api/ask")
    def ask(req: AskRequest, request: Request):
        state = _state(request)
        repos = state.store.list_repos()
        if not repos:
            raise HTTPException(status_code=400, detail="No repos indexed yet. POST /api/index first.")
        repo = req.repo
        if repo is None:
            repo = sorted(repos, key=lambda r: r.get("indexed_at", ""), reverse=True)[0]["name"]
        elif repo not in {r["name"] for r in repos}:
            raise HTTPException(status_code=404, detail=f"Repo not indexed: {repo}")

        searcher = state.searchers.get(repo)
        results = searcher.search(req.question, top_k=req.top_k, mode=req.mode)

        try:
            provider = get_answer_provider(req.provider)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        answer = provider.answer(req.question, results)

        return {
            "repo": repo,
            "question": req.question,
            "mode": req.mode,
            "answer": answer.to_dict(),
            "chunks": [sc.to_dict() for sc in results],
        }

    # Static chat UI at / (mounted last so /api/* wins)
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    return app


app = create_app()
