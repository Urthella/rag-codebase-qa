"""End-to-end API tests via TestClient: index a fixture repo, ask, citations."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGQA_DB_PATH", str(tmp_path / "api-test.db"))
    monkeypatch.setenv("RAGQA_EMBEDDINGS", "local")
    monkeypatch.setenv("RAGQA_ANSWER_PROVIDER", "extractive")
    from ragqa.api import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture
def indexed_client(client, mini_repo):
    resp = client.post("/api/index", json={"source": str(mini_repo), "name": "mini"})
    assert resp.status_code == 200, resp.text
    return client


class TestStatusAndRepos:
    def test_status_before_indexing(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "idle"
        assert body["num_repos"] == 0
        assert body["embedder"] == "local-hash"

    def test_ask_without_repos_is_400(self, client):
        resp = client.post("/api/ask", json={"question": "anything?"})
        assert resp.status_code == 400

    def test_index_and_list_repos(self, indexed_client):
        resp = indexed_client.get("/api/repos")
        assert resp.status_code == 200
        repos = resp.json()["repos"]
        assert len(repos) == 1
        assert repos[0]["name"] == "mini"
        assert repos[0]["num_chunks"] > 3
        assert repos[0]["num_files"] == 3  # calculator.py, tokenBucket.ts, README.md

    def test_index_missing_path_is_404(self, client):
        resp = client.post("/api/index", json={"source": "Z:/definitely/not/here"})
        assert resp.status_code == 404

    def test_status_after_indexing(self, indexed_client):
        body = indexed_client.get("/api/status").json()
        assert body["status"] == "done"
        assert body["last_job"]["repo"] == "mini"
        assert body["num_repos"] == 1


class TestAsk:
    def test_ask_returns_grounded_answer_with_citations(self, indexed_client):
        resp = indexed_client.post(
            "/api/ask",
            json={"question": "Where is the token bucket refill implemented?", "repo": "mini"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        answer = body["answer"]
        assert answer["grounded"] is True
        assert answer["citations"], "expected at least one citation"
        assert any("tokenBucket.ts" in c["path"] for c in answer["citations"])
        # citation label appears in the answer text
        assert answer["citations"][0]["label"] in answer["text"]
        # retrieved chunks carry scores
        assert body["chunks"]
        top = body["chunks"][0]
        assert top["fused_score"] > 0
        assert {"path", "start_line", "end_line", "vector_score", "bm25_score"} <= set(top)

    def test_ask_python_question_hits_python_file(self, indexed_client):
        resp = indexed_client.post(
            "/api/ask",
            json={"question": "How do I add two numbers with add_numbers?", "repo": "mini"},
        )
        body = resp.json()
        assert any("calculator.py" in c["path"] for c in body["answer"]["citations"])

    @pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid"])
    def test_all_retrieval_modes_work(self, indexed_client, mode):
        resp = indexed_client.post(
            "/api/ask",
            json={"question": "token bucket refill", "repo": "mini", "mode": mode},
        )
        assert resp.status_code == 200
        assert resp.json()["mode"] == mode
        assert resp.json()["chunks"]

    def test_unknown_repo_is_404(self, indexed_client):
        resp = indexed_client.post("/api/ask", json={"question": "hello there", "repo": "nope"})
        assert resp.status_code == 404

    def test_unanswerable_question_says_i_dont_know(self, indexed_client):
        resp = indexed_client.post(
            "/api/ask",
            json={"question": "zebra quantum frobnicator xylophone?", "repo": "mini"},
        )
        body = resp.json()
        assert body["answer"]["grounded"] is False
        assert "I don't know" in body["answer"]["text"]

    def test_invalid_mode_is_422(self, indexed_client):
        resp = indexed_client.post(
            "/api/ask", json={"question": "hi there", "mode": "quantum"}
        )
        assert resp.status_code == 422


class TestStaticUI:
    def test_root_serves_chat_ui(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "RAG Codebase Q&amp;A" in resp.text or "RAG Codebase Q&A" in resp.text

    def test_static_assets(self, client):
        assert client.get("/app.js").status_code == 200
        assert client.get("/style.css").status_code == 200
