# RAG-Powered Codebase Q&A Assistant

<!-- badges -->
[![CI](https://github.com/Urthella/rag-codebase-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/Urthella/rag-codebase-qa/actions/workflows/ci.yml)
[![Stars](https://img.shields.io/github/stars/Urthella/rag-codebase-qa?style=flat-square&logo=github)](https://github.com/Urthella/rag-codebase-qa)
[![Last commit](https://img.shields.io/github/last-commit/Urthella/rag-codebase-qa?style=flat-square)](https://github.com/Urthella/rag-codebase-qa/commits/main)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![tree-sitter](https://img.shields.io/badge/tree--sitter-555555?style=flat-square)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white)
![pgvector](https://img.shields.io/badge/pgvector-4169E1?style=flat-square&logo=postgresql&logoColor=white)
![Anthropic](https://img.shields.io/badge/Anthropic-191919?style=flat-square&logo=anthropic&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat-square&logo=openai&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white)

Ask natural-language questions about any code repository and get **grounded
answers with `file:line` citations**. A real retrieval-augmented-generation
pipeline, end to end: AST-aware chunking, pluggable embeddings, a SQLite (or
pgvector) vector store, hybrid retrieval (dense vectors + BM25 fused with
Reciprocal Rank Fusion), strict grounded prompting, and a retrieval eval
harness with hit-rate@k / MRR.

**Everything runs fully offline by default: zero API keys required.** LLM
answerers (Anthropic / OpenAI) and semantic embeddings are optional,
env-configured upgrades.

---

## Architecture

```
                         INGESTION (python -m ragqa.ingest <path-or-url>)
 ┌─────────────┐   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
 │ repo path    │   │ file walker      │   │ AST-aware        │   │ EmbeddingProvider │
 │ or git URL   ├──▶│ (skips node_     ├──▶│ chunker          ├──▶│ local-hash (dflt) │
 │ (git clone   │   │  modules, .git,  │   │ tree-sitter ▸    │   │ openai / sbert    │
 │  --depth 1)  │   │  dist, >1MB ...) │   │ ast / regex      │   └────────┬─────────┘
 └─────────────┘   └──────────────────┘   │ fallback         │            │
                                          └──────────────────┘            ▼
                                                              ┌───────────────────────┐
                                                              │ VectorStore           │
                                                              │ SQLite BLOBs (dflt)   │
                                                              │ pgvector (DATABASE_URL)│
                                                              └───────────┬───────────┘
                         QUERY (POST /api/ask)                            │
 ┌──────────┐   ┌───────────────────────────────────────────┐            │
 │ question ├──▶│           HYBRID RETRIEVAL                │◀───────────┘
 └──────────┘   │  ┌─────────────┐      ┌───────────────┐   │
                │  │ cosine over │      │ BM25 (k1=1.5, │   │
                │  │ embeddings  │      │ b=0.75), code │   │
                │  └──────┬──────┘      │ tokenizer     │   │
                │         │             └──────┬────────┘   │
                │         └──── RRF fusion ────┘            │
                │      score = Σ 1/(60 + rank)              │
                └──────────────────┬────────────────────────┘
                                   ▼
                ┌───────────────────────────────────────────┐     ┌────────────────┐
                │ AnswerProvider                            │     │ chat UI (/)     │
                │ extractive (offline default)              ├────▶│ answer +        │
                │ anthropic (claude-sonnet-5) / openai      │     │ citation chips +│
                │ strict grounding + [file:12-40] citations │     │ scored context  │
                └───────────────────────────────────────────┘     └────────────────┘
```

## Quick start

```powershell
# 1. Install (Windows PowerShell; use source .venv/bin/activate on unix)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .

# 2. Index a repo (local path or git URL)
python -m ragqa.ingest ..\project-02-rate-limiter --name rate-limiter
python -m ragqa.ingest https://github.com/some/repo.git      # clones --depth 1

# 3. Run the server + chat UI
uvicorn ragqa.api:app --port 8000
# open http://127.0.0.1:8000

# 4. Or ask over the API
curl -X POST http://127.0.0.1:8000/api/ask -H "Content-Type: application/json" `
  -d '{"question": "Where is the token bucket refill implemented?"}'

# 5. Tests + retrieval eval
pytest
python scripts\eval.py
```

### API

| Endpoint | Description |
|---|---|
| `POST /api/index` | `{source, name?}`: ingest a local path or git URL (synchronous) |
| `GET /api/status` | Index job state, repo count, active providers |
| `GET /api/repos` | Indexed repos with chunk counts and metadata |
| `POST /api/ask` | `{question, repo?, top_k?, mode? (hybrid\|vector\|bm25), provider?}` → answer + citations + retrieved chunks with per-signal scores |
| `GET /` | Single-page chat UI (plain HTML/JS/CSS, no build step) |

---

## Chunking: AST-aware, not fixed-size

Files are split at **symbol boundaries** (functions, classes, methods,
interfaces, type aliases, multi-line constants), with the leftover top-level
code (imports, module constants) grouped into "module" chunks. Every chunk
records `path`, `start_line`/`end_line` (1-based, inclusive), `symbol`,
`kind`, `language`, and its docstring / leading comment.

**Why AST-aware beats fixed-size windows:**

1. **No mid-function cuts.** A fixed 40-line window routinely splits a
   function's signature from its body, so neither half embeds or matches
   well. A symbol chunk keeps the signature, docstring and body together:
   the unit a developer actually wants back.
2. **Citations are meaningful.** `[src/limiters/memory/tokenBucket.ts:28-57]`
   is exactly the `check()` method, not "lines 40–80 of some window".
3. **Docstrings stay attached** to the code they describe, which is often the
   highest-signal retrieval text.
4. **Granularity matches question granularity.** "Where is the refill
   implemented?" wants a method; "what does this class do?" wants the class.
   We emit both (class chunk + per-method chunks).

Backend selection: **tree-sitter** (`tree-sitter-language-pack`) parses
Python/JS/TS/TSX when installed; if those wheels are unavailable the
dispatcher transparently falls back to Python's built-in **`ast`** module for
`.py` and a **regex + brace-balancing scanner** (string/comment aware) for
JS/TS: same chunk schema, verified by the same tests. Markdown chunks by
heading; JSON/YAML/TOML index whole-file. Chunks longer than 160 lines are
split into 120-line windows with 20-line overlap so one giant function can't
dominate. Force the fallback with `RAGQA_CHUNKER=fallback`.

## Hybrid retrieval + fusion

Two complementary signals per query:

- **Dense vectors**: cosine similarity between the query embedding and chunk
  embeddings (chunk text is embedded together with its path, symbol and
  docstring). Catches paraphrases ("prune expired timestamps" ≈ sliding
  window log cleanup).
- **BM25** (implemented from scratch, k1=1.5, b=0.75, non-negative IDF) over a
  **code-aware tokenizer** that splits `camelCase`/`snake_case` identifiers
  while also keeping the full identifier ("refillPerSecond" matches both
  `refill per second` and the exact symbol). Catches exact identifiers, which
  vector search often blurs.

Rankings are merged with **Reciprocal Rank Fusion**:
`fused(d) = Σ_r 1/(60 + rank_r(d))`. RRF needs no score calibration between
signals of wildly different scales (cosine ∈ [-1,1] vs unbounded BM25) and
rewards documents both signals agree on. A weighted min-max fusion
(`alpha·vec + (1-alpha)·bm25`) is also implemented for comparison. `top_k`
and the mode (`hybrid|vector|bm25`) are per-request parameters.

## Grounding & citations

LLM answerers use a strict system prompt (`ragqa/answer/prompts.py`):

- answer **only** from the provided chunks: no outside knowledge;
- cite every claim inline as `[path:start-end]` (the exact chunk spans);
- reply `"I don't know based on the indexed code."` when context is
  insufficient: guessing is prohibited.

Citations are parsed back out of the model text into structured
`{path, start_line, end_line}` objects for the UI. If a model answers without
citations, the top retrieved chunks are attached as provenance. The
**extractive** default provider is grounded *by construction*: it only emits
text copied from retrieved chunks, with the citation for each excerpt, and
returns "I don't know" when no chunk shares a single term with the question.

## Provider matrix

| Layer | Offline default (no keys) | Env-selected upgrades |
|---|---|---|
| Chunking | tree-sitter if installed, else stdlib `ast` + regex scanner | `RAGQA_CHUNKER=fallback` to force fallback |
| Embeddings | `local-hash`: deterministic hashed char-3-gram + identifier-token features → 384-d numpy vector (BLAKE2b feature hashing, L2-normalized) | `RAGQA_EMBEDDINGS=openai` (text-embedding-3-small) · `RAGQA_EMBEDDINGS=sbert` (sentence-transformers, optional extra: torch not installed by default) |
| Vector store | SQLite, embeddings as float32 BLOBs, brute-force cosine via numpy (matrix cached per repo) | `DATABASE_URL=postgresql://...` → pgvector adapter (HNSW cosine index) |
| Answerer | `extractive`: composed from top chunks with citations | `RAGQA_ANSWER_PROVIDER=anthropic` (`claude-sonnet-5` via `ANTHROPIC_API_KEY`) · `=openai` (`gpt-4o-mini` via `OPENAI_API_KEY`) |

All knobs in `.env.example`. Providers sit behind `EmbeddingProvider`,
`VectorStore` and `AnswerProvider` interfaces; SDKs are imported lazily so
none of the optional packages are needed until selected.

## Evaluation

`scripts/eval.py` + `eval/questions.json`: 18 questions about the sibling
[`project-02-rate-limiter`](../project-02-rate-limiter) corpus (a TypeScript
API-gateway rate limiter), each annotated with the source file(s) a correct
retrieval must surface. The harness re-indexes the corpus into `data/eval.db`,
runs every question through all three retrieval modes and scores:

- **hit-rate@k**: fraction of questions where an expected file appears in the
  top k retrieved chunks (k = 1, 3, 5);
- **MRR**: mean reciprocal rank of the first expected-file hit in the top 10.

### Results (actual run: local-hash embedder, 168 chunks, top_k=10)

| mode   | hit@1 | hit@3 | hit@5 | MRR   |
|--------|-------|-------|-------|-------|
| vector | 50.0% | 83.3% | 94.4% | 0.672 |
| bm25   | 66.7% | 88.9% | 100.0% | 0.781 |
| hybrid | 61.1% | **94.4%** | 94.4% | 0.759 |

Reproduce with `python scripts/eval.py` (add `-v` for per-question ranks).

Reading the numbers: with the deterministic offline embedder, BM25 is the
stronger single signal (the eval questions name identifiers, and the local
embedder is lexical rather than semantic), while **hybrid achieves the best
hit@3**: RRF recovers questions each single signal ranks poorly (e.g. the
Lua-scripts question is a vector-miss but a BM25 top-3; paraphrase questions
lean the other way). With a semantic embedder (`RAGQA_EMBEDDINGS=openai` or
`sbert`) the vector column (and hence hybrid) improves further; the harness
makes that comparison a one-env-var experiment.

## Project layout

```
src/ragqa/
  chunking/      dispatcher + tree-sitter / ast / regex chunkers
  embeddings/    EmbeddingProvider: local-hash (default), openai, sbert
  store/         VectorStore: SQLite (default), pgvector adapter
  retrieval/     code tokenizer, BM25, RRF hybrid searcher
  answer/        AnswerProvider: extractive (default), anthropic, openai + prompts
  ingest.py      pipeline + CLI (python -m ragqa.ingest)
  api.py         FastAPI app (uvicorn ragqa.api:app)
  static/        chat UI (plain HTML/JS/CSS)
scripts/eval.py  retrieval eval harness (hit@k, MRR, per-mode table)
eval/questions.json  18-question eval set for the project-02 corpus
tests/           chunker boundaries, BM25/RRF, store round-trip, extractive
                 citations, API end-to-end via TestClient
```

## Caveats

- The local-hash embedder is **lexical**, not semantic: it exists so the
  pipeline runs offline and deterministically. For semantic recall use the
  OpenAI or sentence-transformers providers.
- SQLite search is brute-force cosine (numpy). Fine to ~100k chunks; beyond
  that use the pgvector store. The pgvector adapter is code-complete but not
  runtime-tested in this environment (no Postgres available).
- `POST /api/index` is synchronous by design (simple + adequate for
  repo-sized corpora); the UI reflects this.
- The regex JS/TS fallback chunker balances braces outside strings/comments
  but does not fully parse: exotic syntax may land in module chunks instead
  of symbol chunks (tree-sitter path handles these).

---

## Related projects

Part of my backend and distributed-systems series:

- [**realtime-collab-editor**](https://github.com/Urthella/realtime-collab-editor): CRDT sync over WebSockets, live presence, debounced persistence
- [**distributed-rate-limiter**](https://github.com/Urthella/distributed-rate-limiter): token bucket, sliding window and fixed window limits with atomic Redis Lua
- [**event-driven-order-system**](https://github.com/Urthella/event-driven-order-system): saga orchestration, compensating transactions, idempotent consumers, tracing
- [**job-queue-orchestrator**](https://github.com/Urthella/job-queue-orchestrator): race-safe job claiming, retries with backoff, dead-letter queue, DAG workflows

More of my work: [github.com/Urthella](https://github.com/Urthella) · [utkudemirtas.tech](https://www.utkudemirtas.tech)
