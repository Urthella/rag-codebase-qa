"""Retrieval evaluation harness.

Indexes the reference corpus (project-02-rate-limiter by default), runs every
question in eval/questions.json through vector-only, BM25-only and hybrid
retrieval, and reports hit-rate@k and MRR per mode. A question is a "hit" at
rank r when the chunk at rank r comes from one of its expected source files.

Usage (from the project root, venv active):
    python scripts/eval.py
    python scripts/eval.py --corpus ..\\project-02-rate-limiter --top-k 10 -v
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # allow running without pip install -e .

from ragqa.embeddings.base import get_embedder  # noqa: E402
from ragqa.ingest import ingest  # noqa: E402
from ragqa.retrieval.hybrid import HybridSearcher  # noqa: E402
from ragqa.store.sqlite_store import SqliteVectorStore  # noqa: E402

MODES = ["vector", "bm25", "hybrid"]
K_VALUES = [1, 3, 5]


def first_hit_rank(results, expected_files: set[str]) -> int | None:
    """Rank (1-based) of the first retrieved chunk from an expected file."""
    for i, sc in enumerate(results, start=1):
        if sc.chunk.path in expected_files:
            return i
    return None


def evaluate(searcher: HybridSearcher, questions: list[dict], top_k: int, verbose: bool):
    table: dict[str, dict] = {}
    details: dict[str, list] = {mode: [] for mode in MODES}
    for mode in MODES:
        hits = {k: 0 for k in K_VALUES}
        rr_sum = 0.0
        for q in questions:
            results = searcher.search(q["question"], top_k=top_k, mode=mode)
            rank = first_hit_rank(results, set(q["expected_files"]))
            details[mode].append((q, rank, results))
            if rank is not None:
                rr_sum += 1.0 / rank
                for k in K_VALUES:
                    if rank <= k:
                        hits[k] += 1
        n = len(questions)
        table[mode] = {
            **{f"hit@{k}": hits[k] / n for k in K_VALUES},
            "mrr": rr_sum / n,
        }

    if verbose:
        for mode in MODES:
            print(f"\n--- {mode} ---")
            for q, rank, results in details[mode]:
                top = results[0].chunk.path if results else "-"
                marker = f"rank {rank}" if rank else "MISS"
                print(f"  [{marker:>7}] Q{q['id']:>2}: {q['question'][:68]}")
                if rank is None or rank > 1:
                    print(f"            top-1 was: {top}")
    return table


def print_table(table: dict[str, dict], n_questions: int, top_k: int, meta: dict):
    print()
    print(f"Retrieval eval: {n_questions} questions, top_k={top_k}, "
          f"corpus={meta.get('corpus', '?')}, embedder={meta.get('embedder', '?')}, "
          f"chunks={meta.get('num_chunks', '?')}")
    header = f"| {'mode':<8} |" + "".join(f" {'hit@' + str(k):>6} |" for k in K_VALUES) + f" {'MRR':>6} |"
    sep = "|" + "-" * 10 + "|" + ("--------|" * (len(K_VALUES) + 1))
    print(header)
    print(sep)
    for mode, row in table.items():
        cells = "".join(f" {row[f'hit@{k}']*100:5.1f}% |" for k in K_VALUES)
        print(f"| {mode:<8} |{cells} {row['mrr']:6.3f} |")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG retrieval evaluation")
    parser.add_argument(
        "--corpus",
        default=str(PROJECT_ROOT.parent / "project-02-rate-limiter"),
        help="Path to the reference corpus (default: sibling project-02-rate-limiter)",
    )
    parser.add_argument(
        "--questions",
        default=str(PROJECT_ROOT / "eval" / "questions.json"),
        help="Path to the eval question set",
    )
    parser.add_argument(
        "--db",
        default=str(PROJECT_ROOT / "data" / "eval.db"),
        help="SQLite db used for the eval index (recreated each run)",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--skip-index", action="store_true",
                        help="Reuse the existing eval index instead of re-ingesting")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    spec = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    questions = spec["questions"]

    store = SqliteVectorStore(args.db)
    embedder = get_embedder()
    repo = "eval-corpus"

    if not args.skip_index or store.repo_version(repo) == (0, 0):
        print(f"Indexing corpus {args.corpus} ...")
        stats = ingest(args.corpus, name=repo, store=store, embedder=embedder)
        print(f"  {stats.num_files} files -> {stats.num_chunks} chunks in {stats.seconds:.1f}s")
        num_chunks = stats.num_chunks
    else:
        num_chunks = store.repo_version(repo)[0]
        print(f"Reusing existing eval index ({num_chunks} chunks)")

    searcher = HybridSearcher(store, embedder, repo)
    table = evaluate(searcher, questions, args.top_k, args.verbose)
    print_table(
        table,
        len(questions),
        args.top_k,
        {"corpus": spec.get("corpus"), "embedder": embedder.name, "num_chunks": num_chunks},
    )

    # Machine-readable output for CI / README updates
    print("JSON:", json.dumps(table))
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
