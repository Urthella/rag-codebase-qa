"""ragqa -- RAG-powered codebase Q&A assistant.

Pipeline: AST-aware chunking -> pluggable embeddings -> SQLite/pgvector store
-> hybrid retrieval (vector + BM25, RRF fusion) -> grounded answers with
file:line citations.
"""

__version__ = "0.1.0"
