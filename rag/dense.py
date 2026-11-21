"""Batched passage encoding and exact pgvector cosine retrieval."""

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from .retrieval import SearchHit, passage_text


MODEL_ID = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
MODEL_REVISION = "b207367332321f8e44f96e224ef15bc607f4dbf0"
DIMENSIONS = 384


@lru_cache(maxsize=1)
def load_encoder():
    from sentence_transformers import SentenceTransformer

    default_home = Path(__file__).resolve().parents[1] / "data" / "hf"
    cache = Path(os.environ.get("HF_HOME", default_home)) / "hub"
    return SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, cache_folder=str(cache), device=os.environ.get("RAG_DENSE_DEVICE"))


def encode_texts(texts: list[str]) -> np.ndarray:
    vectors = load_encoder().encode(
        texts, batch_size=len(texts), normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )
    if vectors.shape != (len(texts), DIMENSIONS):
        raise ValueError(f"Expected {len(texts)} vectors of length {DIMENSIONS}, got {vectors.shape}")
    if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("Cosine search requires finite, nonzero embeddings")
    return vectors


def embed_missing(conn: psycopg.Connection, batch_size: int = 16) -> int:
    """Commit each batch on an idle connection. Retries pick up the remaining NULLs."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    with conn.transaction():
        register_vector(conn)
    written = 0
    while True:
        with conn.transaction(), conn.cursor() as cur:
            rows = cur.execute(
                'SELECT passage_id, title, body FROM passages WHERE embedding IS NULL '
                'ORDER BY passage_id COLLATE "C" LIMIT %s FOR UPDATE', (batch_size,),
            ).fetchall()
            if not rows:
                return written
            texts = [passage_text(title, body) for _, title, body in rows]
            vectors = encode_texts(texts)
            cur.executemany(
                "UPDATE passages SET embedding = %s WHERE passage_id = %s",
                [(vec, row[0]) for row, vec in zip(rows, vectors, strict=True)],
            )
        written += len(rows)


def search_dense(conn: psycopg.Connection, query: str, limit: int = 50) -> list[SearchHit]:
    if limit < 1:
        raise ValueError("limit must be positive")
    if not query.strip():
        return []
    register_vector(conn)
    total, ready = conn.execute("SELECT count(*), count(embedding) FROM passages").fetchone()
    if not total:
        return []
    missing = total - ready
    if missing:
        raise ValueError(f"Missing embeddings: {missing}. Run 'python -m rag embed' first.")
    query_vector = encode_texts([query])[0]
    rows = conn.execute(
        """
        SELECT passage_id, title, body, embedding <=> %s AS distance
        FROM passages
        WHERE embedding IS NOT NULL
        ORDER BY distance, passage_id COLLATE "C"
        LIMIT %s
        """,
        (query_vector, limit),
    ).fetchall()
    return [
        SearchHit(doc_id, rank, float(distance), "cosine_distance", title, body)
        for rank, (doc_id, title, body, distance) in enumerate(rows, start=1)
    ]
