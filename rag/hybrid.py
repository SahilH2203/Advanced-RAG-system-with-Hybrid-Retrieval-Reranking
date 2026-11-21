"""Combine the two Phase 2 candidate lists using Reciprocal Rank Fusion."""

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import psycopg

from .dense import search_dense
from .retrieval import BM25Retriever, SearchHit


CANDIDATE_DEPTH = 50
RRF_CONSTANT = 60


@dataclass(frozen=True)
class HybridHit(SearchHit):
    bm25_rank: int | None
    dense_rank: int | None


def retrieve_candidates(
    conn: psycopg.Connection, lexical_index: BM25Retriever, query: str
) -> tuple[list[SearchHit], list[SearchHit]]:
    return lexical_index.search(query, CANDIDATE_DEPTH), search_dense(conn, query, CANDIDATE_DEPTH)


def _first_hits(ranking: Sequence[SearchHit]) -> dict[str, SearchHit]:
    """Preserve the first occurrence and original rank in an ordered source list."""
    hits = {}
    for hit in ranking[:CANDIDATE_DEPTH]:
        if hit.rank < 1:
            raise ValueError("Source ranks must start at one")
        hits.setdefault(hit.passage_id, hit)
    return hits


def fuse_rankings(
    bm25: Sequence[SearchHit], dense: Sequence[SearchHit], limit: int = CANDIDATE_DEPTH
) -> list[HybridHit]:
    if limit < 1:
        raise ValueError("limit must be positive")
    lexical = _first_hits(bm25)
    semantic = _first_hits(dense)
    scores: dict[str, Fraction] = {}
    for source in (lexical, semantic):
        for doc_id, hit in source.items():
            contribution = Fraction(1, RRF_CONSTANT + hit.rank)
            scores[doc_id] = scores.get(doc_id, 0) + contribution

    ordered = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
    hits = []
    for rank, doc_id in enumerate(ordered[:min(limit, CANDIDATE_DEPTH)], start=1):
        bm25_hit = lexical.get(doc_id)
        dense_hit = semantic.get(doc_id)
        passage = bm25_hit or semantic[doc_id]
        hits.append(HybridHit(
            passage_id=doc_id, rank=rank, score=float(scores[doc_id]), score_type="rrf",
            title=passage.title, body=passage.body,
            bm25_rank=bm25_hit.rank if bm25_hit else None,
            dense_rank=dense_hit.rank if dense_hit else None,
        ))
    return hits
