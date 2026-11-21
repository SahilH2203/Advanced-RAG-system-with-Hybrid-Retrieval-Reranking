"""Shared passage representation and the in-memory lexical retriever."""

import re
from dataclasses import dataclass

import psycopg
from rank_bm25 import BM25Okapi


@dataclass(frozen=True)
class SearchHit:
    passage_id: str
    rank: int
    score: float
    score_type: str
    title: str
    body: str


def passage_text(title: str, body: str) -> str:
    return f"{title}\n{body}".strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    def __init__(self, rows: list[tuple[str, str, str]]):
        # Position i in the BM25 index refers to row i for the life of this object.
        self.rows = tuple(rows)
        documents = [tokenize(passage_text(title, body)) for _, title, body in self.rows]
        self.index = BM25Okapi(documents) if any(documents) else None

    @classmethod
    def from_database(cls, conn: psycopg.Connection) -> "BM25Retriever":
        rows = conn.execute(
            'SELECT passage_id, title, body FROM passages ORDER BY passage_id COLLATE "C"'
        ).fetchall()
        return cls(rows)

    def search(self, query: str, limit: int = 50) -> list[SearchHit]:
        if limit < 1:
            raise ValueError("limit must be positive")
        terms = tokenize(query)
        if self.index is None or not terms:
            return []
        scores = self.index.get_scores(terms)
        # A real match can score zero or less, especially in a tiny corpus.
        matching = [i for i, counts in enumerate(self.index.doc_freqs) if not counts.keys().isdisjoint(terms)]
        matching.sort(key=lambda i: (-scores[i], self.rows[i][0]))
        hits = []
        for rank, i in enumerate(matching[:limit], start=1):
            passage_id, title, body = self.rows[i]
            hits.append(SearchHit(passage_id, rank, float(scores[i]), "bm25", title, body))
        return hits
