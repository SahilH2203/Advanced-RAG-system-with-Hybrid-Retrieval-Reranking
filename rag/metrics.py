"""Binary relevance metrics for unique passage IDs ranked from best to worst."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import mean


def score_query(judgments: Mapping[str, int], ranking: Sequence[str], k: int) -> tuple[float, float]:
    """Return (recall at k, reciprocal rank at k) for one evaluated query."""
    if k < 1:
        raise ValueError("k must be positive")
    if len(ranking) != len(set(ranking)):
        raise ValueError("A ranking must contain unique passage IDs")
    relevant = {doc_id for doc_id, value in judgments.items() if value > 0}
    hits = [rank for rank, doc_id in enumerate(ranking[:k], start=1) if doc_id in relevant]
    if not hits:
        return 0.0, 0.0
    return len(hits) / len(relevant), 1.0 / hits[0]


@dataclass(frozen=True)
class Metrics:
    recall: float
    mrr: float
    query_count: int


def evaluate_at_k(
    qrels: Mapping[str, Mapping[str, int]],
    rankings: Mapping[str, Sequence[str]],
    k: int,
) -> Metrics:
    """Average over every query in qrels, including queries with missing runs."""
    if not qrels:
        raise ValueError("At least one judged query is required")
    scores = [score_query(labels, rankings.get(qid, ()), k) for qid, labels in qrels.items()]
    return Metrics(mean(recall for recall, _ in scores), mean(rr for _, rr in scores), len(scores))
