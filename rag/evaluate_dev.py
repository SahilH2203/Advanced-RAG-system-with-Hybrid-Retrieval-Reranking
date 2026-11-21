"""Compare first-stage or reranked pipelines on the same NFCorpus development split."""

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import psycopg

from .dataset import read_qrels, read_records
from .hybrid import CANDIDATE_DEPTH, fuse_rankings, retrieve_candidates
from .metrics import Metrics, evaluate_at_k
from .rerank import rerank
from .retrieval import BM25Retriever


CUTOFF = 10


@dataclass(frozen=True)
class DevMetrics:
    recall_at_10: float
    mrr_at_10: float
    recall_at_50: float
    query_count: int


def evaluate_dev(
    conn: psycopg.Connection, directory: Path, rankings_file: Path | None = None, *, include_reranked: bool = False
) -> dict[str, Metrics | DevMetrics]:
    queries = read_records(directory / "queries.jsonl.gz")
    qrels = read_qrels(directory / "dev.tsv")
    lexical_index = BM25Retriever.from_database(conn)
    runs = {"bm25": {}, "dense": {}, "hybrid": {}}
    depth = CANDIDATE_DEPTH if include_reranked else CUTOFF
    if include_reranked:
        runs["reranked"] = {}
    for completed, query_id in enumerate(sorted(qrels), start=1):
        query = queries[query_id]["text"]
        bm25, dense = retrieve_candidates(conn, lexical_index, query)
        hybrid = fuse_rankings(bm25, dense, depth)
        for method, hits in (("bm25", bm25), ("dense", dense), ("hybrid", hybrid)):
            runs[method][query_id] = [hit.passage_id for hit in hits[:depth]]
        if include_reranked:
            ids = [hit.passage_id for hit in rerank(query, hybrid)]
            source_ids = runs["hybrid"][query_id]
            if len(ids) != len(source_ids) or set(ids) != set(source_ids):
                raise RuntimeError(f"Reranking changed the candidate set for {query_id}")
            runs["reranked"][query_id] = ids
            if completed % 25 == 0 or completed == len(qrels):
                print(f"Reranked {completed}/{len(qrels)} development queries", file=sys.stderr, flush=True)

    scores = {}
    for method, ranking in runs.items():
        at_ten = evaluate_at_k(qrels, ranking, CUTOFF)
        if include_reranked:
            at_fifty = evaluate_at_k(qrels, ranking, CANDIDATE_DEPTH)
            scores[method] = DevMetrics(at_ten.recall, at_ten.mrr, at_fifty.recall, at_ten.query_count)
        else:
            scores[method] = at_ten
    if rankings_file is not None:
        rankings_file.parent.mkdir(parents=True, exist_ok=True)
        with rankings_file.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["query_id", *runs])
            for query_id in sorted(qrels):
                writer.writerow([query_id, *(" ".join(run[query_id]) for run in runs.values())])
    return scores
