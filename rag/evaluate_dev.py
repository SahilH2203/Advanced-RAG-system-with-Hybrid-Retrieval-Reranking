"""Compare three rankings on NFCorpus development queries at one fixed cutoff."""

import csv
from pathlib import Path

import psycopg

from .dataset import read_qrels, read_records
from .hybrid import fuse_rankings, retrieve_candidates
from .metrics import Metrics, evaluate_at_k
from .retrieval import BM25Retriever


CUTOFF = 10


def evaluate_dev(
    conn: psycopg.Connection, directory: Path, rankings_file: Path | None = None
) -> dict[str, Metrics]:
    queries = read_records(directory / "queries.jsonl.gz")
    qrels = read_qrels(directory / "dev.tsv")
    lexical_index = BM25Retriever.from_database(conn)
    runs = {"bm25": {}, "dense": {}, "hybrid": {}}
    for query_id in sorted(qrels):
        bm25, dense = retrieve_candidates(conn, lexical_index, queries[query_id]["text"])
        hybrid = fuse_rankings(bm25, dense, CUTOFF)
        for method, hits in (("bm25", bm25), ("dense", dense), ("hybrid", hybrid)):
            runs[method][query_id] = [hit.passage_id for hit in hits[:CUTOFF]]

    scores = {method: evaluate_at_k(qrels, ranking, CUTOFF) for method, ranking in runs.items()}
    if rankings_file is not None:
        rankings_file.parent.mkdir(parents=True, exist_ok=True)
        with rankings_file.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["query_id", *runs])
            for query_id in sorted(qrels):
                writer.writerow([query_id, *(" ".join(run[query_id]) for run in runs.values())])
    return scores
