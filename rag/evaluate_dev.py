"""Evaluate NFCorpus retrieval on development or test and replay stored rankings."""

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import psycopg

from .dataset import read_qrels, read_records
from .hybrid import CANDIDATE_DEPTH, fuse_rankings, retrieve_candidates
from .metrics import Metrics, evaluate_at_k
from .rerank import rerank
from .retrieval import BM25Retriever


CUTOFF = 10
METHODS = ("bm25", "dense", "hybrid", "reranked")


@dataclass(frozen=True)
class DevMetrics:
    recall_at_10: float
    mrr_at_10: float
    recall_at_50: float
    query_count: int


def evaluate_dev(
    conn: psycopg.Connection, directory: Path, rankings_file: Path | None = None, *, include_reranked: bool = False
) -> dict[str, Metrics | DevMetrics]:
    return evaluate_split(conn, directory, "dev", rankings_file, include_reranked=include_reranked)


def evaluate_split(
    conn: psycopg.Connection, directory: Path, split: str, rankings_file: Path | None = None, *,
    include_reranked: bool = True,
) -> dict[str, Metrics | DevMetrics]:
    if split not in ("dev", "test") or (split == "test" and not include_reranked):
        raise ValueError("Use dev or the full four-method test evaluation")
    queries = read_records(directory / "queries.jsonl.gz")
    qrels = read_qrels(directory / f"{split}.tsv")
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
            _check_candidates(query_id, source_ids, ids)
            runs["reranked"][query_id] = ids
            if completed % 25 == 0 or completed == len(qrels):
                print(f"Reranked {completed}/{len(qrels)} {split} queries", file=sys.stderr, flush=True)

    scores = _score_runs(qrels, runs)
    if rankings_file is not None:
        _write_rankings(rankings_file, runs, qrels)
    return scores


def _check_candidates(query_id, hybrid, reranked):
    if len(reranked) != len(hybrid) or set(reranked) != set(hybrid):
        raise RuntimeError(f"Reranking changed the candidate set for {query_id}")


def _score_runs(qrels, runs):
    scores = {}
    for method, ranking in runs.items():
        at_ten = evaluate_at_k(qrels, ranking, CUTOFF)
        if "reranked" in runs:
            at_fifty = evaluate_at_k(qrels, ranking, CANDIDATE_DEPTH)
            scores[method] = DevMetrics(at_ten.recall, at_ten.mrr, at_fifty.recall, at_ten.query_count)
        else:
            scores[method] = at_ten
    if "reranked" in runs and scores["hybrid"].recall_at_50 != scores["reranked"].recall_at_50:
        raise RuntimeError("Hybrid and reranked Recall@50 differ")
    return scores


def _write_rankings(path, runs, qrels):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", newline="", delete=False)
    temp_path = Path(temp.name)
    try:
        with temp as stream:
            writer = csv.writer(stream)
            writer.writerow(["query_id", *runs])
            for query_id in sorted(qrels):
                writer.writerow([query_id, *(" ".join(run[query_id]) for run in runs.values())])
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def score_saved_rankings(directory: Path, split: str, rankings_file: Path) -> dict[str, DevMetrics]:
    """Validate the complete four-method export before scoring it against the chosen split."""
    if split not in ("dev", "test"):
        raise ValueError("Expected dev or test judgments")
    qrels = read_qrels(directory / f"{split}.tsv")
    runs = {method: {} for method in METHODS}
    with rankings_file.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["query_id", *METHODS]:
            raise ValueError("Expected a four-method ranking CSV")
        for row in reader:
            qid = row["query_id"]
            if qid not in qrels or qid in runs["bm25"]:
                raise ValueError(f"Unknown or duplicate query ID: {qid}")
            if None in row or None in row.values():
                raise ValueError(f"Wrong number of fields for {qid}")
            for method in METHODS:
                ids = row[method].split()
                if len(ids) > CANDIDATE_DEPTH or len(set(ids)) != len(ids):
                    raise ValueError(f"Invalid top-{CANDIDATE_DEPTH} ranking for {qid}/{method}")
                runs[method][qid] = ids
            _check_candidates(qid, runs["hybrid"][qid], runs["reranked"][qid])
    if runs["bm25"].keys() != qrels.keys():
        raise ValueError("Ranking CSV is missing judged queries")
    return _score_runs(qrels, runs)
