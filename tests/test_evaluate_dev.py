import csv
import gzip
import json

import pytest

from rag import evaluate_dev
from rag.metrics import Metrics
from rag.retrieval import SearchHit


def test_dev_only_comparison_retains_empty_results(tmp_path, monkeypatch):
    with gzip.open(tmp_path / "queries.jsonl.gz", "wt", encoding="utf-8") as stream:
        for query in ("q1", "q2", "unjudged"):
            stream.write(json.dumps({"_id": query, "text": query}) + "\n")
    (tmp_path / "dev.tsv").write_text("query-id\tcorpus-id\tscore\nq1\tD\t1\nq2\tD\t1\n", encoding="utf-8")
    monkeypatch.setattr(evaluate_dev.BM25Retriever, "from_database", lambda conn: object())
    calls = []

    def retrieve(conn, index, query):
        calls.append(query)
        if query == "q2":
            return [], []
        return ([SearchHit("X", 1, 8, "bm25", "X", "")],
                [SearchHit("D", 1, 0, "cosine_distance", "D", ""), SearchHit("X", 2, .1, "cosine_distance", "X", "")])

    monkeypatch.setattr(evaluate_dev, "retrieve_candidates", retrieve)
    output = tmp_path / "rankings.csv"
    scores = evaluate_dev.evaluate_dev(None, tmp_path, output)
    assert calls == ["q1", "q2"]
    assert scores == {"bm25": Metrics(0, 0, 2), "dense": Metrics(.5, .5, 2), "hybrid": Metrics(.5, .25, 2)}
    with output.open(encoding="utf-8", newline="") as stream:
        assert list(csv.reader(stream)) == [
            ["query_id", "bm25", "dense", "hybrid"], ["q1", "X", "D X", "X D"], ["q2", "", "", ""],
        ]


def test_four_method_evaluation_exports_the_full_candidate_set(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate_dev, "read_records", lambda path: {"q": {"text": "question"}})
    monkeypatch.setattr(evaluate_dev, "read_qrels", lambda path: {"q": {"D50": 2}})
    monkeypatch.setattr(evaluate_dev.BM25Retriever, "from_database", lambda conn: object())
    dense = [SearchHit(f"D{i:02}", i, float(i), "cosine_distance", "", "") for i in range(1, 51)]
    monkeypatch.setattr(evaluate_dev, "retrieve_candidates", lambda conn, index, query: ([], dense))
    monkeypatch.setattr(evaluate_dev, "rerank", lambda query, hits: list(reversed(hits)))
    output = tmp_path / "run.csv"
    scores = evaluate_dev.evaluate_dev(None, tmp_path, output, include_reranked=True)
    assert scores["hybrid"] == evaluate_dev.DevMetrics(0, 0, 1, 1)
    assert scores["reranked"] == evaluate_dev.DevMetrics(1, 1, 1, 1)
    with output.open(encoding="utf-8", newline="") as stream:
        row, = csv.DictReader(stream)
    assert row["bm25"] == ""
    assert len(row["reranked"].split()) == 50
    assert row["reranked"].split() == list(reversed(row["hybrid"].split()))

    monkeypatch.setattr(evaluate_dev, "rerank", lambda query, hits: hits[:-1] + hits[:1])
    with pytest.raises(RuntimeError, match="changed the candidate set"):
        evaluate_dev.evaluate_dev(None, tmp_path, include_reranked=True)
