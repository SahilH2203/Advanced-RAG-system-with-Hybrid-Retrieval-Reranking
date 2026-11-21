import csv
import gzip
import json

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
