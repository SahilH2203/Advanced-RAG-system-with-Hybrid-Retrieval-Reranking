import gzip
import json
import sys
from unittest.mock import MagicMock

import pytest

from rag import __main__ as cli
from rag import evaluate_dev as evaluation
from rag.retrieval import SearchHit


def test_test_cli_uses_test_queries_and_replays_without_database(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "test.tsv").write_text("query-id\tcorpus-id\tscore\nA\tX\t0\nA\tP\t2\nB\tP\t1\n")
    (tmp_path / "dev.tsv").write_text("query-id\tcorpus-id\tscore\nC\tX\t1\n")
    with gzip.open(tmp_path / "queries.jsonl.gz", "wt", encoding="utf-8") as stream:
        for qid in "ABC":
            stream.write(json.dumps({"_id": qid, "text": qid.lower()}) + "\n")
    hits = [SearchHit(pid, rank, 0., "cosine_distance", "", "") for rank, pid in enumerate(["X", "P"], 1)]
    calls = []

    def retrieve(conn, lexical_index, query):
        calls.append(query)
        return ([], hits) if query == "a" else ([], [])

    connect = MagicMock()
    monkeypatch.setattr(cli.psycopg, "connect", connect)
    monkeypatch.setattr(evaluation.BM25Retriever, "from_database", lambda conn: object())
    monkeypatch.setattr(evaluation, "retrieve_candidates", retrieve)
    monkeypatch.setattr(evaluation, "rerank", lambda query, candidates: candidates[::-1])
    monkeypatch.setattr(sys, "argv", ["rag", "--data-dir", str(tmp_path), "evaluate-test"])
    cli.main()
    live = json.loads(capsys.readouterr().out)
    assert calls == ["a", "b"] and live["split"] == "test" and live["cutoffs"] == [10, 50]
    assert live["methods"]["dense"] == dict(recall_at_10=.5, mrr_at_10=.25, recall_at_50=.5, query_count=2)
    assert live["methods"]["reranked"]["mrr_at_10"] == .5 and live["methods"]["bm25"]["recall_at_50"] == 0
    connect.return_value.__enter__.return_value.execute.assert_called_once_with(
        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
    )
    connect.side_effect = AssertionError("Offline replay must not connect to PostgreSQL")
    monkeypatch.setattr(sys, "argv", ["rag", "--data-dir", str(tmp_path), "score-rankings", "test",
                                      "results/phase_6_test.csv"])
    cli.main()
    assert json.loads(capsys.readouterr().out) == live


@pytest.mark.parametrize(("rows", "error"), [
    ("", "missing judged"),
    ("Q,,,,\nQ,,,,\n", "duplicate query"),
    ("other,,,,\n", "Unknown"),
    ("Q,,,\n", "number of fields"),
    ("Q,,P P,P,P\n", "Invalid top-50"),
    ("Q,,P,P,X\n", "changed the candidate set"),
    ("Q,," + " ".join(str(i) for i in range(51)) + ",,\n", "Invalid top-50"),
])
def test_replay_rejects_incomplete_or_corrupted_rankings(tmp_path, rows, error):
    (tmp_path / "test.tsv").write_text("query-id\tcorpus-id\tscore\nQ\tP\t1\n")
    run = tmp_path / "scores.csv"
    run.write_text("query_id,bm25,dense,hybrid,reranked\n" + rows)
    with pytest.raises((RuntimeError, ValueError), match=error):
        evaluation.score_saved_rankings(tmp_path, "test", run)
