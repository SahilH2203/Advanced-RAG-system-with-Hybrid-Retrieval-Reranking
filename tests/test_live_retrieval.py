"""Opt-in checks against the fully embedded NFCorpus database."""

import json
import os
import sys

import numpy as np
from pgvector.psycopg import register_vector
import psycopg
import pytest

from rag.__main__ import LOCAL_DSN, main
from rag.dense import encode_texts


EXACT = "X-ray Vision Carrots"
PARAPHRASE = "How can school cafeterias encourage pupils to choose more vegetables?"


@pytest.mark.skipif(os.getenv("RAG_LIVE_TESTS") != "1", reason="requires embedded NFCorpus and cached model")
@pytest.mark.parametrize("query", [EXACT, PARAPHRASE])
@pytest.mark.parametrize("method", ["bm25", "dense"])
def test_real_cli_scores_and_source_text(method, query, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["rag", "search", method, query, "--limit", "5"])
    main()
    hits = json.loads(capsys.readouterr().out)
    assert [hit["rank"] for hit in hits] == [1, 2, 3, 4, 5]
    assert len({hit["passage_id"] for hit in hits}) == 5
    values = [hit["score"] for hit in hits]
    assert values == sorted(values, reverse=(method == "bm25"))
    if method == "bm25" and query == EXACT:
        assert hits[0]["passage_id"] == "MED-3374"

    with psycopg.connect(os.environ.get("RAG_DATABASE_URL", LOCAL_DSN)) as conn:
        register_vector(conn)
        if method == "dense":
            query_vector = encode_texts([query])[0]
            assert query_vector.shape == (384,)
        for hit in hits:
            title, body, vector = conn.execute(
                "SELECT title, body, embedding FROM passages WHERE passage_id = %s",
                (hit["passage_id"],),
            ).fetchone()
            assert (title, body) == (hit["title"], hit["body"])
            if method == "dense":
                assert hit["score_type"] == "cosine_distance"
                similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
                assert hit["score"] == pytest.approx(1 - similarity, abs=2e-6)
            else:
                assert hit["score_type"] == "bm25"
