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


@pytest.mark.skipif(os.getenv("RAG_LIVE_TESTS") != "1", reason="requires embedded NFCorpus and cached model")
def test_real_hybrid_cli_uses_both_top_fifties(monkeypatch, capsys):
    from fractions import Fraction

    def search(method, limit):
        monkeypatch.setattr(sys, "argv", ["rag", "search", method, PARAPHRASE, "--limit", str(limit)])
        main()
        return json.loads(capsys.readouterr().out)

    sources = [search("bm25", 50), search("dense", 50)]
    ranks = [{hit["passage_id"]: hit["rank"] for hit in source} for source in sources]
    passages = {hit["passage_id"]: hit for source in sources for hit in source}
    scores = {doc_id: sum(Fraction(1, 60 + r[doc_id]) for r in ranks if doc_id in r) for doc_id in passages}
    expected_ids = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))[:10]
    hybrid = search("hybrid", 10)
    assert [h["passage_id"] for h in hybrid] == expected_ids
    for position, hit in enumerate(hybrid, start=1):
        doc_id = hit["passage_id"]
        assert (hit["rank"], hit["bm25_rank"], hit["dense_rank"]) == (position, ranks[0].get(doc_id), ranks[1].get(doc_id))
        assert hit["score_type"] == "rrf" and hit["score"] == float(scores[doc_id])
        assert (hit["title"], hit["body"]) == (passages[doc_id]["title"], passages[doc_id]["body"])


@pytest.mark.skipif(os.getenv("RAG_LIVE_TESTS") != "1", reason="requires embedded NFCorpus and cached models")
def test_real_reranked_cli_scores_fifty_before_returning_five(monkeypatch, capsys):
    from rag.rerank import load_cross_encoder
    from rag.retrieval import passage_text

    def search(method, limit):
        monkeypatch.setattr(sys, "argv", ["rag", "search", method, "cilantro", "--limit", str(limit)])
        main()
        return json.loads(capsys.readouterr().out)

    source = search("hybrid", 50)
    model = load_cross_encoder()
    predict = model.predict
    scores = []

    def observe(pairs, **kwargs):
        assert len(pairs) == len(source) == 50
        assert pairs == [("cilantro", passage_text(h["title"], h["body"])) for h in source]
        logits = predict(pairs, **kwargs)
        scores.extend(logits.tolist())
        return logits

    monkeypatch.setattr(model, "predict", observe)
    hits = search("reranked", 5)
    expected = sorted(zip(source, scores, strict=True), key=lambda item: (-item[1], item[0]["passage_id"]))
    assert len(hits) == 5 and load_cross_encoder() is model
    for rank, (hit, (original, logit)) in enumerate(zip(hits, expected[:5], strict=True), 1):
        assert (hit["rank"], hit["hybrid_rank"], hit["score"], hit["score_type"]) == (
            rank, original["rank"], logit, "cross_encoder_logit",
        )
        for field in ("passage_id", "title", "body", "bm25_rank", "dense_rank"):
            assert hit[field] == original[field]
