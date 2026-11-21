from pathlib import Path

import numpy as np
import pytest

from rag import dense
from rag.retrieval import BM25Retriever, SearchHit, passage_text


def test_bm25_five_passages_and_reusable_index():
    rows = [
        ("P5", "Ferry", "A boat crosses the lake."),
        ("P1", "Repair notes", "A gyrocompass measures heading."),
        ("P2", "Reading", "A library opens early."),
        ("P3", "Transport", "An electric tram stops here."),
        ("P4", "Timetable", "The bus leaves at nine."),
    ]
    retriever = BM25Retriever(rows)
    hits = retriever.search("GYROCOMPASS!", 5)
    assert len(hits) == 1
    assert hits[0] == SearchHit("P1", 1, hits[0].score, "bm25", *rows[1][1:])
    assert hits[0].score > 0
    assert retriever.search("neverincorpus") == []
    assert retriever.search("!!!") == []
    assert retriever.search("ferry", 1)[0].passage_id == "P5"


def test_lexical_ties_use_id_even_when_idf_is_zero():
    rows = [("B", "", "tram"), ("A", "", "tram"), ("C", "", "bus"), ("D", "", "bus")]
    hits = BM25Retriever(rows).search("tram")
    assert [(h.rank, h.passage_id, h.score) for h in hits] == [(1, "A", 0.0), (2, "B", 0.0)]
    assert BM25Retriever([]).search("train") == []


@pytest.fixture
def passages(db):
    schema = Path(dense.__file__).with_name("schema.sql")
    db.execute(schema.read_text(encoding="utf-8"))
    return db


def direction(x, y):
    vector = np.zeros(384, dtype=np.float32)
    vector[:2] = (x, y)
    return vector


def test_known_cosine_directions_and_shared_text(passages, monkeypatch):
    rows = [
        ("C", "Opposite", "West.", direction(-1, 0)),
        ("B", "Right angle", "North.", direction(0, 1)),
        ("A", "Parallel", "East.", direction(1, 0)),
    ]
    dense.register_vector(passages)
    with passages.cursor() as cur:
        cur.executemany("INSERT INTO passages VALUES (%s, %s, %s, %s)", rows)
    seen = []

    def encode_query(texts):
        seen.extend(texts)
        return np.array([direction(1, 0)])

    monkeypatch.setattr(dense, "encode_texts", encode_query)
    hits = dense.search_dense(passages, "towards sunrise", 3)
    assert seen == ["towards sunrise"]
    assert [(h.passage_id, h.rank, h.score) for h in hits] == [("A", 1, 0), ("B", 2, 1), ("C", 3, 2)]
    assert all(h.score_type == "cosine_distance" for h in hits)
    lexical = BM25Retriever.from_database(passages).search("parallel")[0]
    assert (hits[0].passage_id, hits[0].title, hits[0].body) == (lexical.passage_id, lexical.title, lexical.body)
    passages.execute("UPDATE passages SET embedding = %s WHERE passage_id = 'B'", (direction(1, 0),))
    assert [h.passage_id for h in dense.search_dense(passages, "towards sunrise", 2)] == ["A", "B"]


def test_batch_retry_resumes_and_complete_run_is_a_noop(passages, monkeypatch):
    with passages.cursor() as cur:
        cur.executemany("INSERT INTO passages (passage_id, title, body) VALUES (%s, %s, %s)",
                        [(c, c, "body") for c in "CBA"])
    batches = []

    def encode_batch(texts):
        batches.append(texts)
        if len(batches) == 2:
            raise RuntimeError("injected failure")
        return np.stack([direction(1, 0)] * len(texts))

    monkeypatch.setattr(dense, "encode_texts", encode_batch)
    with pytest.raises(RuntimeError, match="injected failure"):
        dense.embed_missing(passages, 2)
    assert passages.execute("SELECT count(embedding) FROM passages").fetchone() == (2,)
    with pytest.raises(ValueError, match="Missing embeddings: 1"):
        dense.search_dense(passages, "question")
    assert dense.embed_missing(passages, 2) == 1
    assert dense.embed_missing(passages, 2) == 0
    assert batches == [[passage_text("A", "body"), passage_text("B", "body")], ["C\nbody"], ["C\nbody"]]
    assert passages.execute("SELECT DISTINCT vector_dims(embedding) FROM passages").fetchall() == [(384,)]


def test_empty_database_and_limits(passages):
    assert dense.search_dense(passages, "question") == []
    assert dense.search_dense(passages, " ") == []
    assert dense.embed_missing(passages) == 0
    with pytest.raises(ValueError, match="positive"):
        BM25Retriever([]).search("q", 0)
    with pytest.raises(ValueError, match="positive"):
        dense.search_dense(passages, "q", 0)
    with pytest.raises(ValueError, match="positive"):
        dense.embed_missing(passages, 0)
