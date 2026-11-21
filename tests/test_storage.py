import gzip
import json

import psycopg
import pytest

from rag.dataset import SPLITS
from rag.storage import ingest, inspect_dev, verify


def write_records(path, records):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")


@pytest.fixture
def dataset(tmp_path):
    write_records(tmp_path / "corpus.jsonl.gz", [
        {"_id": "MED-1", "title": "Bus timetable", "text": "The last bus leaves at nine."},
        {"_id": "MED-2", "title": "Library", "text": "The library opens at ten."},
    ])
    write_records(tmp_path / "queries.jsonl.gz", [{"_id": "Q1", "text": "When is the last bus?"}])
    for split in SPLITS:
        (tmp_path / f"{split}.tsv").write_text(
            "query-id\tcorpus-id\tscore\nQ1\tMED-1\t2\nQ1\tMED-2\t0\n", encoding="utf-8"
        )
    return tmp_path


def test_ingest_twice_and_inspect_judged_evidence(db, dataset):
    assert ingest(db, dataset) == 2
    assert ingest(db, dataset) == 2
    assert db.execute("SELECT count(embedding) FROM passages").fetchone() == (0,)
    column_type = db.execute("""
        SELECT format_type(atttypid, atttypmod) FROM pg_attribute
        WHERE attrelid = 'passages'::regclass AND attname = 'embedding'
    """).fetchone()
    assert column_type == ("vector(384)",)
    assert verify(db, dataset) == [
        "2 database passage IDs match the corpus",
        *(f"{split}: queries=1, judgments=2, all IDs resolved" for split in SPLITS),
    ]
    output = inspect_dev(db, dataset, "Q1", limit=2)
    assert "When is the last bus?" in output
    assert "MED-1 | qrel 2 | Bus timetable" in output
    assert "The last bus leaves at nine." in output
    assert "MED-2" not in output


def test_verify_finds_a_dangling_judgment(db, dataset):
    ingest(db, dataset)
    (dataset / "dev.tsv").write_text(
        "query-id\tcorpus-id\tscore\nQ404\tMED-404\t0\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"dev:.*Q404.*MED-404"):
        verify(db, dataset)


def test_ingest_rolls_back_a_partially_invalid_batch(db, dataset):
    ingest(db, dataset)
    write_records(dataset / "corpus.jsonl.gz", [
        {"_id": "MED-1", "title": "Changed", "text": "Updated body"},
        {"_id": "MED-3", "title": "Missing body", "text": None},
    ])
    with pytest.raises(psycopg.errors.NotNullViolation):
        ingest(db, dataset)
    assert db.execute("SELECT title FROM passages WHERE passage_id = 'MED-1'").fetchone() == ("Bus timetable",)
    assert db.execute("SELECT count(*) FROM passages").fetchone() == (2,)


def test_reingest_invalidates_only_changed_embeddings(db, dataset, monkeypatch):
    import numpy as np
    from rag import dense

    ingest(db, dataset)
    seen = []

    def encode(texts):
        seen.extend(texts)
        return np.ones((len(texts), 384), dtype=np.float32)

    monkeypatch.setattr(dense, "encode_texts", encode)
    assert dense.embed_missing(db) == 2
    ingest(db, dataset)
    assert dense.embed_missing(db) == 0
    write_records(dataset / "corpus.jsonl.gz", [
        {"_id": "MED-1", "title": "Bus timetable", "text": "The last bus leaves at eight."},
        {"_id": "MED-2", "title": "Library", "text": "The library opens at ten."},
    ])
    ingest(db, dataset)
    assert dense.embed_missing(db) == 1
    assert seen[-1] == "Bus timetable\nThe last bus leaves at eight."
