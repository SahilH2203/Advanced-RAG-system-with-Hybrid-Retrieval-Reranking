"""Read the BEIR source records and query-to-passage relevance judgments."""

import csv
import gzip
import json
from pathlib import Path


SPLITS = ("dev", "train", "test")


def read_records(path: Path) -> dict[str, dict]:
    """Index the JSON objects by their original BEIR string ID."""
    records = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            record_id = record["_id"]
            if record_id in records:
                raise ValueError(f"Duplicate ID {record_id} in {path.name}")
            records[record_id] = record
    return records


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Group passage judgments by query, retaining the original integer scores."""
    judgments = {}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["query-id", "corpus-id", "score"]:
            raise ValueError(f"Unexpected qrel header in {path.name}")
        for row in reader:
            papers = judgments.setdefault(row["query-id"], {})
            passage_id = row["corpus-id"]
            if passage_id in papers:
                raise ValueError(f"Duplicate judgment for {row['query-id']} and {passage_id}")
            papers[passage_id] = int(row["score"])
    if not judgments:
        raise ValueError(f"No judgments in {path.name}")
    return judgments
