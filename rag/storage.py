"""Load passages in a single transaction and audit IDs against the input files."""

from pathlib import Path

import psycopg

from .dataset import SPLITS, read_qrels, read_records


def ingest(conn: psycopg.Connection, directory: Path) -> int:
    corpus = read_records(directory / "corpus.jsonl.gz")
    if not corpus:
        raise ValueError("Empty passage corpus")
    schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(schema)
        cur.executemany(
            """
            INSERT INTO passages (passage_id, title, body) VALUES (%s, %s, %s)
            ON CONFLICT (passage_id) DO UPDATE
            SET title = EXCLUDED.title, body = EXCLUDED.body, embedding = NULL
            WHERE (passages.title, passages.body) IS DISTINCT FROM (EXCLUDED.title, EXCLUDED.body)
            """,
            ((doc_id, row["title"], row["text"]) for doc_id, row in corpus.items()),
        )
        return cur.execute("SELECT count(*) FROM passages").fetchone()[0]


def verify(conn: psycopg.Connection, directory: Path) -> list[str]:
    """Check dataset identity and every labeled query/passage reference across splits."""
    source_ids = read_records(directory / "corpus.jsonl.gz").keys()
    stored_ids = {row[0] for row in conn.execute("SELECT passage_id FROM passages")}
    if stored_ids != source_ids:
        missing = sorted(source_ids - stored_ids)
        extra = sorted(stored_ids - source_ids)
        raise ValueError(f"Database differs from corpus. Missing IDs {missing[:5]}, extra IDs {extra[:5]}")
    queries = read_records(directory / "queries.jsonl.gz")
    summary = [f"{len(stored_ids)} database passage IDs match the corpus"]
    for split in SPLITS:
        qrels = read_qrels(directory / f"{split}.tsv")
        missing_queries = qrels.keys() - queries.keys()
        judged_ids = {doc_id for papers in qrels.values() for doc_id in papers}
        missing_passages = judged_ids - stored_ids
        if missing_queries or missing_passages:
            raise ValueError(
                f"{split}: unresolved queries {sorted(missing_queries)[:5]}, "
                f"unresolved passages {sorted(missing_passages)[:5]}"
            )
        pairs = sum(len(papers) for papers in qrels.values())
        summary.append(f"{split}: queries={len(qrels)}, judgments={pairs}, all IDs resolved")
    return summary


def inspect_dev(conn: psycopg.Connection, directory: Path, query_id: str, limit: int) -> str:
    if limit < 1:
        raise ValueError("limit must be positive")
    queries = read_records(directory / "queries.jsonl.gz")
    dev = read_qrels(directory / "dev.tsv")
    if query_id not in dev:
        raise ValueError(f"{query_id} has no development judgments")
    positive = [(doc_id, grade) for doc_id, grade in dev[query_id].items() if grade > 0]
    positive.sort(key=lambda item: (-item[1], item[0]))
    lines = [f"{query_id} | {queries[query_id]['text']}", f"{len(positive)} positive judgments"]
    for doc_id, grade in positive[:limit]:
        row = conn.execute("SELECT title, body FROM passages WHERE passage_id = %s", (doc_id,)).fetchone()
        if row is None:
            raise ValueError(f"No stored passage for {doc_id}")
        title, body = row
        lines.extend(["", f"{doc_id} | qrel {grade} | {title}", body])
    return "\n".join(lines)
