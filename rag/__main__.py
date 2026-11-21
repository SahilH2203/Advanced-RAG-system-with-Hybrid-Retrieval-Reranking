import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import psycopg

from .download import download
from .storage import ingest, inspect_dev, verify


def local_dsn() -> str:
    dsn = "postgresql://rag@127.0.0.1:5433/rag"
    password_file = Path(__file__).resolve().parent.parent / ".env.db-password"
    if password_file.is_file():
        return psycopg.conninfo.make_conninfo(dsn, password=password_file.read_text(encoding="utf-8").strip())
    return dsn


LOCAL_DSN = local_dsn()


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description="NFCorpus data and relevance judgments")
    parser.add_argument("--data-dir", type=Path, default=Path("data/nfcorpus"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="Download or verify the five pinned input files")
    commands.add_parser("ingest", help="Create the schema and load the passages")
    commands.add_parser("verify", help="Check corpus and judgment IDs across all splits")
    inspect = commands.add_parser("inspect", help="Read a development query and its labeled evidence")
    inspect.add_argument("query_id")
    inspect.add_argument("--limit", type=int, default=2)
    embed = commands.add_parser("embed", help="Fill missing passage embeddings in batches")
    embed.add_argument("--batch-size", type=positive_int, default=16)
    search = commands.add_parser("search", help="Rank passages with BM25, dense search, or RRF")
    search.add_argument("method", choices=("bm25", "dense", "hybrid"))
    search.add_argument("query")
    search.add_argument("--limit", type=positive_int, default=50)
    evaluate = commands.add_parser("evaluate-dev", help="Compare BM25, dense, and hybrid at rank 10")
    evaluate.add_argument("--save-rankings", type=Path, metavar="CSV", help="Write the top-ten IDs per query")
    args = parser.parse_args()

    if args.command == "download":
        download(args.data_dir)
        return
    with psycopg.connect(os.environ.get("RAG_DATABASE_URL", LOCAL_DSN)) as conn:
        if args.command == "evaluate-dev" or (args.command == "search" and args.method == "hybrid"):
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        if args.command == "ingest":
            print(f"{ingest(conn, args.data_dir)} passages in the database")
        elif args.command == "verify":
            print("\n".join(verify(conn, args.data_dir)))
        elif args.command == "inspect":
            print(inspect_dev(conn, args.data_dir, args.query_id, args.limit))
        elif args.command == "embed":
            from .dense import embed_missing

            print(f"Stored {embed_missing(conn, args.batch_size)} new passage embeddings")
        elif args.command == "evaluate-dev":
            from .evaluate_dev import CUTOFF, evaluate_dev

            scores = evaluate_dev(conn, args.data_dir, args.save_rankings)
            summary = {name: asdict(result) for name, result in scores.items()}
            print(json.dumps({"split": "dev", "cutoff": CUTOFF, "methods": summary}, indent=2))
        else:
            try:
                if args.method == "dense":
                    from .dense import search_dense

                    hits = search_dense(conn, args.query, args.limit)
                else:
                    from .retrieval import BM25Retriever

                    lexical_index = BM25Retriever.from_database(conn)
                    if args.method == "bm25":
                        hits = lexical_index.search(args.query, args.limit)
                    else:
                        from .hybrid import fuse_rankings, retrieve_candidates

                        bm25, dense = retrieve_candidates(conn, lexical_index, args.query)
                        hits = fuse_rankings(bm25, dense, args.limit)
            except ValueError as error:
                parser.error(str(error))
            print(json.dumps([asdict(hit) for hit in hits], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
