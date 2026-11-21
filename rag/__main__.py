import argparse
import os
from pathlib import Path

import psycopg

from .download import download
from .storage import ingest, inspect_dev, verify


LOCAL_DSN = "postgresql://rag:rag_local@127.0.0.1:5433/rag"


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
    args = parser.parse_args()

    if args.command == "download":
        download(args.data_dir)
        return
    with psycopg.connect(os.environ.get("RAG_DATABASE_URL", LOCAL_DSN)) as conn:
        if args.command == "ingest":
            print(f"{ingest(conn, args.data_dir)} passages in the database")
        elif args.command == "verify":
            print("\n".join(verify(conn, args.data_dir)))
        else:
            print(inspect_dev(conn, args.data_dir, args.query_id, args.limit))


if __name__ == "__main__":
    main()
