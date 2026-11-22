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
    search = commands.add_parser("search", help="Rank passages with BM25, dense, hybrid, or reranked search")
    search.add_argument("method", choices=("bm25", "dense", "hybrid", "reranked"))
    search.add_argument("query")
    search.add_argument("--limit", type=positive_int, default=50)
    ask = commands.add_parser("ask", help="Answer a question from the top three reranked passages using local Ollama")
    ask.add_argument("question")
    evaluate = commands.add_parser("evaluate-dev", help="Compare development rankings, optionally with a reranker")
    evaluate.add_argument("--reranked", action="store_true", help="Include the cross-encoder and report Recall@50")
    evaluate.add_argument("--save-rankings", type=Path, metavar="CSV", help="Write ranked IDs (top 50 with --reranked)")
    test = commands.add_parser("evaluate-test", help="Run all four frozen retrieval pipelines on the NFCorpus test split")
    test.add_argument("--save-rankings", type=Path, default=Path("results/phase_6_test.csv"), metavar="CSV",
                      help="Ranking export (default: results/phase_6_test.csv)")
    replay = commands.add_parser("score-rankings", help="Recompute four-method metrics from saved rankings and qrels")
    replay.add_argument("split", choices=("dev", "test"))
    replay.add_argument("rankings", type=Path, metavar="CSV")
    args = parser.parse_args()

    if args.command == "download":
        download(args.data_dir)
        return
    if args.command == "score-rankings":
        from .evaluate_dev import CUTOFF, score_saved_rankings
        from .hybrid import CANDIDATE_DEPTH

        try:
            scores = score_saved_rankings(args.data_dir, args.split, args.rankings)
        except (ValueError, OSError, RuntimeError) as error:
            parser.error(str(error))
        print(json.dumps({"split": args.split, "cutoffs": [CUTOFF, CANDIDATE_DEPTH],
                          "methods": {name: asdict(value) for name, value in scores.items()}}, indent=2))
        return
    with psycopg.connect(os.environ.get("RAG_DATABASE_URL", LOCAL_DSN)) as conn:
        if args.command in ("ask", "evaluate-dev", "evaluate-test") or (args.command == "search" and args.method in ("hybrid", "reranked")):
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
        elif args.command in ("evaluate-dev", "evaluate-test"):
            from .evaluate_dev import CUTOFF, evaluate_split
            from .hybrid import CANDIDATE_DEPTH
            from .rerank import RerankError

            split = "test" if args.command == "evaluate-test" else "dev"
            reranked = split == "test" or args.reranked
            try:
                scores = evaluate_split(conn, args.data_dir, split, args.save_rankings, include_reranked=reranked)
            except RerankError as error:
                parser.error(str(error))
            summary = {name: asdict(result) for name, result in scores.items()}
            cutoffs = {"cutoffs": [CUTOFF, CANDIDATE_DEPTH]} if reranked else {"cutoff": CUTOFF}
            print(json.dumps({"split": split, **cutoffs, "methods": summary}, indent=2))
        elif args.command == "ask":
            from .hybrid import CANDIDATE_DEPTH
            from .ollama import OllamaError
            from .qa import answer_from_passages
            from .rerank import RerankError

            if not args.question.strip():
                parser.error("A nonblank question is required.")
            try:
                ranked = _search(conn, "reranked", args.question, CANDIDATE_DEPTH)
                conn.commit()  # Release the database snapshot before local LLM inference.
                result = answer_from_passages(args.question, ranked)
            except (OllamaError, RerankError, ValueError) as error:
                parser.error(str(error))
            # Escaped JSON also round-trips full source text through Windows code-page pipes.
            print(json.dumps(asdict(result), indent=2))
            if result.citation_validation.status == "invalid":
                parser.exit(2)
        else:
            from .rerank import RerankError

            try:
                hits = _search(conn, args.method, args.query, args.limit)
            except (ValueError, RerankError) as error:
                parser.error(str(error))
            print(json.dumps([asdict(hit) for hit in hits], indent=2, ensure_ascii=False))


def _search(conn: psycopg.Connection, method: str, query: str, limit: int):
    """Dispatch a CLI method already validated by argparse."""
    from .dense import search_dense
    from .hybrid import fuse_rankings, retrieve_candidates
    from .rerank import rerank
    from .retrieval import BM25Retriever

    if method == "dense":
        return search_dense(conn, query, limit)
    lexical_index = BM25Retriever.from_database(conn)
    if method == "bm25":
        return lexical_index.search(query, limit)
    bm25, dense = retrieve_candidates(conn, lexical_index, query)
    if method == "hybrid":
        return fuse_rankings(bm25, dense, limit)
    return rerank(query, fuse_rankings(bm25, dense), limit)


if __name__ == "__main__":
    main()
