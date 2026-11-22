"""Replay the Phase 5 grounding examples against a populated database and local Ollama."""

import argparse
import json
import os
from dataclasses import asdict
from http.client import HTTPConnection
from pathlib import Path
from unittest.mock import patch

import psycopg

from rag.__main__ import LOCAL_DSN, _search
from rag.hybrid import CANDIDATE_DEPTH
from rag.ollama import MODEL, OPTIONS
from rag.qa import CONTEXT_SIZE, SYSTEM_PROMPT, answer_from_passages


def model_info():
    conn = HTTPConnection("127.0.0.1", 11434, timeout=10)
    try:
        conn.request("GET", "/api/version")
        version = json.load(conn.getresponse())["version"]
        conn.request("GET", "/api/tags")
        models = json.load(conn.getresponse())["models"]
    finally:
        conn.close()
    return {"ollama_version": version, "model": MODEL,
            "digest": next(m["digest"] for m in models if m["name"] == MODEL), "options": OPTIONS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Path for the JSON transcript")
    args = parser.parse_args()
    examples = json.loads(Path(__file__).with_name("phase_5_qa.json").read_text(encoding="utf-8"))
    runs = []
    transcript = {**model_info(), "system_message": SYSTEM_PROMPT, "runs": runs}

    def record(case, mode, ranked_count, result):
        runs.append({"example_id": case["id"], "mode": mode, "reranked_count": ranked_count, **asdict(result)})
        print(case["id"], mode, result.citation_validation.status, flush=True)

    for case in examples:
        question = case["question"]
        with psycopg.connect(os.environ.get("RAG_DATABASE_URL", LOCAL_DSN)) as conn:
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            ranked = _search(conn, "reranked", question, CANDIDATE_DEPTH)
        record(case, "live", len(ranked), answer_from_passages(question, ranked))
        if case["id"] == "cookie-count":
            original = ranked[:CONTEXT_SIZE]
            reduced = [hit for hit in original if hit.passage_id not in case["support_ids"]]
            if len(reduced) != CONTEXT_SIZE - 1:
                raise ValueError("Cannot run the ablation: expected the supporting passage once in the top three.")
            record(case, "live_removed_support", len(ranked), answer_from_passages(question, reduced))
            with patch("rag.qa.chat", return_value="The study had 77 participants [MED-999999]."):
                record(case, "mock_fabricated_id", len(ranked), answer_from_passages(question, original))

    args.output.write_text(json.dumps(transcript, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
