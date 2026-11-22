"""Pair the released Phase 5 prompt and the current CLI generator on identical evidence."""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import psycopg

from examples.run_phase_5 import model_info
from rag import qa
from rag.__main__ import LOCAL_DSN, _search
from rag.hybrid import CANDIDATE_DEPTH
from rag.ollama import chat


ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "results/phase_5_demos.json"
BASELINE_SUFFIX = (
    "Find the passage about the subject of this question. If it states the requested detail, "
    "give that detail followed by its passage ID in square brackets. If the detail is missing, "
    "give the specified abstention."
)


def compare(question, hits, baseline_system, injected=None):
    evidence = [qa.Evidence(h.passage_id, h.title, h.body) for h in hits[:qa.CONTEXT_SIZE]]
    fields = [asdict(item) for item in evidence]
    sources = json.dumps(fields, ensure_ascii=False, indent=2)
    user = f"Source passages:\n{sources}\n\nQuestion: {question}\n\n{BASELINE_SUFFIX}"
    if injected is None:
        before = chat(baseline_system, user) if evidence else qa.ABSTENTION
        after = qa.answer_from_passages(question, hits)
    else:
        before = injected
        with patch("rag.qa.chat", return_value=injected):
            after = qa.answer_from_passages(question, hits)
    if after.evidence != evidence:
        raise ValueError("The two prompts received different evidence.")
    return {
        "evidence": fields,
        "baseline": {"user_message": user, "answer": before,
                     "citation_validation": asdict(qa.validate_citations(before, evidence))},
        "refined": {"user_message": after.user_message, "answer": after.answer,
                    "citation_validation": asdict(after.citation_validation)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=["original", "fresh"])
    parser.add_argument("output", type=Path, help="Write a new JSON transcript at this path")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path so existing transcripts are preserved.")
    history = json.loads(HISTORY.read_bytes())
    meta = model_info()
    if any(meta[key] != history[key] for key in meta):
        raise ValueError("The local model, Ollama version or options differ from Phase 5.")
    name = "phase_5_qa.json" if args.cohort == "original" else "phase_5_refinement_fresh.json"
    cases = json.loads(Path(__file__).with_name(name).read_bytes())
    old_runs = {(r["example_id"], r["mode"]): r for r in history["runs"]}
    runs = []
    for case in cases:
        with psycopg.connect(os.environ.get("RAG_DATABASE_URL", LOCAL_DSN)) as conn:
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            ranked = _search(conn, "reranked", case["question"], CANDIDATE_DEPTH)
        top = ranked[:qa.CONTEXT_SIZE]
        modes = [("live", ranked, None)]
        if args.cohort == "original" and case["id"] == "cookie-count":
            reduced = [h for h in top if h.passage_id not in case["support_ids"]]
            if len(reduced) != len(top) - 1:
                raise ValueError("The ablation requires exactly one supporting passage.")
            fabricated = old_runs[case["id"], "mock_fabricated_id"]["answer"]
            modes.extend([("live_removed_support", reduced, None), ("mock_fabricated_id", top, fabricated)])
        for mode, hits, injected in modes:
            result = compare(case["question"], hits, history["system_message"], injected)
            if args.cohort == "original":
                old = old_runs[case["id"], mode]
                if result["evidence"] != old["evidence"] or result["baseline"]["user_message"] != old["user_message"]:
                    raise ValueError("The original evidence or released prompt changed.")
            removed = mode == "live_removed_support"
            runs.append({
                "example_id": case["id"], "mode": mode, "question": case["question"],
                "expected_answer": qa.ABSTENTION if removed else case["expected_answer"],
                "support_ids": [] if removed else case["support_ids"],
                "reranked_count": len(ranked), "top_three_ids": [h.passage_id for h in top], **result,
            })
            print(case["id"], mode, result["baseline"]["answer"], "=>", result["refined"]["answer"], flush=True)
    transcript = {**meta, "cohort": args.cohort, "baseline_system": history["system_message"],
                  "refined_system": qa.SYSTEM_PROMPT, "runs": runs}
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(transcript, out, ensure_ascii=False, indent=2)
        out.write("\n")


if __name__ == "__main__":
    main()
