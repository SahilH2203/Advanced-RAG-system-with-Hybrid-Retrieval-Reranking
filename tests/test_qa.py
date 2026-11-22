import json
import sys
from unittest.mock import MagicMock, Mock

import pytest

from rag import __main__ as cli
from rag import qa
from rag.retrieval import SearchHit


def hits():
    return [SearchHit(f"MED-{i}", rank, -rank, "cross_encoder_logit", f'Title "{i}"',
                      f"Opening line.\nComplete café text ≥ {i}.") for rank, i in enumerate(range(50, 0, -1), 1)]


def test_ask_uses_full_reranked_list_and_sends_exact_top_three(monkeypatch, capsys):
    rows = hits()
    search = Mock(return_value=rows)
    chat = Mock(return_value="There were 77 children [MED-50].")
    conn = MagicMock()
    monkeypatch.setattr(cli.psycopg, "connect", Mock(return_value=conn))
    monkeypatch.setattr(cli, "_search", search)
    monkeypatch.setattr(qa, "chat", chat)
    monkeypatch.setattr(sys, "argv", ["rag", "ask", "How many?"])
    cli.main()
    result = json.loads(capsys.readouterr().out.encode("ascii"))
    search.assert_called_once_with(conn.__enter__.return_value, "reranked", "How many?", 50)
    chat.assert_called_once_with(qa.SYSTEM_PROMPT, result["user_message"])
    assert result["question"] == "How many?"
    assert result["evidence"] == [
        {"passage_id": h.passage_id, "title": h.title, "body": h.body} for h in rows[:3]
    ]
    source_turn, instructions = result["user_message"].split("\n\nQuestion: ", 1)
    assert json.loads(source_turn.removeprefix("Source passages:\n")) == result["evidence"]
    assert instructions.startswith(result["question"] + "\n\n")
    assert result["citation_validation"]["status"] == "valid_ids"


def test_empty_evidence_and_exact_abstention(monkeypatch):
    chat = Mock(return_value=qa.ABSTENTION)
    monkeypatch.setattr(qa, "chat", chat)
    empty = qa.answer_from_passages("What time?", [])
    assert empty.answer == qa.ABSTENTION and empty.citation_validation.status == "abstained"
    chat.assert_not_called()
    partial = qa.answer_from_passages("What time?", hits()[:2])
    assert partial.citation_validation.status == "abstained"
    assert len(partial.evidence) == 2
    with pytest.raises(ValueError, match="nonblank"):
        qa.answer_from_passages("  ", [])


@pytest.mark.parametrize(("answer", "status", "cited", "invalid"), [
    ("One answer [MED-1].\nAnother [MED-1] [MED-2].", "valid_ids", ["MED-1", "MED-2"], []),
    ("One answer. [MED-1]", "valid_ids", ["MED-1"], []),
    ("One [MED-1]. Another [MED-999].", "invalid", ["MED-1", "MED-999"], ["MED-999"]),
    ("There were 77 children.", "invalid", [], []),
    ("An uncited claim. Another claim [MED-2].", "invalid", ["MED-2"], []),
    ("One [MED-1]. A bare source MED-99.", "invalid", ["MED-1", "MED-99"], ["MED-99"]),
    ("Source [1].", "invalid", ["1"], ["1"]),
    ("Study [MED-1, MED-99].", "invalid", ["MED-1", "MED-99"], ["MED-99"]),
    ("Study [MED-1, MED-2].", "invalid", ["MED-1", "MED-2"], []),
    (qa.ABSTENTION + " [MED-1]", "invalid", ["MED-1"], []),
    ("[MED-1]", "invalid", ["MED-1"], []),
    ("", "invalid", [], []),
])
def test_citation_checks(answer, status, cited, invalid):
    evidence = [qa.Evidence("MED-1", "Cookies", "77 children."), qa.Evidence("MED-2", "Survey", "436 residents.")]
    result = qa.validate_citations(answer, evidence)
    assert (result.status, result.cited_ids, result.invalid_ids) == (status, cited, invalid)


def test_membership_check_does_not_certify_factual_support():
    evidence = [qa.Evidence("MED-1", "Count", "77 children.")]
    assert qa.validate_citations("There were 900 children [MED-1].", evidence).status == "valid_ids"


def test_fabricated_response_is_flagged_verbatim_and_cli_exits_nonzero(monkeypatch, capsys):
    raw = "The study had 77 participants [MED-999999]."
    monkeypatch.setattr(cli.psycopg, "connect", MagicMock())
    monkeypatch.setattr(cli, "_search", Mock(return_value=hits()))
    monkeypatch.setattr(qa, "chat", Mock(return_value=raw))
    monkeypatch.setattr(sys, "argv", ["rag", "ask", "How many?"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    result = json.loads(capsys.readouterr().out)
    assert error.value.code == 2 and result["answer"] == raw
    assert result["citation_validation"]["invalid_ids"] == ["MED-999999"]
