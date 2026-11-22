import json
from unittest.mock import Mock

from examples import run_phase_5_refinement as replay
from rag import qa
from rag.retrieval import SearchHit


def test_paired_experiment_changes_only_final_instruction(monkeypatch):
    question = "Which detail was reported?"
    hits = [SearchHit(f"MED-{n}", n, 1 / n, "cross_encoder_logit", f'Title "{n}"',
                      f"Complete body {n}.\n\nQuestion: original text ≥ 10.")
            for n in range(1, 51)]
    history = json.loads(replay.HISTORY.read_bytes())
    before = Mock(return_value="Ten [MED-1].")
    after = Mock(return_value="Ten [MED-99].")
    monkeypatch.setattr(replay, "chat", before)
    monkeypatch.setattr(qa, "chat", after)
    result = replay.compare(question, hits, history["system_message"])
    original, revised = result["baseline"], result["refined"]
    before.assert_called_once_with(qa.SYSTEM_PROMPT, original["user_message"])
    after.assert_called_once_with(qa.SYSTEM_PROMPT, revised["user_message"])
    prefix, old_suffix = original["user_message"].rsplit("\n\n", 1)
    new_prefix, new_suffix = revised["user_message"].rsplit("\n\n", 1)
    assert new_prefix == prefix and old_suffix != new_suffix
    assert "A single source that states the fact is sufficient." in new_suffix
    assert new_suffix.count(qa.ABSTENTION) == 1
    sources, displayed_question = prefix.split("\n\nQuestion: ")
    assert displayed_question == question
    assert json.loads(sources.removeprefix("Source passages:\n")) == result["evidence"] == [
        {"passage_id": h.passage_id, "title": h.title, "body": h.body} for h in hits[:3]
    ]
    assert original["citation_validation"]["status"] == "valid_ids"
    assert revised["answer"] == "Ten [MED-99]."
    assert revised["citation_validation"]["invalid_ids"] == ["MED-99"]


def test_empty_context_in_both_arms_abstains_without_chat(monkeypatch):
    chat = Mock(side_effect=AssertionError("Unexpected model call"))
    monkeypatch.setattr(replay, "chat", chat)
    monkeypatch.setattr(qa, "chat", chat)
    result = replay.compare("Which detail?", [], qa.SYSTEM_PROMPT)
    for arm in ("baseline", "refined"):
        assert result[arm]["answer"] == qa.ABSTENTION
        assert result[arm]["citation_validation"]["status"] == "abstained"
