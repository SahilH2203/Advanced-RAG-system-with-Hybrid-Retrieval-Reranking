from unittest.mock import Mock

import numpy as np
import pytest

from rag import rerank as module
from rag.hybrid import HybridHit
from rag.metrics import score_query


def candidate(doc_id, rank):
    return HybridHit(doc_id, rank, 1 / (60 + rank), "rrf", f"Title {doc_id}", f"Body {doc_id}",
                     None if rank == 2 else rank, rank + 1)


def fake_model(monkeypatch, scores):
    model = Mock()
    model.predict.return_value = scores
    monkeypatch.setattr(module, "load_cross_encoder", lambda: model)
    return model


def test_scores_remain_attached_to_pairs_and_original_ranks(monkeypatch):
    rows = [candidate("C", 1), candidate("A", 2), candidate("B", 3)]
    model = fake_model(monkeypatch, [-4, 2, -1])
    hits = module.rerank("question", rows)
    assert [(h.passage_id, h.rank, h.score, h.hybrid_rank, h.bm25_rank, h.dense_rank) for h in hits] == [
        ("A", 1, 2, 2, None, 3), ("B", 2, -1, 3, 3, 4), ("C", 3, -4, 1, 1, 2),
    ]
    assert all(h.score_type == "cross_encoder_logit" and h.title == f"Title {h.passage_id}"
               and h.body == f"Body {h.passage_id}" for h in hits)
    model.predict.assert_called_once_with(
        [("question", f"Title {key}\nBody {key}") for key in "CAB"],
        batch_size=16, show_progress_bar=False, apply_softmax=False, convert_to_numpy=True,
    )


def test_all_fifty_scored_before_display_limit_and_recall_preserved(monkeypatch):
    rows = [candidate(f"D{i:02}", i) for i in range(1, 51)]
    model = fake_model(monkeypatch, list(range(50)))
    all_hits = module.rerank("question", rows)
    assert len(all_hits) == 50
    assert {row.passage_id for row in rows} == {hit.passage_id for hit in all_hits}
    labels = {"D50": 2, "unretrieved": 1}
    before = [h.passage_id for h in rows]
    after = [h.passage_id for h in all_hits]
    assert score_query(labels, before, 50)[0] == score_query(labels, after, 50)[0] == .5
    assert score_query(labels, before, 10)[0] == 0
    assert score_query(labels, after, 10)[0] == .5
    assert module.rerank("question", rows, limit=1) == all_hits[:1]
    assert len(model.predict.call_args.args[0]) == 50


def test_tie_break_is_id_order(monkeypatch):
    fake_model(monkeypatch, [1, 1, 1])
    hits = module.rerank("q", [candidate(d, i) for i, d in enumerate("CBA", 1)], limit=2)
    assert [hit.passage_id for hit in hits] == ["A", "B"]


def test_empty_and_invalid_inputs_do_not_load_model(monkeypatch):
    loader = Mock(side_effect=AssertionError("unexpected model load"))
    monkeypatch.setattr(module, "load_cross_encoder", loader)
    assert module.rerank("q", []) == []
    with pytest.raises(ValueError, match="nonblank"):
        module.rerank("  ", [], limit=1)
    with pytest.raises(ValueError, match="positive"):
        module.rerank("q", [], limit=0)
    with pytest.raises(ValueError, match="unique"):
        module.rerank("q", [candidate("A", 1), candidate("A", 2)])
    with pytest.raises(ValueError, match="at most"):
        module.rerank("q", [candidate(str(i), i) for i in range(1, 52)])
    loader.assert_not_called()


@pytest.mark.parametrize("bad_scores", [[], [np.nan], [np.inf], [[1.0]]])
def test_bad_model_output_rejected(monkeypatch, bad_scores):
    fake_model(monkeypatch, bad_scores)
    with pytest.raises(module.RerankError, match="one finite logit"):
        module.rerank("q", [candidate("A", 1)])


def test_scoring_failure_preserves_cause(monkeypatch):
    model = fake_model(monkeypatch, [])
    cause = RuntimeError("device failure")
    model.predict.side_effect = cause
    with pytest.raises(module.RerankError, match="Could not score 1 pairs") as caught:
        module.rerank("q", [candidate("A", 1)])
    assert caught.value.__cause__ is cause
