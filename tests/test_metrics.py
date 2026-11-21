import pytest

from rag.metrics import evaluate_at_k, score_query


def test_plan_example_includes_missing_query_in_both_means():
    labels = {"Q1": {"B": 1, "D": 2, "E": 1}, "Q2": {"F": 1}}
    runs = {"Q1": ["A", "B", "C", "D"]}
    assert score_query(labels["Q1"], runs["Q1"], 4) == pytest.approx((2 / 3, 1 / 2))
    result = evaluate_at_k(labels, runs, 4)
    assert result.query_count == 2
    assert result.recall == pytest.approx(1 / 3)
    assert result.mrr == 0.25


@pytest.mark.parametrize(("judgments", "ranking", "k", "expected"), [
    ({"A": 1}, ["B", "A"], 1, (0, 0)),          # First hit is beyond the cutoff.
    ({"A": 1}, ["B", "A"], 2, (1, 0.5)),        # Hit exactly at the cutoff.
    ({"A": 2, "B": 0}, ["B", "A"], 5, (1, 0.5)),
    ({"A": 1}, [], 3, (0, 0)),
    ({"A": 0, "B": -1}, ["B", "A"], 3, (0, 0)),
])
def test_rank_and_relevance_boundaries(judgments, ranking, k, expected):
    assert score_query(judgments, ranking, k) == expected


def test_invalid_metrics_inputs():
    with pytest.raises(ValueError, match="positive"):
        score_query({"A": 1}, ["A"], 0)
    with pytest.raises(ValueError, match="unique"):
        score_query({"A": 1}, ["A", "A"], 2)
    with pytest.raises(ValueError, match="judged query"):
        evaluate_at_k({}, {}, 2)
