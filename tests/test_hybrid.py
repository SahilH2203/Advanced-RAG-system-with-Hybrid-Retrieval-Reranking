from fractions import Fraction

import pytest

from rag.hybrid import fuse_rankings
from rag.retrieval import SearchHit


def ranking(ids, metric):
    return [SearchHit(doc_id, rank, 0.0, metric, f"Title {doc_id}", f"Body {doc_id}")
            for rank, doc_id in enumerate(ids, start=1)]


def test_hand_calculated_overlap_and_single_list_hits():
    bm25 = ranking("BC", "bm25")
    dense = ranking("AB", "cosine_distance")
    hits = fuse_rankings(bm25, dense)
    assert [(h.passage_id, h.rank, h.bm25_rank, h.dense_rank) for h in hits] == [
        ("B", 1, 1, 2), ("A", 2, None, 1), ("C", 3, 2, None),
    ]
    assert [h.score for h in hits] == pytest.approx([1/61 + 1/62, 1/61, 1/62])
    assert all(h.score_type == "rrf" and h.body == f"Body {h.passage_id}" for h in hits)


def test_empty_lists_and_duplicate_votes():
    dense = ranking("BCA", "cosine_distance")
    assert [h.passage_id for h in fuse_rankings([], dense)] == list("BCA")
    assert fuse_rankings([], []) == []
    hits = fuse_rankings(ranking("AAB", "bm25"), ranking("BBA", "cosine_distance"))
    assert [(h.passage_id, h.bm25_rank, h.dense_rank) for h in hits] == [("A", 1, 3), ("B", 3, 1)]
    assert [h.score for h in hits] == pytest.approx([1/61 + 1/63] * 2)


def test_mathematical_ties_sort_by_id():
    # These reciprocal sums are equal but direct binary-float additions differ.
    assert Fraction(1, 66) + Fraction(1, 99) == Fraction(1, 72) + Fraction(1, 88)
    bm25 = ranking([f"L{i}" for i in range(50)], "bm25")
    dense = ranking([f"V{i}" for i in range(50)], "cosine_distance")
    for source, index, doc_id in [(bm25, 5, "Z"), (dense, 38, "Z"), (bm25, 11, "A"), (dense, 27, "A")]:
        source[index] = SearchHit(doc_id, index + 1, 0, "unused", doc_id, "")
    first, second = fuse_rankings(bm25, dense, limit=2)
    assert (first.passage_id, second.passage_id) == ("A", "Z")
    assert first.score == second.score


def test_candidate_and_display_limits_are_independent():
    lexical = ranking([f"L{i}" for i in range(49)] + ["shared", "too_late"], "bm25")
    semantic = ranking([f"V{i}" for i in range(49)] + ["shared", "too_late"], "cosine_distance")
    winner, = fuse_rankings(lexical, semantic, limit=1)
    assert (winner.passage_id, winner.bm25_rank, winner.dense_rank) == ("shared", 50, 50)
    hits = fuse_rankings(lexical, semantic, limit=100)
    assert len(hits) == len({hit.passage_id for hit in hits}) == 50
    assert "too_late" not in {hit.passage_id for hit in hits}
    with pytest.raises(ValueError, match="positive"):
        fuse_rankings(lexical, semantic, limit=0)
