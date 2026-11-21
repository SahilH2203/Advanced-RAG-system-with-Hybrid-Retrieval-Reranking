"""Jointly score the fixed RRF candidate set with a pretrained cross-encoder."""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .hybrid import CANDIDATE_DEPTH, HybridHit
from .retrieval import passage_text


MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
MODEL_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
MAX_TOKENS = 512
BATCH_SIZE = 16


@dataclass(frozen=True)
class RerankedHit(HybridHit):
    hybrid_rank: int


class RerankError(RuntimeError):
    """The model could not supply a valid score for every candidate."""


@lru_cache(maxsize=1)
def load_cross_encoder():
    from sentence_transformers import CrossEncoder
    from torch.nn import Identity

    default_home = Path(__file__).resolve().parents[1] / "data" / "hf"
    cache = Path(os.environ.get("HF_HOME", default_home)) / "hub"
    return CrossEncoder(
        MODEL_ID, revision=MODEL_REVISION, cache_folder=str(cache),
        max_length=MAX_TOKENS, activation_fn=Identity(),
    )


def rerank(query: str, candidates: Sequence[HybridHit], limit: int = CANDIDATE_DEPTH) -> list[RerankedHit]:
    if not isinstance(limit, int) or limit < 1 or not isinstance(query, str) or not query.strip():
        raise ValueError("Reranking requires a nonblank query and a positive limit")
    if len(candidates) > CANDIDATE_DEPTH:
        raise ValueError(f"Expected at most {CANDIDATE_DEPTH} fused candidates")
    if len({hit.passage_id for hit in candidates}) != len(candidates):
        raise ValueError("Fused candidates must have unique passage IDs")
    if not candidates:
        return []

    pairs = [(query, passage_text(hit.title, hit.body)) for hit in candidates]
    try:
        scores = np.asarray(load_cross_encoder().predict(
            pairs, batch_size=BATCH_SIZE, show_progress_bar=False, apply_softmax=False,
            convert_to_numpy=True,
        ), dtype=float)
    except Exception as exc:
        raise RerankError(f"Could not score {len(pairs)} pairs with {MODEL_ID}: {exc}") from exc
    if scores.shape != (len(pairs),) or not np.isfinite(scores).all():
        raise RerankError(f"Expected one finite logit per pair ({len(pairs)} pairs, score shape {scores.shape})")

    scored = sorted(zip(candidates, scores, strict=True), key=lambda item: (-item[1], item[0].passage_id))
    return [
        RerankedHit(
            passage_id=hit.passage_id, rank=rank, score=float(score), score_type="cross_encoder_logit",
            title=hit.title, body=hit.body, bm25_rank=hit.bm25_rank, dense_rank=hit.dense_rank,
            hybrid_rank=hit.rank,
        )
        for rank, (hit, score) in enumerate(scored[:limit], start=1)
    ]
