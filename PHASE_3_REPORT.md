# Phase 3 report — Hybrid retrieval with Reciprocal Rank Fusion

On **324 NFCorpus development queries**, combining BM25 and dense search raised mean Recall@10 from **0.118067** for dense alone to **0.129586**, and MRR@10 from **0.480242** to **0.498738**. The averages hide both gains and losses. The “cilantro” and “walnut” examples in section 7 show how the same fusion rule produced opposite outcomes. All **26 tests passed**, and a second complete development run reproduced the saved rankings byte for byte. [Development rankings](results/phase_3_dev.csv)

## 1. Concepts

Hybrid retrieval uses multiple ways to look for evidence. In this project, **BM25** finds passages by word overlap and rewards distinctive terms. **Dense search** finds passages whose transformer embeddings point in a similar direction to the query embedding. BM25 assigns higher scores to better matches, while pgvector assigns smaller cosine distances to closer matches. [Phase 2 implementation and examples](PHASE_2_REPORT.md)

**Reciprocal Rank Fusion (RRF)** gives a passage one contribution per list containing it. Rank 1 contributes `1/61`, rank 2 contributes `1/62`, and a missing passage contributes zero. Adding these contributions produces a common sorting score. The score depends on rank positions, so a very large BM25 score has exactly the same influence as any other BM25 rank-1 result. An RRF score is an ordering statistic, with no probability interpretation. [Original RRF paper](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)

## 2. Architecture and data flow

| Stage | What happens to one query |
|---|---|
| Read the corpus | The hybrid CLI opens a read-only, repeatable-read PostgreSQL transaction. BM25 and dense search see the same database snapshot. |
| Retrieve independently | BM25 tokenizes the query and returns up to 50 matching passages. The dense retriever encodes the same query, then PostgreSQL returns up to 50 passages in ascending cosine distance. Both use Phase 2’s `SearchHit` with the original passage ID and 1-based rank. |
| Form the candidate set | Index each source list by passage ID and keep its first occurrence. The union contains at most 100 distinct IDs. |
| Fuse | Add `1 / (60 + source_rank)` once for each source containing the ID. Sort by descending sum, then ascending passage ID. |
| Return and inspect | Return the first `min(requested_limit, 50)` passages. The JSON includes `passage_id`, `bm25_rank`, `dense_rank`, `rank` (fused rank), `score` (RRF score), `score_type="rrf"`, `title`, and `body`. A missing source rank is JSON `null`. |

The evaluation command builds the BM25 index once for the entire development run. For each query it retrieves both lists once, saves their top ten as the two baselines, and feeds those same candidate lists into fusion. Relevance judgments enter only when scoring the resulting rankings. [Candidate retrieval](rag/hybrid.py#L23) · [Development evaluation](rag/evaluate_dev.py#L17)

## 3. Files created or modified

| File | Purpose |
|---|---|
| [rag/hybrid.py](rag/hybrid.py) | Implements fixed-depth candidate retrieval, the hybrid hit type, per-source deduplication, and RRF. |
| [rag/evaluate_dev.py](rag/evaluate_dev.py) | Runs the three-method comparison on `dev.tsv`, applies the Phase 1 metrics, and optionally exports top-ten rankings. |
| [rag/__main__.py](rag/__main__.py) | Adds `search hybrid` and `evaluate-dev` to the existing CLI and selects a consistent database snapshot for both commands. |
| [tests/test_hybrid.py](tests/test_hybrid.py) | Covers the hand example, missing lists, duplicates, exact ties, and candidate and display limits. |
| [tests/test_evaluate_dev.py](tests/test_evaluate_dev.py) | Exercises development-only evaluation with a tiny labeled fixture and verifies both metrics and the exported file. |
| [tests/test_live_retrieval.py](tests/test_live_retrieval.py) | Adds a live hybrid CLI test using the real Phase 2 top-50 lists as independent inputs to its expected result. |
| [.gitignore](.gitignore) | Removes the blanket Markdown ignore rule so the plan and reports are eligible for Git. The root `/phase_*.zip` rule still excludes review archives. |
| [results/phase_3_dev.csv](results/phase_3_dev.csv) | Stores one row per development query. Each method column contains space-separated passage IDs, best first. The file supports offline rescoring. |
| [PHASE_3_REPORT.md](PHASE_3_REPORT.md) | Records the implementation, measurements, teaching examples, and reproduction commands. |

The root-level `phase_3.zip` contains the complete review snapshot. Source files from previous phases are included alongside the plan, all three reports, tests, configuration, and the development rankings.

## 4. Code walkthrough

1. **Start from the Phase 2 contract.** `SearchHit` carries the passage ID, rank, native score, score type, title, and body. `HybridHit` inherits those fields and adds two optional source ranks. This lets the same CLI serializer handle all three methods. [SearchHit](rag/retrieval.py#L11) · [HybridHit](rag/hybrid.py#L18)
2. **Fetch both lists.** `retrieve_candidates` calls `BM25Retriever.search` and `search_dense` with `CANDIDATE_DEPTH = 50`. It has no display-limit argument, so requesting one visible row cannot accidentally shrink the candidate lists. [Candidate retrieval](rag/hybrid.py#L23) · [BM25 search](rag/retrieval.py#L42) · [Dense search](rag/dense.py#L64)
3. **Count each source once.** `_first_hits` reads a source ranking in best-first order. It keeps the first occurrence of each ID, including its original 1-based rank. If a malformed list contains `A, A, B`, A votes once at rank 1 and B retains rank 3. [Source deduplication](rag/hybrid.py#L29)
4. **Calculate scores and assign ranks.** `fuse_rankings` accumulates a rational score for each ID, sorts the union, applies the limit, and numbers the output starting at 1. It copies the source title and body alongside the two source ranks. [Fusion implementation](rag/hybrid.py#L39)
5. **Score the comparison.** `evaluate_dev` reads query text and development judgments, generates the three rankings, and calls the existing `evaluate_at_k`. That function averages over all judged query IDs. A query returning no relevant passage still contributes zero to both means. [Run construction](rag/evaluate_dev.py#L17) · [Aggregation](rag/metrics.py#L28) · [Per-query arithmetic](rag/metrics.py#L8)

## 5. Calculate the plan’s example by hand

Dense returns `A, B`. BM25 returns `B, C`. A dash means the passage was absent from that source.

| Passage | Dense rank | BM25 rank | Dense contribution | BM25 contribution | RRF score | Fused rank |
|---|---:|---:|---:|---:|---:|---:|
| B | 2 | 1 | 1/62 | 1/61 | 123/3782 ≈ 0.032522475 | 1 |
| A | 1 | — | 1/61 | 0 | 0.016393443 | 2 |
| C | — | 2 | 0 | 1/62 | 0.016129032 | 3 |

B appears once and receives credit from both sources. If BM25 returns an empty list, dense ranks 1 and 2 receive `1/61` and `1/62` respectively. Because the reciprocal decreases with rank, the dense order survives fusion. [Executable hand example and empty-list test](tests/test_hybrid.py#L14)

## 6. Design decisions

- **Fixed constant and depth.** The plan specifies a fusion constant of 60, up to 50 candidates per source, and up to 50 fused results. The evaluation cutoff of 10 controls what is scored. It leaves all 50 candidates eligible to contribute before truncation.
- **Agreement has a concrete effect.** Even a passage ranked 50th in both lists earns `2/110 ≈ 0.018182`. That exceeds `1/61 ≈ 0.016393` for a passage ranked first in just one list. With these settings, every overlapping passage outranks every single-source passage. This explains both some recoveries and some losses.
- **Deterministic ties.** Python’s standard-library `Fraction` preserves exact sums through sorting. For example, `1/66 + 1/99` and `1/72 + 1/88` are mathematically equal, though direct floating-point addition can produce different last bits. Ascending passage ID decides the tie. Conversion to a float happens when constructing the output hit. [Tie regression](tests/test_hybrid.py#L34)
- **Bounded work.** For U unique candidates, fusion stores O(U) entries and sorts in O(U log U) time. Here U is at most 100. The database schema, model, existing retrievers, and dependency list already provided everything needed.

## 7. Development-set evaluation

All methods used the same 3,633 embedded passages, the same 324 judged development queries, the original query text, and the same `dev.tsv`. A positive qrel counts as relevant. For each query, Recall@10 is the fraction of its judged-positive passages found in the first ten. Reciprocal rank@10 is the inverse position of its first relevant hit within ten, or zero for a miss. The table shows the means across all 324 queries. [Saved rankings](results/phase_3_dev.csv) · [Scoring rules](rag/metrics.py#L8)

| Method | Queries | Recall@10 | MRR@10 |
|---|---:|---:|---:|
| BM25 | 324 | 0.109858 | 0.464605 |
| Dense | 324 | 0.118067 | 0.480242 |
| Hybrid RRF | 324 | **0.129586** | **0.498738** |

Hybrid gained **0.011519 absolute Recall@10**, or approximately **1.15 percentage points**, over dense. Its absolute MRR@10 gain was **0.018496**. Comparing queries individually, recall improved on **86**, declined on **46**, and tied on **192**. Reciprocal rank improved on **59**, declined on **53**, and tied on **212**.

### Where fusion helped — `PLAIN-912`, “cilantro”

There are four judged-positive passages. BM25 retrieved just three matching passages overall. For `MED-1441`, a study about salsa components, BM25 rank 2 and dense rank 13 combined to give `1/62 + 1/73 ≈ 0.029828`. That brought it to fused rank 1. The other recovered positive, `MED-1442`, was BM25 rank 1 and absent from the dense top 50. It tied with dense-only rank-1 passage `MED-3510`, and the ID tie-break put `MED-1442` at fused rank 2.

| Method | Judged positives in top 10 | Recall@10 | Reciprocal rank@10 |
|---|---|---:|---:|
| BM25 | MED-1442, MED-1441 | 2/4 | 1 |
| Dense | None | 0 | 0 |
| Hybrid | MED-1441, MED-1442 | 2/4 | 1 |

### Where fusion hurt — `PLAIN-2353`, “walnut”

There are six judged-positive passages. Dense found `MED-2597`, *Should we go nuts about nuts?*, at rank 7. BM25 returned 16 matches and omitted this passage, leaving it with one contribution of `1/67 ≈ 0.014925`. It fell to fused rank 14. By comparison, `MED-2152` appeared first in both lists and `MED-2379` appeared at BM25 rank 2 and dense rank 5. They took fused ranks 1 and 2. Neither carries a positive qrel for this query.

| Method | Judged positives in top 10 | Recall@10 | Reciprocal rank@10 |
|---|---|---:|---:|
| BM25 | None | 0 | 0 |
| Dense | MED-2597 | 1/6 | 1/7 |
| Hybrid | None | 0 | 0 |

These examples use published judgments. A topically plausible passage can receive zero evaluation credit if it has no positive label. The measured gain applies to this NFCorpus development split and these fixed settings. Final-test performance remains unmeasured.

## 8. Executed tests and checks

| Check | Observed result |
|---|---|
| `python -X utf8 -m pytest -q tests` with `RAG_LIVE_TESTS=1`, `HF_HUB_OFFLINE=1`, and `OMP_NUM_THREADS=4` | **26 passed in 13.77 seconds**, including all 20 existing tests. |
| Four new fusion unit tests | Verified the hand-calculated scores, single-source and overlapping IDs, duplicate votes, empty lists, exact ties, and top-K truncation. The depth test makes a double rank-50 passage win even when only one fused hit is displayed. [Unit tests](tests/test_hybrid.py) |
| Development evaluator fixture | Verified all three metric pairs by hand, retained a query with empty results, and exported its empty CSV fields. The fixture supplied only development judgments. [Evaluator test](tests/test_evaluate_dev.py#L10) |
| Live hybrid CLI test | Rebuilt expected scores from independent real top-50 CLI outputs and matched the fused top ten, both source ranks, float scores, and full passage text. [Live test](tests/test_live_retrieval.py#L54) |
| `python -X utf8 -m rag search hybrid 'cilantro' --limit 10` | Returned ten inspectable hits. The first hit was `MED-1441` with BM25 rank 2, dense rank 13, and RRF score 0.029827662395050816. |
| `python -X utf8 -m rag evaluate-dev --save-rankings results/phase_3_dev.csv` | Completed the full split. Rescoring the CSV reproduced the three metric pairs. A second full run to a temporary file yielded byte-identical rankings. |

The rankings CSV is 85,493 bytes. Its SHA-256 for this run is `4ba14d5c482d02ccdd9dc974d81786d54e3fa98ce08d855536d4f3cdd7424d07`. The file records the evaluated top ten. To inspect candidates below rank ten, rerun search with `--limit 50`.

## 9. Reproduce with PowerShell

Start in the project root. A fresh checkout needs the environment, private database password file, dataset, and embeddings from the [Phase 2 setup instructions](PHASE_2_REPORT.md#7-how-to-run). For an already populated workspace, run:

```powershell
docker compose up -d --wait db
$env:HF_HUB_OFFLINE = '1'
$env:OMP_NUM_THREADS = '4'
.\.venv\Scripts\python.exe -X utf8 -m rag search hybrid "cilantro" --limit 10
.\.venv\Scripts\python.exe -X utf8 -m rag search hybrid "walnut" --limit 50
.\.venv\Scripts\python.exe -X utf8 -m rag evaluate-dev --save-rankings results/phase_3_dev.csv
$env:RAG_LIVE_TESTS = '1'
.\.venv\Scripts\python.exe -X utf8 -m pytest -q tests
```

To rescore the saved rankings without starting PostgreSQL or loading the model, use the downloaded development judgments and the existing metric code:

```powershell
@'
import csv
from pathlib import Path
from rag.dataset import read_qrels
from rag.metrics import evaluate_at_k

labels = read_qrels(Path("data/nfcorpus/dev.tsv"))
with open("results/phase_3_dev.csv", newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == len(labels) and {r["query_id"] for r in rows} == labels.keys()
for method in ("bm25", "dense", "hybrid"):
    runs = {r["query_id"]: r[method].split() for r in rows}
    print(method, evaluate_at_k(labels, runs, 10))
'@ | .\.venv\Scripts\python.exe -X utf8 -
```

## 10. Interview preparation

| Question | Answer using this implementation |
|---|---|
| Why fuse ranks? | Each retriever supplies a meaningful ordering. Rank-based contributions put both lists on a common scale, without estimating a score conversion. The tradeoff is losing information about the margin between adjacent hits. |
| Why fetch 50 when the user asks for 10? | A passage below rank 10 in either list can enter the fused top ten. `MED-1441` came from dense rank 13 and became fused rank 1. |
| What if a document appears twice, or only one method finds it? | The original passage ID identifies it. It receives at most one contribution per source. A missing source contributes zero and appears as `null` in the result. |
| Can RRF recover evidence missed by both? | The candidate set is the union of the two top-50 lists. An ID outside that union has no route into the output. |
| Why might recall drop even when both sources agree? | Agreement raises a passage’s rank independent of its relevance label. For “walnut,” shared candidates moved ahead of a relevant dense-only result and pushed it outside the cutoff. |
| What evidence supports the résumé claim? | Using the same labels, cutoff, and queries, RRF achieved 0.129586 Recall@10 versus 0.118067 for dense and 0.109858 for BM25. Quote the corpus and split with the numbers, and be ready to explain one gain and one loss. |
