# Phase 4 report — Reranking hybrid candidates with a cross-encoder

On **324 NFCorpus development queries**, the cross-encoder raised Recall@10 from **0.129586 to 0.151834** and MRR@10 from **0.498738 to 0.537592**. Hybrid and reranked Recall@50 were both **0.235890**. Every query retained exactly the same 50 passage IDs. The saved rankings let you reproduce these results without running a model. [Development rankings](results/phase_4_dev.csv)

## 1. Concepts

A **bi-encoder** computes a vector for a passage independently of any question. That makes it practical to store all 3,633 passage vectors once and search them for each new query. **BM25** contributes a second ranking using word matches. **Reciprocal Rank Fusion (RRF)** combines their positions to choose 50 candidates. [Phase 2 report](PHASE_2_REPORT.md) · [Phase 3 report](PHASE_3_REPORT.md)

A **cross-encoder** reads the query and one passage together. Its attention layers can relate a word in the question directly to words in that passage. A classification head produces one relevance score for the pair. The same passage needs a fresh score for a different query. Reranking after RRF limits this cost to at most 50 pairs per search. The selected model is a pretrained MS MARCO passage ranker. [Pinned model checkpoint](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/tree/233902d25c440f23af6f7d6e94d2946bac0bee0a)

## 2. Architecture and data flow

| Stage | Input and output |
|---|---|
| Retrieve | One query goes to the existing BM25 and dense retrievers. BM25 returns up to 50 matching passage IDs. Dense search returns up to 50 by ascending pgvector cosine distance. |
| Fuse | RRF merges the two lists by passage ID with its existing constant of 60. It retains the best 50 unique passages. Each hybrid hit already carries the stored title, complete abstract body, and its source ranks. |
| Prepare inputs | For each fused hit, build `(query, title + newline + body)` using the existing passage formatter. Candidate order matches pair order. |
| Score | Load the pinned cross-encoder once per Python process. Submit all pairs in batches of 16 with padding and truncation. Receive one raw logit per pair and validate the output. |
| Order and display | Attach each score to its input hit. Sort by descending logit, then ascending passage ID, assign 1-based ranks, and apply the display limit. The JSON preserves the ID, title, body, hybrid rank, BM25 rank, and dense rank. |
| Evaluate | Retrieve and fuse once per judged development query. The baseline and reranker share that fused list. Save the ordered IDs for all four methods and score them with the Phase 1 metrics. |

Hybrid search, reranked search, and development evaluation use a read-only, repeatable-read database transaction. A candidate’s ID and text come from the same snapshot. [CLI transaction](rag/__main__.py#L57)

## 3. Files created or modified

| File | Purpose |
|---|---|
| [rag/rerank.py](rag/rerank.py) | New checkpoint loader, reranked result type, pair scoring, validation, sorting, and error reporting. |
| [rag/__main__.py](rag/__main__.py) | Adds `search reranked` and `evaluate-dev --reranked` and routes retrieval methods through a shared CLI dispatcher. |
| [rag/evaluate_dev.py](rag/evaluate_dev.py) | Adds an opt-in four-method comparison, full-depth ranking export, explicit per-query candidate check, and Recall@50. The existing three-method command keeps its previous output format. |
| [rag/dense.py](rag/dense.py#L20) | Accepts optional `RAG_DENSE_DEVICE`. Setting it to `cpu` reproduces Phase 3’s dense-query inference while allowing the cross-encoder to use the GPU. With the variable unset, device selection remains automatic. |
| [tests/test_rerank.py](tests/test_rerank.py) | Exercises ordered pairs, score-to-ID mapping, ties, truncation of the displayed list, metadata, the Recall@50 invariant, invalid inputs, and model failures. |
| [tests/test_evaluate_dev.py](tests/test_evaluate_dev.py#L38) | Adds a four-method fixture with a known metric change and an injected candidate substitution. |
| [tests/test_live_retrieval.py](tests/test_live_retrieval.py#L77) | Observes a real cross-encoder call and checks its scores and source fields against the actual CLI output. |
| [results/phase_4_dev.csv](results/phase_4_dev.csv) | One row per query, with ordered passage IDs for BM25, dense, hybrid, and reranked retrieval. Each list has at most 50 IDs. |
| [PHASE_4_REPORT.md](PHASE_4_REPORT.md) | Implementation and verification record. The root-level `phase_4.zip` is its companion review archive. |

The existing pinned dependencies provide both encoders. The PostgreSQL schema, corpus, dense embeddings, RRF settings, and original implementation plan were sufficient for this phase.

## 4. Code walkthrough

1. **Choose the candidate depth.** In the reranked CLI route, `fuse_rankings(bm25, dense)` uses its default depth of 50. The requested `limit` is passed to `rerank`. This placement is what allows hybrid rank 12 to become a displayed top-five result. [CLI dispatcher](rag/__main__.py#L90) · [Fusion algorithm](rag/hybrid.py#L39)
2. **Load the model.** `load_cross_encoder` uses `lru_cache(maxsize=1)`, pins revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`, sets a 512-token limit, and chooses an identity activation. Its disk cache is under `data/hf/hub` unless `HF_HOME` selects another location. The process cache saves loading time across queries, while the disk cache saves downloads across processes. [Loader](rag/rerank.py#L31)
3. **Build and score pairs.** `rerank` validates the inputs, returns an empty list for zero candidates, and uses `passage_text` for each nonempty pair. `predict` handles batches and returns scores in pair order. The implementation requires a one-dimensional output with exactly one finite value per candidate. A loading or scoring exception is wrapped as `RerankError` with the original cause. [Reranking function](rag/rerank.py#L43) · [Passage formatter](rag/retrieval.py#L20)
4. **Keep score and identity together.** `zip(candidates, scores, strict=True)` binds each hit to its score before sorting. The sort key is `(-score, passage_id)`. The inherited `rank` and `score` describe the new order, and `hybrid_rank` records the old position. Missing upstream ranks stay null. [Sorting and output](rag/rerank.py#L64) · [Result type](rag/rerank.py#L22)
5. **Check the invariant.** The evaluator compares the length and the ID set of the full reranked list with the hybrid list for **each query** before accepting that run. It then computes Recall@50 independently for both methods. An aggregate equality alone could hide errors that cancel across queries. [Per-query check](rag/evaluate_dev.py#L47) · [Metric calculation](rag/evaluate_dev.py#L57)

## 5. Design decisions and observed model behavior

- **Score interpretation.** The model supplies a single classification logit per pair. Identity activation preserves that value, and softmax is disabled. Higher is better. Negative values are valid. Scores are used to order passages for the same question and do not represent calibrated probabilities.
- **Input length.** The 512-token budget covers the query, passage, and tokenizer special tokens together. In the installed Sentence Transformers implementation, `predict` calls the tokenizer with `padding=True` and `truncation=True`. For paired text, the tokenizer uses longest-first truncation and removes tokens from the right. With a short question and long abstract, this usually cuts the end of the abstract. The CLI still returns the complete database text.
- **Measured truncation and batching.** For development query `PLAIN-912` (“cilantro”), **3 of 50** pairs exceeded 512 tokens before truncation. The longest was 624 tokens and became 512. A forward-pass hook observed batch shapes **16×512, 16×466, 16×512, and 2×292**. All 50 pairs reached the model even though only 10 results were displayed. The first two CPU logits were −0.45577 for `MED-1442` and −1.43768 for `MED-1441`.
- **Failure behavior.** Empty candidates avoid a model load. A blank question, nonpositive limit, duplicate IDs, or a list longer than 50 raises an input error. Missing, nonfinite, or wrongly shaped scores fail the request before ranking. A model failure produces a diagnostic rather than a partially scored list.
- **Runtime choice.** The 16,200-pair run used an NVIDIA GeForce RTX 3050 Laptop GPU for reranking and CPU for dense queries. An optional official `torch==2.9.1+cu126` build was installed in the ignored `.bootstrap` folder. The project virtual environment retained its original CPU build. The GPU speeds up this run and is optional for normal use.

## 6. Development evaluation

All four methods used the same **324 queries**, **3,633 passages**, and published development judgments. The evaluator treated a positive judgment as relevant and averaged over all judged queries. It computed Recall@K as the fraction of a query’s judged-positive passages present within K. MRR@10 averages the reciprocal rank of the first positive in the top ten, with zero for a miss. [Metric definitions in source](rag/metrics.py#L8) · [Development evaluator](rag/evaluate_dev.py#L28)

| Method | Recall@10 | MRR@10 | Recall@50 |
|---|---:|---:|---:|
| BM25 | 0.109858 | 0.464605 | 0.179698 |
| Dense | 0.118067 | 0.480242 | 0.215056 |
| Hybrid RRF | 0.129586 | 0.498738 | 0.235890 |
| Hybrid + cross-encoder | **0.151834** | **0.537592** | **0.235890** |

Reranking gained **0.022248 absolute Recall@10** (about **2.22 percentage points**) and **0.038853 absolute MRR@10** over RRF. Per-query Recall@10 improved on **86**, declined on **42**, and tied on **196** queries. Reciprocal rank@10 improved on **68**, declined on **42**, and tied on **214**.

Every fused list contained 50 unique passages, and its reranked list contained the same IDs. Both methods therefore found the same number of judged-positive passages at depth 50 for **324 of 324 queries**. Each first-stage method’s top ten also matched its saved Phase 3 ranking for every query. These are development measurements; performance on the final test split remains unmeasured. [Phase 3 rankings](results/phase_3_dev.csv) · [Phase 4 rankings](results/phase_4_dev.csv)

## 7. Examples from the measured rankings

**Improvement — `PLAIN-3280`, “Brown Rice vs. Black Rice.”** Of seven judged-positive passages, RRF placed one in the top ten. The cross-encoder placed three there. Two papers about black rice pigments or pigmented rice bran moved across the display cutoff. The first positive also moved from fourth to second. Recall@10 rose from **1/7 to 3/7**, and reciprocal rank@10 rose from **1/4 to 1/2**.

| Passage | Development grade | RRF rank | Reranked rank | Observed logit |
|---|---:|---:|---:|---:|
| MED-4905, black rice pigment supplementation | 2 | 4 | 2 | 1.73877 |
| MED-4907, black rice anthocyanins | 2 | 12 | 5 | −0.74075 |
| MED-4906, pigmented rice bran extracts | 2 | 11 | 6 | −0.87663 |

The biggest moves in this query were outside the displayed top ten. `MED-5328`, a study of vegetarian diets and diabetes incidence, moved from 45 to 25. `MED-3769`, about food cost, energy density, and nutrient adequacy, moved from 25 to 49. Neither had a positive judgment for this query. Tracking movement and checking judgments answers two different questions.

**Regression — `PLAIN-2549`, “How the Egg Board Designs Misleading Studies.”** RRF’s three top-ten positives were `MED-2374` at rank 1, `MED-4324` at rank 8, and `MED-2372` at rank 10. They fell to **12, 27, and 14**. `MED-1890`, a meta-analysis about eggs and cholesterol, rose from 22 to 1 but carried no positive judgment for this query. All three former top-ten positives remained available at depth 50. Recall@10 fell from **3/8 to 0**, and reciprocal rank@10 fell from **1 to 0**. The titles show topical overlap among candidates; the saved rankings establish the movements, without establishing the model’s internal reasoning.

## 8. Tests and verification

| Executed check | Observed result |
|---|---|
| `.\.venv\Scripts\python.exe -X utf8 -m pytest -q tests` with `RAG_LIVE_TESTS=1`, `HF_HUB_OFFLINE=1`, and `OMP_NUM_THREADS=4` | **37 passed in 28.64 seconds** on the final CPU run. A CUDA run with dense queries on CPU also passed all **37** cases in 11.70 seconds. This includes all 26 previous tests. |
| Three deliberately different mocked logits and a separate equal-score case | Correct passage–score mapping, preserved titles and prior ranks, and deterministic ascending-ID tie order. [Mapping test](tests/test_rerank.py#L23) · [Tie test](tests/test_rerank.py#L54) |
| Reverse the order of 50 candidates, then request one displayed hit | All 50 were scored before truncation. Full ID sets and Recall@50 were identical; Recall@10 changed in the fixture. [Invariant test](tests/test_rerank.py#L38) |
| Exercise failure paths | Empty input skipped the model. Invalid input, wrong score count or shape, NaN, infinity, and an injected runtime failure produced the expected outcomes. A same-length candidate substitution was rejected by the evaluator. [Failure tests](tests/test_rerank.py#L60) · [Evaluator test](tests/test_evaluate_dev.py#L38) |
| Inspect real inputs, scores, and output fields | The real CLI test observed all 50 ordered input pairs for “cilantro,” requested five results, and independently sorted the captured logits to check every displayed field. [Live regression](tests/test_live_retrieval.py#L77) |
| `python -X utf8 -m rag search reranked 'Brown Rice vs. Black Rice' --limit 10` | Returned ten results, including `MED-4907` at rank 5 from RRF rank 12. |
| `python -X utf8 -m rag evaluate-dev --reranked --save-rankings results/phase_4_dev.csv` | Completed all 324 queries. A second full inference run produced byte-identical rankings. Offline rescoring of the CSV reproduced all twelve table entries. |
| `python -X utf8 -m rag evaluate-dev --save-rankings <temporary CSV>` | The legacy three-method command reproduced the Phase 3 CSV byte for byte. |

The saved four-method CSV is **539,783 bytes**. Its SHA-256 is `9d0c26ca239bc714382710645dc0e259bf2f955fea760e590a20423614103ed8`. Model scores can vary slightly with hardware or PyTorch build. Evidence beyond the input token limit and passages excluded by RRF cannot affect the reranked result.

## 9. Reproduce in PowerShell

Run from the project root. On a fresh checkout, first follow the [Phase 2 setup](PHASE_2_REPORT.md#7-how-to-run) to start PostgreSQL, download NFCorpus, ingest the passages, and build the dense embeddings. In the prepared workspace, the following commands run the feature using the normal project environment:

```powershell
docker compose up -d --wait db
$env:RAG_DENSE_DEVICE = 'cpu'
$env:OMP_NUM_THREADS = '4'
$env:HF_HUB_OFFLINE = '0'
.\.venv\Scripts\python.exe -X utf8 -c "from rag.rerank import load_cross_encoder; load_cross_encoder()"
$env:HF_HUB_OFFLINE = '1'
.\.venv\Scripts\python.exe -X utf8 -m rag search reranked "Brown Rice vs. Black Rice" --limit 10
.\.venv\Scripts\python.exe -X utf8 -m rag evaluate-dev --reranked --save-rankings results/phase_4_dev.csv
$env:RAG_LIVE_TESTS = '1'
.\.venv\Scripts\python.exe -X utf8 -m pytest -q tests
```

To reproduce the **measured GPU configuration** in this workspace, use the CUDA build already available locally:

```powershell
$env:PYTHONPATH = (Resolve-Path '.bootstrap\torch-cu126').Path
$env:RAG_DENSE_DEVICE = 'cpu'
$env:OMP_NUM_THREADS = '4'
$env:HF_HUB_OFFLINE = '1'
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.\.venv\Scripts\python.exe -X utf8 -m rag evaluate-dev --reranked --save-rankings results/phase_4_dev.csv
```

For a fresh Windows Python 3.11 environment with a compatible NVIDIA driver, install the optional GPU wheel before setting `PYTHONPATH`. The CPU instructions above are sufficient elsewhere.

```powershell
.\.venv\Scripts\python.exe -m pip install --target .bootstrap\torch-cu126 --no-deps --index-url https://download.pytorch.org/whl/cu126 "torch==2.9.1+cu126"
```

To verify the result **without either model or PostgreSQL**, use the saved rankings and downloaded development judgments:

```powershell
@'
import csv
from pathlib import Path
from rag.dataset import read_qrels
from rag.metrics import evaluate_at_k

labels = read_qrels(Path("data/nfcorpus/dev.tsv"))
with open("results/phase_4_dev.csv", newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == len(labels) == 324 and {r["query_id"] for r in rows} == labels.keys()
for row in rows:
    h, r = row["hybrid"].split(), row["reranked"].split()
    assert len(h) == len(r) == len(set(h)) == 50 and set(h) == set(r)
for method in ("bm25", "dense", "hybrid", "reranked"):
    run = {r["query_id"]: r[method].split() for r in rows}
    ten = evaluate_at_k(labels, run, 10)
    fifty = evaluate_at_k(labels, run, 50)
    print(method, ten.recall, ten.mrr, fifty.recall)
'@ | .\.venv\Scripts\python.exe -X utf8 -
```

## 10. Interview preparation

| Question | Answer to demonstrate |
|---|---|
| Why use a second transformer after dense search? | The bi-encoder can reuse stored passage vectors. The cross-encoder jointly processes query and passage, allowing more detailed matching on a bounded shortlist. |
| Why would passing `--limit 10` to RRF before reranking be a bug? | Candidates 11–50 would be removed before they could be scored. In the rice query, that would have excluded two judged passages that finished in the top six. |
| Why can Recall@10 change while Recall@50 stays identical? | The first ten positions contain a different subset after sorting. The full set of 50 is the same for each query, so its intersection with the judged-positive IDs has the same size. |
| What would a Recall@50 mismatch tell you? | Inspect dropped IDs, duplicates, candidate depth, query membership, and qrels. A pure permutation alone cannot produce that mismatch. |
| Can a negative logit be a good hit? | Yes. The ranking uses relative order within the query. `MED-4907` ranked fifth with −0.74075. |
| What happens to an important sentence at the end of a long abstract? | It may be truncated during scoring. Displaying the complete source body does not mean the model processed every token. |

## 11. Completion checklist

| Requirement | Implementation and verification |
|---|---|
| Named model and one load per process | Pinned checkpoint, cached loader, raw-logit activation, and live identity check. [Loader](rag/rerank.py#L31) |
| Reuse BM25, dense, RRF, and title + body | Existing retrieval and fusion functions supply the fixed candidate set and stored text. [Dispatcher](rag/__main__.py#L90) · [Pair preparation](rag/rerank.py#L53) |
| Batch scoring, descending order, ID ties, preserved metadata | Batch size 16, strict score validation, stable ID association, and all source ranks on the result. [Scorer](rag/rerank.py#L43) · [Tests](tests/test_rerank.py) |
| `search reranked` and independent display limit | Both the five-result live test and ten-result real probe observed all 50 candidate pairs. [Live test](tests/test_live_retrieval.py#L77) |
| Invalid inputs, empty candidates, failure handling | Checked directly with injected invalid outputs and exceptions. [Failure cases](tests/test_rerank.py#L60) |
| Same IDs and Recall@50 | Asserted in fixtures and independently verified for all 324 queries in the exported run. [Evaluation guard](rag/evaluate_dev.py#L47) |
| Four-method evaluation and interpretable examples | Section 6 gives all twelve metrics. Section 7 traces both a gain and a loss to specific passage positions. [Full ranked IDs](results/phase_4_dev.csv) |
| Preserve earlier functionality and produce review materials | Original tests passed, the legacy evaluator reproduced its artifact, and `phase_4.zip` packages the complete source, tests, configuration, plan, four reports, and both development rankings. |
