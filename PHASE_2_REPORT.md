# Phase 2 report — BM25 and dense retrieval

The project now has two independent search methods over NFCorpus. Run `python -m rag search bm25` to inspect word matches, or `python -m rag search dense` to inspect nearest neighbors of an encoded question. Both return ranked source passages. This phase implements the second milestone of the [implementation plan](ADVANCED_RAG_IMPLEMENTATION_PLAN.md) on top of the [Phase 1 database and evaluator](PHASE_1_REPORT.md).

## 1. Concepts implemented

**BM25.** Imagine searching for a distinctive name that appears in just one abstract. Matching that name provides more evidence than matching a word found throughout the corpus. Okapi BM25 weights query terms by how many documents contain them. Repeated occurrences within one passage help with diminishing returns, and passage length is measured relative to the corpus average. We use the `rank_bm25` implementation with its defaults, including `k1=1.5` and `b=0.75`. A larger score ranks earlier. [Library source](https://github.com/dorianbrown/rank_bm25/blob/master/rank_bm25.py)

**Dense retrieval.** The same pretrained transformer independently encodes a passage and a question. Its tokenizer splits text into model tokens, and its transformer calculates a representation for each token in context. This checkpoint averages the token representations using the attention mask, then normalizes the result to unit length. Each input becomes one vector of 384 floating-point values. This lets texts with different wording be neighbors. Passage vectors can be reused across questions because encoding a passage does not require knowing the question. [Pinned model card](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1/tree/b207367332321f8e44f96e224ef15bc607f4dbf0)

**Cosine distance.** Cosine similarity is the dot product divided by the product of the two vector lengths. pgvector’s `<=>` operator returns `1 − cosine_similarity`. Parallel vectors have distance 0, perpendicular vectors 1, and opposite vectors 2. Lower distances rank earlier. These are different units from BM25 scores, so every search result identifies its score type. [pgvector operators](https://github.com/pgvector/pgvector#querying)

## 2. Architecture and data flow

| Operation | Data flow |
|---|---|
| Common input | Phase 1 loads one PostgreSQL `passages` row per source ID. Both retrievers join the title, a newline, and the abstract body in that order. The original ID, title, and body travel back with each hit. |
| Build BM25 | Read `passage_id, title, body` in ID order → tokenize each joined passage → construct `BM25Okapi` → retain the source rows in the same positions as the indexed documents. |
| Search BM25 | Tokenize the query → score the corpus → keep only documents sharing at least one query token → sort by descending BM25 score, then ascending passage ID → take up to the requested limit. |
| Build dense vectors | Read and lock a batch of rows with `NULL` embeddings → join title and body → encode with the pinned transformer → validate the vector batch → write each vector to the matching primary key → commit the batch and repeat. |
| Search dense | Check for a complete embedding set → encode the question once → bind its vector to a parameterized SQL query → compute exact cosine distances in PostgreSQL → sort by ascending distance, then ascending ID → return the requested neighbors. |
| Display | Both paths construct the same `SearchHit` objects. The CLI serializes them as a JSON array with `passage_id`, `rank`, `score`, `score_type`, `title`, and `body`. Ranks start at 1. No matches produces `[]`. |

An in-process caller can issue several queries to the same `BM25Retriever`. A new CLI invocation starts a new Python process and rebuilds that in-memory index. The transformer is loaded once per process on first use. The model files survive in the disk cache and the passage vectors survive in the database volume.

## 3. Files created or modified

| File | Purpose in Phase 2 |
|---|---|
| [rag/retrieval.py](rag/retrieval.py) | New shared result type, passage formatting, lexical tokenizer, and BM25 retriever. |
| [rag/dense.py](rag/dense.py) | New pinned encoder, batch validation, resumable storage, and exact dense search. |
| [rag/__main__.py](rag/__main__.py) | Adds `embed` and `search` to the existing CLI. Reads the local database password from a private file and preserves the `RAG_DATABASE_URL` override. |
| [requirements.txt](requirements.txt) | Pins rank-bm25, the pgvector Psycopg adapter, NumPy, Sentence Transformers, Transformers, and PyTorch alongside the Phase 1 dependencies. |
| [compose.yaml](compose.yaml) | Configures PostgreSQL to read the local development password from a Compose secret backed by the ignored `.env.db-password` file. The database, port, and named volume retain their Phase 1 settings. |
| [tests/conftest.py](tests/conftest.py) | Shares the existing isolated PostgreSQL fixture across the storage and retrieval suites. |
| [tests/test_storage.py](tests/test_storage.py) | Uses the shared fixture and adds a regression test for re-ingesting passages after embedding. |
| [tests/test_retrieval.py](tests/test_retrieval.py) | New small-corpus BM25, SQL cosine-ordering, batch retry, tie, and boundary tests. |
| [tests/test_live_retrieval.py](tests/test_live_retrieval.py) | New opt-in tests run both real CLI routes on two questions, resolve every displayed passage, and independently calculate the dense distances. |
| [PHASE_2_REPORT.md](PHASE_2_REPORT.md) | Teaching walkthrough, observed results, and reproduction commands. |

The existing [schema](rag/schema.sql#L3) already declares `embedding vector(384)`. The original [ingestion function](rag/storage.py#L10) already clears a vector when its source text changes. The existing [.gitignore](.gitignore) already includes `/phase_*.zip` and `.env.*`. The local password was moved outside the source so the review ZIP can include the exact runnable source files. For an existing database volume, the local password file must contain that volume’s existing password.

## 4. Code walkthrough

| Source location | What to follow |
|---|---|
| [SearchHit, line 11](rag/retrieval.py#L11) | The immutable common output. A BM25 hit uses `score_type="bm25"`. A dense hit uses `"cosine_distance"`. Both carry enough text to inspect the evidence immediately. |
| [passage_text, line 20](rag/retrieval.py#L20) and [tokenize, line 24](rag/retrieval.py#L24) | The passage builder is shared. The lexical tokenizer lowercases and extracts Unicode word sequences using `\w+`. The transformer uses its own pretrained tokenizer. |
| [BM25Retriever constructor, line 29](rag/retrieval.py#L29) and [from_database, line 36](rag/retrieval.py#L36) | Index position `i` corresponds to source row `i`. Keeping that mapping is essential because `get_scores` returns an array of numbers without document IDs. |
| [BM25Retriever.search, line 42](rag/retrieval.py#L42) | Checks token overlap using the indexed frequency dictionaries. A matched document can have a score of zero in a small corpus, so testing `score > 0` would lose valid matches. |
| [load_encoder, line 20](rag/dense.py#L20) and [encode_texts, line 28](rag/dense.py#L28) | Load revision `b207367332321f8e44f96e224ef15bc607f4dbf0` of `sentence-transformers/multi-qa-MiniLM-L6-cos-v1`. For N inputs, validate an `N × 384` matrix with finite, nonzero vectors. |
| [embed_missing, line 40](rag/dense.py#L40) | Register the pgvector adapter once on the connection, select one batch in primary-key order with `FOR UPDATE`, pair each output with its input ID, and write using bound parameters. On the CLI connection each completed batch commits independently. |
| [search_dense, line 64](rag/dense.py#L64) | Fail with the missing count if the collection is partly embedded. Encode one query and ask PostgreSQL to rank and limit the results. The same SQL row supplies the ID, score, title, and body. |
| [CLI entry point, line 31](rag/__main__.py#L31) | Dispatch on the requested method. Lexical search runs independently of the transformer. Both search routes serialize the same fields and accept `--limit`. |

## 5. Design decisions

- **One vector per source record.** The model’s output width matches the existing 384-dimensional column. One joined title and abstract maps directly to one NFCorpus ID. The encoder’s observed maximum input length is 512 model tokens including special tokens. Long passages can lose trailing text during dense encoding, while the returned source body remains complete.
- **Transparent lexical matching.** The same lowercase tokenizer handles documents and questions. Hyphens split words, so “X-ray” contributes `x` and `ray`. There is no phrase constraint, stemming, or stop-word removal. A passage sharing just one term is eligible.
- **Exact retrieval and deterministic ties.** A full vector scan is practical for 3,633 passages. PostgreSQL calculates distances and applies the limit. Ties use bytewise passage-ID order (`COLLATE "C"` in SQL). BM25 uses ascending string ID for ties. The default limit is 50 in both APIs.
- **Recoverable indexing.** The default batch size is 16 and can be changed at the CLI. The batch row locks keep ingestion from changing the text while its vector is being written. Completed batches are reusable after an interruption. An unchanged re-ingest preserves embeddings. A changed passage becomes eligible for encoding again. Dense search reports any missing coverage before returning a ranking.

## 6. Verification and observed results

All checks below were executed against Python 3.11.18, PostgreSQL 16.12, and pgvector 0.8.1. Real inference used PyTorch 2.9.1 on CPU. In the command table, `python` denotes `.\.venv\Scripts\python.exe`.

| Command or check executed | Observed result |
|---|---|
| `python -X utf8 -m rag embed --batch-size 16` | Stored **3,633 new passage embeddings**. |
| `docker compose up -d --wait db` followed by the SQL count and dimension query in section 7 | The recreated container became healthy on the same volume. The table contained **3,633 rows, 3,633 vectors, minimum width 384, maximum width 384**. Stored vector norms ranged from 0.9999998874 to 1.0000001355. The only index was the passage primary key. |
| `python -m rag ingest` followed by `python -m rag embed --batch-size 16` | Re-ingestion reported 3,633 rows. The second embedding run reported **0 new embeddings**. |
| `python -m rag verify` | All database IDs matched the corpus. All 11,385 development, 110,575 training, and 12,334 test judgments resolved. |
| `python -m pytest -q tests` | **16 passed, 4 skipped** in 1.01 s. These include the Phase 1 suite, the five-passage lexical fixture, the known-vector SQL fixture, and the batch and re-ingestion regressions. The skipped cases explicitly require the embedded corpus. |
| `python -X utf8 -m pytest -q tests` with `RAG_LIVE_TESTS=1` and `HF_HUB_OFFLINE=1` | **20 passed** in 14.89 s. The four live cases checked 20 displayed results across two methods and two queries. Every ID resolved to the displayed title and complete body. For all 10 dense hits, a separate NumPy cosine calculation agreed with the SQL distance within 0.000002. |
| `python -X utf8 -m rag search bm25 zxqneverincorpus --limit 5` | Returned `[]`. |
| `encode_texts([PARAPHRASE])` with the real cached model | Shape `(1, 384)` and vector norm `1.0000001`. The loaded model reported mean-token pooling, normalization, and a 512-token maximum. |

**A controlled test of vector direction.** Against a query pointing east, three database vectors point east, north, and west. SQL returned their IDs in that order, with distances **0, 1, 2**. Making the north vector identical to the east vector then exercised the secondary ID sort. This isolates the SQL ordering from a model’s judgment about language. [Known-vector test](tests/test_retrieval.py#L48)

### What the real searches returned

The named-carrot phrase comes from the body of `MED-3374`, a grade-2 passage for development query `PLAIN-111`. We predicted that its distinctive wording would favor BM25 and that a question about school meals would yield topically related passages in dense search. We ran `python -X utf8 -m rag search METHOD "QUERY" --limit 5` with both methods and both strings below. Scores are rounded to six decimal places for display.

| Rank | BM25 for “X-ray Vision Carrots” | Dense for “X-ray Vision Carrots” |
|---:|---|---|
| 1 | MED-3374 · 26.159137 | MED-2894 · 0.615662 |
| 2 | MED-3282 · 12.192353 | MED-3621 · 0.629933 |
| 3 | MED-3962 · 11.287004 | MED-4385 · 0.631375 |
| 4 | MED-1023 · 10.400826 | MED-4884 · 0.634911 |
| 5 | MED-3397 · 9.805045 | MED-2069 · 0.641564 |

BM25 put the actual school study first. Dense search’s first two titles concern visual fields in glaucoma and dental X-rays. The playful name gives a short query several plausible interpretations, and the source study fell outside the dense top five.

| Rank | BM25 for “How can school cafeterias encourage pupils to choose more vegetables?” | Dense for the same paraphrase |
|---:|---|---|
| 1 | MED-3374 · 24.391019 | MED-3374 · 0.384141 |
| 2 | MED-3370 · 21.972104 | MED-3369 · 0.385572 |
| 3 | MED-5029 · 19.118799 | MED-4172 · 0.397736 |
| 4 | MED-2201 · 18.100944 | MED-3370 · 0.409347 |
| 5 | MED-1162 · 16.828695 | MED-3373 · 0.428842 |

Both methods placed the school study first. The next three dense passages discuss incorporating vegetables into children’s meals, preschool rewards, and serving styles. The fifth concerns adult community gardens. BM25’s third passage concerns a school trial about head lice. Reading these texts shows how matching “school” can recover the setting while missing the question’s subject.

**Limitations.** These two inspection queries demonstrate retrieval behavior on one topic. Overall Recall@K and MRR have yet to be measured on a held-out split. BM25 builds an in-memory index per CLI process, and dense search scans every stored vector per query. Encoding long abstracts can discard text past the token limit. Raw floating-point scores can vary slightly with hardware and library versions.

## 7. How to run

Run the commands in PowerShell from the project root. The current workspace already has an environment, populated database, private password file, and model cache. On a **fresh checkout and new database volume**, initialize them first:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path -LiteralPath '.env.db-password')) {
    [IO.File]::WriteAllText((Join-Path $PWD '.env.db-password'), [guid]::NewGuid().ToString('N'))
}
docker compose up -d --wait db
.\.venv\Scripts\python.exe -m rag download
.\.venv\Scripts\python.exe -m rag ingest
```

Build the vectors and inspect the two searches. The first embedding run downloads the model. Its default cache is `data/hf/hub`, and setting `HF_HOME` selects another cache root. On the verification machine `OMP_NUM_THREADS=4` was used for CPU inference.

```powershell
$env:OMP_NUM_THREADS = '4'
.\.venv\Scripts\python.exe -X utf8 -m rag embed --batch-size 16
.\.venv\Scripts\python.exe -X utf8 -m rag search bm25 "X-ray Vision Carrots" --limit 5
.\.venv\Scripts\python.exe -X utf8 -m rag search dense "How can school cafeterias encourage pupils to choose more vegetables?" --limit 5
```

Run the full suite, including the real-model checks. Setting `HF_HUB_OFFLINE` here uses the weights already cached by `embed`. The ordinary suite can run with only the project database and installed packages. Its four full-corpus cases are opt-in.

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:RAG_LIVE_TESTS = '1'
.\.venv\Scripts\python.exe -X utf8 -m pytest -q tests
.\.venv\Scripts\python.exe -m rag verify
.\.venv\Scripts\python.exe -m rag embed --batch-size 16
docker compose exec -T db psql -U rag -d rag -c "SELECT count(*), count(embedding), min(vector_dims(embedding)), max(vector_dims(embedding)) FROM passages;"
```

To inspect every top-five list, switch `bm25` and `dense` in the two example commands. To examine the human-readable benchmark evidence used to choose the example, run `.\.venv\Scripts\python.exe -X utf8 -m rag inspect PLAIN-111 --limit 2`.

## 8. Phase completion checklist

| Phase 2 requirement | Implementation | Verification |
|---|---|---|
| 1. BM25 lexical retrieval | [BM25Okapi index and search](rag/retrieval.py#L28) | Distinctive term ranks first in a five-passage fixture. Unknown terms yield no hits. Exact-term corpus results are recorded above. |
| 2. Dense transformer retrieval | [Model loader](rag/dense.py#L20) | Loaded the pinned MiniLM checkpoint and ran both real queries with its cached weights. |
| 3. Batched 384-dimensional storage | [Embedding loop](rag/dense.py#L40) | SQL confirmed all 3,633 widths. The retry test recovered after a failed batch. Repeating the full build wrote zero additional vectors. |
| 4. pgvector cosine retrieval | [Dense search](rag/dense.py#L64) | Known-vector distances were 0, 1, and 2. Independently recalculated all ten live dense distances. |
| 5. Consistent ranked outputs | [SearchHit](rag/retrieval.py#L11) | All four real searches produced ranks 1–5 with unique IDs and text matching PostgreSQL. |
| 6. Independent CLI commands | [CLI](rag/__main__.py#L31) | Ran `embed`, `search bm25`, and `search dense` through their normal entry points. |
| 7. Unit and real-query verification | [Fixture tests](tests/test_retrieval.py) and [full-corpus tests](tests/test_live_retrieval.py#L23) | The live-enabled suite passed all 20 cases. Both top-five comparisons are recorded in section 6. |
| 8. Preserve Phase 1 behavior | [Storage tests](tests/test_storage.py) and [metric tests](tests/test_metrics.py) | The original ten tests passed as part of the suite. Real `download`, `ingest`, `inspect`, and `verify` commands also ran successfully. |

The root-level `phase_2.zip` is the review copy. It includes the complete source, tests, Compose file, requirements, ignore rules, plan, and both phase reports. Reproduce the data and model from their pinned sources using section 7.
