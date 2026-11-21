# Phase 1 report — Establish the data and the scoring rules

Phase 1 of the [implementation plan](ADVANCED_RAG_IMPLEMENTATION_PLAN.md) is complete. PostgreSQL contains all **3,633 NFCorpus passages**. Two consecutive ingestions produced the same row count. All **134,294 published query–passage judgments** resolved to existing query and passage IDs. The metric and database test suite passed **10 tests**.

## 1. Concepts and technologies

A **corpus** is the collection of documents available to search. Here, one title and its abstract form one identifiable passage. A **query** is the user-facing text, such as “How to Get Kids to Eat Their Vegetables”. A **qrel** is the benchmark’s judgment about one query–passage pair. A **ranking** is an ordered list of passage IDs proposed as results for a query. A query, a judgment, and a ranking are three separate inputs. You can test the scoring logic by supplying the last one by hand.

Python 3.11 reads the benchmark using the standard library. Each line of the gzipped JSON Lines files contains a complete source record. The tab-separated judgment files contain a query ID, a passage ID, and an integer grade. Psycopg 3 sends passage data to PostgreSQL 16. The pgvector extension gives the database its `vector(384)` type. Pytest exercises the arithmetic and the actual database behavior. [Corpus source revision](https://huggingface.co/datasets/BeIR/nfcorpus/tree/e28763e68d85db0fa71652ba3c4afabf8c5b3bb7) · [Judgment source revision](https://huggingface.co/datasets/BeIR/nfcorpus-qrels/tree/a451b3b26d3ae1358f259c1a3a4dd61fcea35a65)

## 2. Architecture and complete data flow

| Operation | Input → processing → output |
|---|---|
| `download` | Two pinned BEIR dataset revisions → verify each file against its SHA-256 checksum → save five inputs under `data/nfcorpus/`. A repeat run reuses local files with matching checksums. |
| `ingest` | `corpus.jsonl.gz` → read each record keyed by its original `_id` → map `_id`, `title`, and `text` to `passage_id`, `title`, and `body` → insert or update the `passages` table in one transaction. A Docker volume holds the database between runs. |
| `inspect` | Development query ID → look up its text in `queries.jsonl.gz` and its judgments in `dev.tsv` → select positive grades → fetch matching titles and bodies from PostgreSQL by ID → display the question, positive judgment count, grades, and source text. |
| `verify` | Source passage IDs and stored passage IDs → check set equality → read each split’s qrels → check every referenced query against the query file and every referenced passage against the database → print per-split counts. |
| Metric evaluation | In-memory judgments and a hand-written ranked list for each query → calculate Recall@K and reciprocal rank per query → average over the judged queries → return mean recall, MRR, and query count. This path runs independently of PostgreSQL. |

### Following one development query

`PLAIN-111` has **47 positive judgments**. With `--limit 2`, inspection displays two passages with grade 2, ordered by passage ID within that grade. The first, `MED-3374`, describes school lunchroom studies that gave vegetables attractive names. The second, `MED-3376`, describes a study that varied cookie portion sizes. Both carry the same published grade, but the first passage addresses the vegetable question more directly. Reading the abstracts shows why a relevance label is useful for scoring and why it takes a separate check to decide whether a passage supports an answer. These two displayed positions come from the inspection command’s ordering rule.

## 3. Files created and their purposes

| File | Purpose |
|---|---|
| [compose.yaml](compose.yaml) | Defines the PostgreSQL 16 and pgvector service, pinned image digest, health check, loopback port 5433, and persistent volume. |
| [requirements.txt](requirements.txt) | Pins the Psycopg binary distribution and pytest versions used here. |
| [.gitignore](.gitignore) | Excludes downloaded inputs, local Python environments, bytecode, and pytest caches. |
| [rag/__init__.py](rag/__init__.py) | Declares the Python package. |
| [rag/__main__.py](rag/__main__.py) | Implements `python -m rag`, parses arguments, opens the database connection when needed, and dispatches commands. |
| [rag/download.py](rag/download.py) | Records the two source revisions and five file hashes. Downloads and verifies the benchmark inputs. |
| [rag/dataset.py](rag/dataset.py) | Parses the compressed JSON records and TSV judgments. Defines the three official split names. |
| [rag/schema.sql](rag/schema.sql) | Enables pgvector and defines the four columns of `passages`. |
| [rag/storage.py](rag/storage.py) | Implements passage loading, cross-split ID verification, and development-query inspection. |
| [rag/metrics.py](rag/metrics.py) | Implements per-query scores and their averages. |
| [tests/test_metrics.py](tests/test_metrics.py) | Tests the hand-calculated example, cutoff behavior, missing results, relevance grades, and invalid inputs. |
| [tests/test_storage.py](tests/test_storage.py) | Tests the schema, repeated loads, reading evidence, unresolved references, and rollback in PostgreSQL. |
| [PHASE_1_REPORT.md](PHASE_1_REPORT.md) | Records the implementation, evidence, and reproduction instructions for this phase. |

The runtime data files are `corpus.jsonl.gz`, `queries.jsonl.gz`, `dev.tsv`, `train.tsv`, and `test.tsv`. The download command recreates them. In this workspace, `.venv` is ready to use, and `.bootstrap` holds the local Python 3.11 installation used to create it.

## 4. Code walkthrough and algorithms

1. **Read the source formats.** `read_records` builds a dictionary keyed by the original string ID and rejects duplicate IDs. `read_qrels` builds a nested dictionary, so `qrels[query_id][passage_id]` gives the original grade. It checks the header and rejects duplicate query–passage judgments and empty judgment files. [Record reader](rag/dataset.py#L12) · [Judgment reader](rag/dataset.py#L25)
2. **Write the corpus.** `ingest` reads all records, runs the schema SQL, and passes parameterized rows to Psycopg’s `executemany`. On a primary-key conflict, PostgreSQL updates a row only when its title or body changed. A failed write rolls back the batch. [Ingestion function](rag/storage.py#L10) · [Table declaration](rag/schema.sql#L3)
3. **Follow a judgment to the source.** `inspect_dev` filters grades greater than zero, sorts by descending grade and then ascending passage ID, and looks up each selected ID in PostgreSQL. `verify` checks all grades, including zero if present, because every judgment needs a valid reference. [Development inspector](rag/storage.py#L54) · [Integrity check](rag/storage.py#L29)
4. **Calculate a query’s scores.** `score_query` collects IDs with positive grades, takes the first K positions of the ranking, and numbers them starting at one. Recall is the number of hits divided by the total number of relevant IDs. Reciprocal rank is one divided by the position of the first hit. A list with no hit scores zero on both. [Per-query implementation](rag/metrics.py#L8)
5. **Average over the query set.** `evaluate_at_k` iterates over the queries in the supplied judgments. A query missing from the rankings receives an empty list, so it stays in the denominator. The immutable `Metrics` result holds `recall`, `mrr`, and `query_count`. [Averaging function](rag/metrics.py#L28) · [Result class](rag/metrics.py#L22)

### Work the metric example by hand

| Query | Positive passage IDs | Ranking | Hits in first 4 | Recall@4 | Reciprocal rank@4 |
|---|---|---|---|---|---|
| Q1 | B, D, E | A, B, C, D | B at 2 and D at 4 | 2/3 | 1/2 |
| Q2 | F | Empty | None | 0 | 0 |
| **Mean** | | | | **1/3** | **1/4** |

The denominator for Q1’s recall is three because there are three known relevant passages. The denominator for mean reciprocal rank is two because there are two evaluated queries. Prediction exercise: move `B` from second place to first while keeping `D` fourth. Q1’s recall stays at 2/3, its reciprocal rank becomes 1, and the two-query MRR becomes 1/2. [Executable worked example](tests/test_metrics.py#L6)

## 5. Design decisions

- **Preserve the original IDs.** One benchmark record corresponds to one database row and one evaluation ID. The primary key prevents a second ingestion from adding another copy of a passage. Title and body remain separate so you can inspect either source field.
- **Keep labels with the evaluator.** PostgreSQL stores the corpus. The inspection and verification tools read qrels from disk. This keeps the information used to judge a ranking independent of the text used to build it.
- **Allow `NULL` embeddings.** A missing vector accurately represents a passage that has only been ingested. If the text of an existing row changes, the update clears its embedding because a vector for the old text would be stale.
- **Make metric conventions explicit.** A positive grade is relevant. Unjudged IDs receive no credit. Each evaluated query has equal weight. A query with no positive judgments receives zero by convention. Duplicate IDs in one ranking, a nonpositive K, and an empty evaluation query set are rejected. [Metric implementation](rag/metrics.py)
- **Pin source inputs and the database image.** The original archive host timed out during setup. The downloader uses fixed revisions of the official BEIR mirror and verifies the five file hashes. The corpus is small enough to read into memory, which keeps the reader and the ID audit straightforward. [Download manifest](rag/download.py) · [Database image](compose.yaml)

## 6. Tests, results, and limitations

The verified environment used **Python 3.11.18, PostgreSQL 16.12, pgvector 0.8.1, Psycopg 3.2.13, and pytest 8.4.2**.

| Check performed | Observed result |
|---|---|
| Run the automated suite | **10 passed**. Seven metric cases and three live database tests. Each database test used a unique schema and rolled back its transaction. |
| Run ingestion twice against the full corpus | **3,633 rows after each run**. The database contained zero populated embeddings. |
| Audit the development split | **324 queries, 11,385 judgments**. All IDs resolved. |
| Audit the training split | **2,590 queries, 110,575 judgments**. All IDs resolved. |
| Audit the test split | **323 queries, 12,334 judgments**. All IDs resolved. The test file was read for reference integrity only. |
| Run the plan’s two-query example | Q1 returned recall **2/3** and reciprocal rank **1/2**. Across both queries, mean recall was **1/3** and MRR was **0.25**. |
| Exercise failure paths | A dangling query and passage ID produced a diagnostic. A batch with a valid update followed by an invalid row failed and restored the prior title and row count. |

**Interpretation.** The numerical metric results above describe the hand-ranked exercise. Retrieval quality on NFCorpus remains unmeasured. The ID audit establishes that records can be joined correctly. The `PLAIN-111` example illustrates that a positive qrel alone cannot establish factual support for an answer.

## 7. Run and verify

Run these commands in PowerShell from the project root. This workspace already has the virtual environment, dataset, and database. To inspect the completed phase immediately:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m rag inspect PLAIN-111 --limit 2
.\.venv\Scripts\python.exe -m rag verify
.\.venv\Scripts\python.exe -m pytest -q tests
```

On a **fresh checkout**, install Python 3.11 and start Docker Desktop with Linux containers, then run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
docker compose up -d --wait db
.\.venv\Scripts\python.exe -m rag download
.\.venv\Scripts\python.exe -m rag ingest
.\.venv\Scripts\python.exe -m rag ingest
.\.venv\Scripts\python.exe -m rag verify
.\.venv\Scripts\python.exe -X utf8 -m rag inspect PLAIN-111 --limit 2
.\.venv\Scripts\python.exe -m pytest -q tests
```

Expect `3633 passages in the database` on **both** ingestions. Compare the verification output to the three split counts above. To see all positive judgments for the worked query, use `--limit 47`. To run the arithmetic tests without a database, use `.\.venv\Scripts\python.exe -m pytest -q tests/test_metrics.py`.

The CLI defaults to the project’s Compose database at `127.0.0.1:5433`. An explicit `RAG_DATABASE_URL` environment variable selects a different PostgreSQL database with pgvector available. To stop the local project database while keeping the data, run `docker compose stop db`. The same `docker compose up -d --wait db` command starts it again.
