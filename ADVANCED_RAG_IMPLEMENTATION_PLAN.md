# Advanced RAG with Hybrid Retrieval & Reranking

Build a **command-line document QA system backed by one PostgreSQL database**. Give it two entry points. `search` shows the passages each retrieval method found. `ask` takes the best reranked passages and produces an answer with citations. Building `search` first lets you answer an important debugging question in RAG: *Did the system find the evidence before it asked the LLM to answer?*

For the benchmark, use **BEIR’s NFCorpus**. It has about 3,600 documents, natural-language queries, and published judgments linking queries to relevant documents. That gives you a manageable corpus and an independent way to measure retrieval quality. Use each record’s title and abstract as one passage, preserving its original document ID. The unit you retrieve, cite, and evaluate will then be the same. [BEIR corpus](https://huggingface.co/datasets/BeIR/nfcorpus) · [Relevance judgments](https://huggingface.co/datasets/BeIR/nfcorpus-qrels)

## Technology stack

| Component | Choice | Why it earns its place |
|---|---|---|
| Application | Python 3.11, a small CLI, `pytest` | You can run and test each stage independently. |
| Storage | PostgreSQL 16 with pgvector, started through Docker Compose. Connect with Psycopg 3 and `pgvector` for Python. | One database holds passage IDs, source text, and vectors. Exact cosine search is a sensible starting point for this corpus size. [pgvector documentation](https://github.com/pgvector/pgvector) |
| Dense retrieval | Sentence Transformers with `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` | A pretrained question–passage retrieval model that produces 384-dimensional vectors. [Model card](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1) |
| Lexical retrieval | `rank_bm25` with `BM25Okapi` | Build the BM25 index in Python from the same PostgreSQL rows. This makes the lexical ranking easy to inspect and avoids running another search service. [rank_bm25 documentation](https://github.com/dorianbrown/rank_bm25) |
| Fusion and reranking | A small Reciprocal Rank Fusion (RRF) function you write, followed by `cross-encoder/ms-marco-MiniLM-L6-v2` | RRF is short enough to verify by hand. The pretrained cross-encoder scores each question and candidate passage together. [RRF paper](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) · [Sentence Transformers guide](https://www.sbert.net/examples/sparse_encoder/applications/retrieve_rerank/README.html) |
| Answer generation | Ollama running `qwen2.5:3b` locally | A concrete, API-key-free choice for producing cited answers. Only `ask` needs the LLM running. [Ollama model page](https://ollama.com/library/qwen2.5:3b) |

## Architecture

| When | What happens |
|---|---|
| **During ingestion** | Load the corpus → preserve each source ID → combine title and abstract → embed each passage once → store text and vector in PostgreSQL. |
| **At application startup** | Read passage IDs and text from PostgreSQL in a stable order → tokenize the text → build the BM25 index and its ID mapping. Reuse it across queries. |
| **For each search** | The same question goes to both retrievers. Dense search returns its top 50 by cosine distance. BM25 returns up to 50 matching passages. RRF merges their ranked IDs. The cross-encoder reranks the best 50 fused candidates. |
| **For each answer** | Give the LLM the question and the top 3 reranked passages, labeled with source IDs. Return the answer alongside the cited passages so a reader can inspect the evidence. |
| **During evaluation** | Run the same held-out queries through dense, BM25, hybrid, and hybrid-plus-reranking. Compare their ranked IDs with the published relevance judgments. The evaluator never needs to call the LLM. |

The numbers **50 candidates and 3 context passages** are starting decisions for this project. Keeping them fixed makes the first experiment interpretable. Fifty gives the reranker room to move a relevant passage upward. Three makes it practical to read everything the LLM saw.

## Development phases and learning roadmap

### Phase 1 — Establish the data and the scoring rules

**Learn first.** A *corpus* is the collection you search. A *query* is a search request. A *qrel* is a benchmark judgment connecting a query ID to a relevant document ID. Reading one example of each will make the rest of the project easier to understand.

**Build in small steps.**

1. Inspect one development query, its positive qrels, and the corresponding source passages. Read them yourself.
2. Load the corpus into a `passages` table with `passage_id` as its primary key, plus `title`, `body`, and an `embedding` column of type `vector(384)`. Keep qrels in the evaluator.
3. Write the two metric calculations against a tiny, hand-ranked example before connecting a retriever. Define a positive qrel as relevant and average each metric across queries. [Retrieval metric definitions](https://ir-measur.es/en/latest/measures.html)

**Test before moving on.** Suppose a query has three relevant passages, `B`, `D`, and `E`, and your system returns `A, B, C, D`. At cutoff 4, **Recall@4 is 2/3** because you found two of the three. **Reciprocal rank is 1/2** because the first relevant result is second. If a second query with relevant judgments retrieves nothing, it scores zero, so **MRR@4 across the two queries is 0.25**. Also verify that every judged passage ID resolves to a database row and that ingesting the corpus twice preserves the row count.

### Phase 2 — Build two retrievers you can inspect separately

**Learn first.** BM25 rewards matching query terms, especially terms that are rare in the corpus. It accounts for passage length and gives diminishing returns for repeating a word. Dense retrieval encodes each text as a vector and compares vector directions. This gives a paraphrase a chance to match even when its words differ.

**Build in small steps.**

1. Implement BM25 first. Use the same lowercase word tokenizer for documents and queries. Keep an explicit mapping from every BM25 position to its passage ID.
2. Embed passages in batches and store the vectors. At query time, embed only the question and have pgvector order passages by ascending cosine distance. Use passage ID to break ties consistently. [pgvector distance operators](https://github.com/pgvector/pgvector)
3. Give both retrievers the same output shape: passage ID, rank, and native score or distance. Run an exact-term query and a paraphrased query. Predict which retriever will do better, then read the top five results from each.

**Test before moving on.** In a five-passage fixture, put a distinctive term in one passage and check that BM25 ranks it first. A query with no matching terms should produce an empty lexical list. For dense search, store three vectors with known directions and check the SQL ordering by hand. Then run one real query to verify the model produces 384 values and the returned IDs resolve to the displayed text.

### Phase 3 — Fuse dense and lexical rankings with RRF

**Learn first.** A BM25 score and a cosine distance use different scales. RRF uses rank positions, so you can combine the lists without calibrating their scores. A passage gets credit from each list that contains it. [Original RRF paper](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)

**Build in small steps.**

1. Number each input list starting at rank 1.
2. For each passage ID, add `1 / (60 + rank)` for each list in which it appears. A missing ID contributes zero from that list.
3. Sort by total score descending, break ties by passage ID, and retain the best 50. In `search`, display the dense rank, BM25 rank, and fused rank side by side.

**Test before moving on.** With dense results `A, B` and BM25 results `B, C`, predict the winner. `B` receives `1/62 + 1/61`, so it should finish first and appear exactly once. Also test an empty BM25 list, which should preserve the dense order. On the development split, compare hybrid and dense Recall@10. Inspect a query where fusion helped and, if one exists, a query where it pushed evidence down.

### Phase 4 — Rerank the candidates with a cross-encoder

**Learn first.** The dense retriever computes a passage vector once and reuses it. A cross-encoder reads the question and passage **together**, producing a relevance score specific to that pair. This costs a model evaluation per candidate, which is why you use it after retrieval. [Cross-encoder documentation](https://www.sbert.net/docs/cross_encoder/pretrained_models.html)

**Build in small steps.**

1. Take the top 50 fused IDs and fetch their passage text.
2. Form one `(question, passage)` pair per ID, score the pairs in batches, and sort by descending score. Keep each score attached to its original ID.
3. Inspect one development query before and after reranking. Read the passages that moved most. Record the model’s input length limit and check whether long abstracts get truncated.

**Test before moving on.** First use three fake scores with a known order to catch ID/score mix-ups. Then run the real model and verify that the input and output ID sets are identical. **Hybrid Recall@50 and reranked Recall@50 must match** when all 50 candidates are retained. Recall@10 and MRR@10 can change because the order changed. This is a particularly good test to explain in an interview.

### Phase 5 — Generate answers grounded in retrieved evidence

**Learn first.** A passage can be relevant to a topic and still omit the fact needed to answer a question. Grounding requires checking whether the cited text actually supports what the model says.

**Build in small steps.**

1. Format the top three passages with their IDs, titles, and text. Save the exact context sent to the model for each demonstration.
2. Ask for a short answer with a passage ID after each factual claim. Instruct the model to say “The retrieved passages do not provide enough evidence” when appropriate.
3. Validate that every cited ID belongs to the supplied context. Display the cited text beside the answer.

**Test before moving on.** Choose a passage that explicitly reports a fact, such as a study’s participant count. Ask for that fact and verify the value and citation against the text. Then remove the only supporting passage and ask again. The expected answer is an abstention. Finally, test an invented citation ID and verify that your application flags it. The ID check can be automated. Checking whether the cited sentence supports the answer requires reading it.

### Phase 6 — Run the held-out comparison and make it reproducible

**Learn first.** Recall@K measures how much of the **known relevant evidence** appears in the first K results. MRR@K measures how early the **first relevant result** appears, assigning zero when there is no hit within K. A system can improve one measure while losing on the other.

**Build in small steps.**

1. Use development queries for debugging and parameter choices. Before the final run, freeze the corpus checksum, model revisions, package versions, database image, tokenizer, input limits, candidate counts, and tie-breaking rule.
2. Evaluate every pipeline on the same official test queries and corpus. Save one ranked ID list per query and pipeline so you can reproduce the metrics without rerunning inference.
3. Publish one table with rows for **dense**, **BM25**, **dense + BM25 via RRF**, and **RRF + cross-encoder**. Use columns for **Recall@10**, **MRR@10**, and **Recall@50**. Record the number of evaluated queries. BM25 is already implemented, and its row tells you whether fusion beats your strongest individual retriever.
4. In the README, give the steps to start the database, ingest the corpus, run the tests, evaluate retrieval, and ask one cited question. Include a worked query showing the evidence passage’s rank at each stage.

**Test before calling it complete.** Rebuild from an empty database with the recorded configuration. Check the corpus count, query count, and results table against the original run. Recompute the table from saved rankings, retaining zero-hit queries in the averages. Finally, run the documented `ask` example and check its evidence directly.

## What the finished project should let you defend

| Résumé claim | Evidence you should be able to show |
|---|---|
| Document QA with PostgreSQL, pgvector, BM25, and transformer embeddings | Run one question, show its database passage ID, and trace that ID through both retrievers to the cited answer. |
| Hybrid retrieval for stronger recall | Show the held-out **hybrid minus dense Recall@10** difference at the same cutoff. Use “improved recall” only if your measured result supports it. |
| Cross-encoder reranking before grounded generation | Show one candidate list before and after reranking, the three passages actually sent to the LLM, and the sentence supporting its answer. |
| Evaluation with Recall@K and MRR | Recreate the comparison table from saved rankings and explain one query’s scores by hand. |

**Your first learning milestone is small:** take one development query, read two judged-relevant passages, invent a five-result ranking, and calculate its recall and reciprocal rank on paper. Once you can explain those two numbers, every later phase has a clear purpose and a result you know how to check.
