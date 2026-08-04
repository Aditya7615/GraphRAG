# Lengraph — Enterprise RAG with Hybrid Search & Zero-Hallucination Guardrails

A production-shaped Retrieval-Augmented Generation system built to scale to 100,000+ PDFs. Ships with a synthetic enterprise corpus generator, so you can run the entire pipeline end-to-end before you have any real documents.

```
                        ┌────────────────────────────────────────────┐
  PDF / DOCX / MD  ──▶  │  Loader → Parent chunks (~2000 tok)        │──▶  Docstore
                        │           └─ Child chunks (~400 tok)       │      (Postgres/JSON)
                        └────────────────────────────────────────────┘           │
                                        │ embed (dense) + BM25 (sparse)          │
                                        ▼                                        │
                             ┌──────────────────────┐                            │
   query  ──────────────────▶│  Qdrant hybrid RRF   │──▶ child hits ──▶ expand ──┘
                             └──────────────────────┘                     │
                                                                          ▼
                                                        parent context ──▶ Google Gemini (T=0.0)
                                                                          │
                                                          guardrail: citations validated
                                                                          ▼
                                                              grounded answer + sources
```

---

## Features

- **Parent-child chunking** — small chunks for precise embedding, large chunks for LLM context
- **Hybrid dense + BM25 search** — Reciprocal Rank Fusion inside Qdrant for one round-trip
- **Zero-hallucination guardrails** — post-generation citation validation strips ungrounded claims
- **Grounded answers** — every factual sentence must cite a retrieved source `[S1]`, `[S2]`, etc.
- **10 document types** — financial, HR, engineering, compliance, legal, marketing, IT, sales, customer success, product
- **Scalable to 100k+ PDFs** — int8 quantization, on-disk payloads, HNSW tuning
- **Full API** — FastAPI with search, chat, ingestion, and admin endpoints
- **Streamlit UI** — dark-mode analyst workspace with chat, ingestion console, and retrieval inspector

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | |
| Docker | 20.10+ | For Qdrant and Postgres |
| Google API Key | — | Free tier available at [aistudio.google.com](https://aistudio.google.com/apikey) |

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/aditya7615/Lengraph.git
cd Lengraph
```

### 2. Set up Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set your Google API key:

```bash
GOOGLE_API_KEY=AIzaSy...   # https://aistudio.google.com/apikey
```

Everything else has working defaults. Embeddings run locally via `sentence-transformers` (no second API key needed). The model (~130 MB) downloads on first use.

### 4. Start infrastructure

```bash
docker compose up -d
```

This starts:
- **Qdrant** — vector database at `http://localhost:6333` (dashboard: `http://localhost:6333/dashboard`)
- **Postgres** — parent document store at `localhost:5432` (`rag` / `rag` / `ragdb`)

Verify:

```bash
curl -s localhost:6333/healthz
docker compose ps
```

> **No Docker?** Qdrant is required. Postgres is optional — `DOCSTORE_BACKEND=auto` automatically falls back to a local JSON store at `data/docstore/parents.json`.

### 5. Generate synthetic corpus

```bash
# 18 hand-crafted enterprise PDFs (quick, ~5s)
python scripts/generate_synthetic_data.py

# OR 10,000 Faker-generated PDFs (~2 min with 8 workers)
python scripts/generate_large_corpus.py --count 10000 --out data/synthetic
```

### 6. Ingest documents

```bash
python scripts/ingest.py --dir data/synthetic
```

Re-running is idempotent — chunk IDs are deterministic UUIDv5s, so documents upsert in place.

### 7. Run the API

```bash
uvicorn backend.main:app --reload --port 8000
```

Interactive docs: http://localhost:8000/docs

### 8. Run the UI

In a second terminal:

```bash
streamlit run frontend/app.py
```

Opens at http://localhost:8501.

---

## Try It

**Grounded answer with citations:**

```bash
curl -s localhost:8000/chat -H 'Content-Type: application/json' -d '{
  "question": "How many weeks of paid leave do primary caregivers receive?"
}' | python -m json.tool
```

**Exact-identifier lookup** (BM25 carries the hybrid search):

```bash
curl -s localhost:8000/chat -H 'Content-Type: application/json' -d '{
  "question": "What was the root cause of incident INC-2291?"
}' | python -m json.tool
```

**Guardrail firing** — nothing in the corpus answers this:

```bash
curl -s localhost:8000/chat -H 'Content-Type: application/json' -d '{
  "question": "What is the CEO'\''s home address?"
}'
# → "answer": "I cannot find the answer in the provided documentation."
# → "grounded": false, "citations": []
```

**Conflicting sources** — the access-review window differs between documents:

```bash
curl -s localhost:8000/chat -H 'Content-Type: application/json' -d '{
  "question": "How long do asset owners have to complete an access review?"
}' | python -m json.tool
```

---

## API Reference

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Qdrant / docstore / LLM readiness |
| `GET` | `/stats` | Vector counts, indexed documents, model config |
| `POST` | `/ingest/file` | Multipart upload of a single document |
| `POST` | `/ingest/directory` | Bulk-ingest a server-side folder |
| `POST` | `/search` | Raw child-chunk hits, no LLM (retrieval debugging) |
| `POST` | `/chat` | Grounded answer + validated citations |
| `DELETE` | `/documents/{doc_id}` | Remove a document from index and docstore |
| `POST` | `/admin/reset?confirm=true` | Wipe the collection and docstore |

### `/chat` request body

```json
{
  "question": "What is the RTO for tier-1 services?",
  "top_k": 12,
  "max_parents": 4,
  "doc_ids": null,
  "search_mode": "hybrid",
  "include_parent_text": false
}
```

### `/chat` response

```json
{
  "answer": "The recovery time objective for tier-1 services is 15 minutes [S1].",
  "grounded": true,
  "citations": [{
    "marker": "S1",
    "source": "AWS_Cloud_Architecture_Guide.pdf",
    "section": "4. Networking and Edge",
    "page": 2,
    "score": 0.0328,
    "used_by_llm": true,
    "snippet": "The recovery time objective (RTO) for tier-1 services is..."
  }],
  "latency_ms": {"retrieval": 41.2, "generation": 612.8, "total": 654.0},
  "model": "gemini-2.0-flash",
  "search_mode": "hybrid"
}
```

---

## Project Structure

```
Lengraph/
├── backend/
│   ├── config.py                  # Central configuration (env-overridable)
│   ├── main.py                    # FastAPI app: ingest, search, chat, stats, admin
│   ├── ingestion/
│   │   ├── chunker.py             # Parent-child splitting + section/page mapping
│   │   ├── loader.py              # PDF / DOCX / TXT / MD → SourceDocument
│   │   ├── embeddings.py          # Local bge-small dense embeddings
│   │   ├── sparse.py              # BM25 sparse encoder (IDF applied by Qdrant)
│   │   ├── vector_store.py        # Qdrant collection setup + hybrid RRF search
│   │   ├── docstore.py            # Parent store: Postgres, with JSON fallback
│   │   └── pipeline.py            # Ingestion orchestration
│   ├── generation/
│   │   ├── prompts.py             # Grounding rules + verbatim fallback string
│   │   ├── retriever.py           # Child hits → deduplicated parent context
│   │   └── chain.py               # Gemini chain + citation validation guardrail
│   └── models/
│       └── schemas.py             # Pydantic API contracts
├── frontend/
│   └── app.py                     # Streamlit: chat, citations, ingest, retrieval debug
├── scripts/
│   ├── generate_synthetic_data.py # 18 hand-crafted enterprise PDFs
│   ├── generate_large_corpus.py   # Scalable Faker PDF generator (1M+ PDFs)
│   ├── corpus_extra.py            # Additional document definitions
│   └── ingest.py                  # CLI ingestion tool
├── tests/
│   └── test_pipeline.py           # Unit tests (no external services needed)
├── docker-compose.yml             # Qdrant + Postgres
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variable template
└── ruff.toml                      # Linter configuration
```

---

## How the Guardrails Work

Three layers enforce zero hallucination:

1. **Prompt** (`generation/prompts.py`) — closed-book rule, verbatim fallback sentence, mandatory `[S#]` markers on every factual sentence, explicit instructions for conflicting and partially-covered questions.

2. **Empty retrieval short-circuit** (`generation/chain.py`) — if hybrid search returns nothing, the fallback is returned without an LLM call.

3. **Post-generation validation** (`chain._enforce_guardrails`):
   - Markers not in the retrieved set are stripped as hallucinated
   - An answer left with **zero** valid citations is replaced by the fallback
   - Soft refusals ("the context does not contain...") are normalised to the exact fallback string

`grounded: false` is the machine-readable signal that the guardrail fired.

---

## Configuration

All settings are env-overridable (see `backend/config.py` or `.env.example`).

| Setting | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | — | Google Gemini API key (required for `/chat`) |
| `LLM_MODEL` | `gemini-2.0-flash` | Gemini model to use |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local sentence-transformers model |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant connection URL |
| `DOCSTORE_BACKEND` | `auto` | `postgres`, `json`, or `auto` (try Postgres, fallback to JSON) |
| `PARENT_CHUNK_TOKENS` | `2000` | Target tokens per parent chunk |
| `CHILD_CHUNK_TOKENS` | `400` | Target tokens per child chunk |
| `RETRIEVAL_TOP_K` | `12` | Child chunks retrieved per query |
| `MAX_PARENTS_IN_CONTEXT` | `4` | Parent chunks sent to the LLM |
| `MAX_CONTEXT_TOKENS` | `6000` | Token budget for LLM context |

### Tuning Guide

| Symptom | Knob |
|---|---|
| Answers miss context that exists | Raise `RETRIEVAL_TOP_K`, `MAX_PARENTS_IN_CONTEXT` |
| Answers are diluted / slow | Lower `MAX_PARENTS_IN_CONTEXT`, `MAX_CONTEXT_TOKENS` |
| Exact IDs not being found | Raise `SPARSE_PREFETCH_LIMIT` |
| Paraphrased questions fail | Raise `DENSE_PREFETCH_LIMIT`; try `BAAI/bge-base-en-v1.5` |
| Chunks split mid-topic | Raise `PARENT_CHUNK_TOKENS` / `CHILD_CHUNK_TOKENS` |
| Query latency too high | Lower `QDRANT_SEARCH_HNSW_EF` |

Changing `EMBEDDING_MODEL` or `EMBEDDING_DIM` requires a reindex:
```bash
python scripts/ingest.py --reset --dir data/synthetic
```

---

## Scaling to 100k+ PDFs

Already in place:

- **int8 scalar quantization** with rescoring + 2x oversampling — ~4x less vector RAM
- **On-disk payloads and vectors** — only the HNSW graph stays hot
- **Keyword payload indexes** on `doc_id`, `parent_id`, `source`, `doc_type` — filters stay sub-linear
- **`m=32`, `ef_construct=256`** — tuned for recall at high vector counts
- **Batched upserts** with deterministic IDs — safe to retry, no duplicates
- **Parents excluded from the vector store** — ~5x fewer vectors than chunking naively

### Generating a large corpus

```bash
# 10,000 PDFs (~2 min)
python scripts/generate_large_corpus.py --count 10000 --out data/synthetic

# 100,000 PDFs (~20 min)
python scripts/generate_large_corpus.py --count 100000 --out data/synthetic

# 1,000,000 PDFs (~3 hours)
python scripts/generate_large_corpus.py --count 1000000 --out data/large
```

### Scaling further

- Shard the collection (`shard_number` > 1) across a Qdrant cluster
- Move ingestion behind a queue (the `ingest_workers` config is the seam)
- Add a cross-encoder reranker between retrieval and generation
- Use GPU for embeddings (`EMBEDDING_DEVICE=cuda`)

---

## Testing

```bash
pytest tests/ -v
```

Covers chunking invariants, deterministic IDs, BM25 encoding, docstore round-trips, guardrail branches, and rate-limit mapping. No Qdrant, Postgres, or API key needed.

---

## Linting

```bash
ruff check backend/ frontend/ tests/ scripts/
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Qdrant is not reachable` | `docker compose up -d qdrant`, then `curl localhost:6333/healthz` |
| `GOOGLE_API_KEY is not set` (HTTP 503) | Add the key to `.env` and restart uvicorn |
| Falls back to JSON docstore | Expected without Postgres; `docker compose up -d postgres` to use it |
| First query is slow | One-time embedding model download + load; subsequent queries are warm |
| Everything returns the fallback | Check `/stats` — `vectors: 0` means nothing was ingested |
| Vector dimension mismatch | Changed the embedding model; run `python scripts/ingest.py --reset --dir data/synthetic` |
| Qdrant unhealthy after bulk ingest | `docker restart rag-qdrant`, then re-run ingestion |

---

## License

MIT
