# ⚡ Semantic Cache Router

> **Cost-aware LLM routing with semantic caching** — route simple queries to a small model, complex ones to a large model, cache answers by query similarity, and prove the savings on a real-time dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41-FF4B4B?logo=streamlit)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC382D)
![LiteLLM](https://img.shields.io/badge/LiteLLM-Multi_Model-orange)

---

## 🎯 Problem Statement

Enterprise RAG systems face two compounding cost problems:

1. **Every query hits the most expensive model** — even when 60-70% of queries are simple look-ups that a cheap model handles equally well.
2. **Semantically identical questions re-run the full pipeline** — "How do I reset my password?" and "Password reset steps" generate the same answer but pay for two full LLM calls.

This project solves both with a **semantic cache** and a **cost-aware router**, then **measures the savings** with hard numbers.

---

## 🏗️ Architecture

```
                    ┌──────────────┐
                    │   FastAPI    │
                    │   /query     │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Normalize   │
                    │  & Embed     │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐     YES    ┌────────────┐
                    │  Cache Hit?  ├────────────►│  Return    │
                    │  (Qdrant)    │             │  Cached    │
                    └──────┬───────┘             └────────────┘
                           │ NO
                    ┌──────▼───────┐
                    │   Router     │
                    │  (Classify)  │
                    └──┬───────┬───┘
                       │       │
              simple   │       │  complex
                       │       │
              ┌────────▼─┐  ┌──▼──────────┐
              │ Small LLM │  │  Large LLM  │
              │(gpt-4o-   │  │  (gpt-4o)   │
              │  mini)    │  │             │
              └────┬──────┘  └──────┬──────┘
                   │                │
                   │  ┌─────────┐   │
                   ├──► Needs   ├───┤
                   │  │Escalate?│   │
                   │  └─────────┘   │
                   │                │
              ┌────▼────────────────▼──┐
              │  Cache Store + Log     │
              └────────────┬───────────┘
                           │
              ┌────────────▼───────────┐
              │  Response + Metrics    │
              └────────────────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Semantic Cache** | Caches LLM answers by query embedding similarity. Paraphrased questions hit the cache without re-running the pipeline. |
| **Corpus-Version Gating** | Cache entries are tagged with a corpus version. Re-ingesting documents invalidates stale answers automatically. |
| **TTL Expiry** | Configurable time-to-live prevents serving outdated answers even without re-ingestion. |
| **Cost-Aware Routing** | Rule-based classifier routes simple queries (definitions, look-ups) to `gpt-4o-mini` and complex queries (comparisons, analysis) to `gpt-4o`. |
| **Automatic Escalation** | If the small model's answer quality is low (weak retrieval scores, hedging language, too short), the query is transparently re-run on the large model. |
| **Request Logging** | Every request is logged to SQLite with query, model, tokens, cost, latency, route, and quality score. |
| **Real-Time Dashboard** | Streamlit dashboard showing cost per request, cache hit rate, route split, latency distribution, quality scores, and savings vs baseline. |
| **Threshold Tuning** | Evaluation script with 50 paraphrase pairs and 50 near-miss pairs to empirically find the optimal cache similarity threshold with zero false hits. |
| **Benchmark Suite** | End-to-end benchmark replays eval queries with realistic repeat traffic and reports cost savings, latency improvement, and quality delta. |

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/semantic-cache-router.git
cd semantic-cache-router

python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key (or any LiteLLM-supported provider)
```

### 3. Start the API server

```bash
uvicorn app.main:app --reload --port 8000
```

The server auto-ingests sample documents from `data/documents/` on startup.

### 4. Send a query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the company refund policy?"}'
```

### 5. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Open **http://localhost:8501** to see real-time metrics.

---

## 📊 Evaluation & Benchmarking

### Tune the cache threshold
```bash
python -m eval.threshold_tuner
```

Sweeps similarity thresholds from 0.85 to 0.97 on:
- **50 paraphrase pairs** (should hit cache) — e.g., "How do I reset my password?" ↔ "Password reset steps"
- **50 near-miss pairs** (must NOT hit cache) — e.g., "Refund policy in the EU" ↔ "Refund policy in the US"

Outputs the **lowest threshold with zero false hits**.

### Evaluate router accuracy
```bash
python -m eval.router_eval
```

Tests the rule-based classifier against 100 labelled queries (simple vs complex).

### Run the full benchmark
```bash
# Ensure the server is running
python -m eval.benchmark
```

Replays 70+ queries (unique + exact repeats + paraphrases) and reports:
- Cost reduction vs always-large-model baseline
- Cache hit rate with realistic repeat traffic
- p50 and p95 latency comparison
- Quality score analysis per route

---

## 📁 Project Structure

```
semantic-cache-router/
├── app/
│   ├── __init__.py
│   ├── config.py          # Centralised configuration from .env
│   ├── models.py          # Pydantic models (request, response, cache entry, logs)
│   ├── embeddings.py      # Sentence-transformer embedding service
│   ├── cache.py           # Semantic cache with Qdrant (in-memory)
│   ├── router.py          # Query complexity classifier + escalation logic
│   ├── rag.py             # RAG pipeline: chunking, retrieval, generation
│   ├── logger.py          # SQLite request logging
│   └── main.py            # FastAPI application
├── dashboard/
│   └── app.py             # Streamlit analytics dashboard
├── eval/
│   ├── benchmark.py       # End-to-end benchmark script
│   ├── threshold_tuner.py # Cache threshold tuning on test pairs
│   ├── router_eval.py     # Router accuracy evaluation
│   └── data/
│       ├── paraphrase_pairs.json   # 50 pairs that should cache-hit
│       ├── near_miss_pairs.json    # 50 pairs that must NOT cache-hit
│       └── eval_queries.json       # 100 labelled queries for router eval
├── data/
│   └── documents/         # Sample enterprise documents (Markdown)
│       ├── employee_handbook.md
│       ├── it_security_policy.md
│       ├── benefits_compensation.md
│       ├── product_customer_policy.md
│       └── facilities_operations.md
├── scripts/
│   └── ingest.py          # Standalone document ingestion script
├── db/                    # SQLite database (auto-created)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🔧 Configuration

All settings are in `.env` (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Your OpenAI API key (or any LiteLLM-supported key) |
| `SMALL_MODEL` | `gpt-4o-mini` | Model for simple queries |
| `LARGE_MODEL` | `gpt-4o` | Model for complex queries |
| `CACHE_SIMILARITY_THRESHOLD` | `0.92` | Minimum cosine similarity for a cache hit |
| `CACHE_TTL_SECONDS` | `3600` | Cache entry time-to-live |
| `CORPUS_VERSION` | `v1` | Bump to invalidate all cached answers |
| `ESCALATION_MIN_RETRIEVAL_SCORE` | `0.3` | Retrieval score below which small-model answers are escalated |
| `TOP_K_RETRIEVAL` | `5` | Number of chunks to retrieve |

---

## 🛡️ Design Decisions

### Why tune the threshold on near-miss pairs?
Most tutorials set a cache threshold by feel (0.85, 0.9) and move on. But similar-sounding queries can have **completely different correct answers**. Example:

> "Refund policy **in the EU**" vs "Refund policy **in the US**"

These embed very close together (cosine ~0.93) but require different answers. If your threshold is below that, **you serve wrong answers from cache**. The threshold tuner finds the lowest safe value empirically.

### Why rule-based routing first?
An LLM-based router adds latency and cost to every request. The rule-based approach handles 85%+ of queries correctly with zero added latency. The escalation mechanism catches the remaining edge cases.

### Why not use Redis for caching?
Qdrant runs in-memory mode with zero infrastructure — `QdrantClient(":memory:")`. For production, swap to a persistent Qdrant instance or Redis with the RediSearch module. The cache layer is abstracted to make this a one-line change.

---

## 🔌 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | Submit a query — returns answer + routing/cache/cost metadata |
| `POST` | `/ingest` | Re-ingest documents from a directory |
| `GET` | `/stats` | Aggregate metrics (cost, cache rate, latency percentiles) |
| `GET` | `/logs` | Raw request log data |
| `GET` | `/health` | System health check |

---

## 🧰 Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| API Server | **FastAPI** | Async, auto-docs, Pydantic validation |
| Embeddings | **sentence-transformers** (`all-MiniLM-L6-v2`) | Fast, 384-dim, good quality for semantic similarity |
| Vector Store | **Qdrant** (in-memory) | No server needed, clean Python API, production-ready |
| LLM Gateway | **LiteLLM** | One interface for OpenAI, Anthropic, Gemini, etc. |
| Logging | **SQLite** | Zero-config, good enough for analytics workloads |
| Dashboard | **Streamlit** + **Plotly** | Interactive charts, rapid prototyping |

---

## 📄 License

MIT

---

## 👤 Author

**Ammanamanchi Bharadwaj**

Built as a demonstration of cost-optimised LLM infrastructure with production-grade semantic caching, intelligent routing, and observability.
