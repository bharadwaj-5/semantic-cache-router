"""Centralised configuration loaded from environment / .env file."""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env from project root ──────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ── LLM Models ───────────────────────────────────────────────
SMALL_MODEL: str = os.getenv("SMALL_MODEL", "gpt-4o-mini")
LARGE_MODEL: str = os.getenv("LARGE_MODEL", "gpt-4o")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM: int = 384  # all-MiniLM-L6-v2 output dimensionality

# ── Cost table (USD per 1 M tokens) ─────────────────────────
MODEL_COSTS: dict[str, dict[str, float]] = {
    "gpt-4o-mini":    {"input": 0.15,  "output": 0.60},
    "gpt-4o":         {"input": 2.50,  "output": 10.00},
    "gpt-4-turbo":    {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo":  {"input": 0.50,  "output": 1.50},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307":    {"input": 0.25, "output": 1.25},
    "gemini/gemini-1.5-flash":    {"input": 0.075, "output": 0.30},
    "gemini/gemini-1.5-pro":      {"input": 1.25, "output": 5.00},
}

# ── Semantic Cache ───────────────────────────────────────────
CACHE_SIMILARITY_THRESHOLD: float = float(
    os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.92")
)
CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
CORPUS_VERSION: str = os.getenv("CORPUS_VERSION", "v1")
QDRANT_CACHE_COLLECTION: str = "query_cache"
QDRANT_DOCS_COLLECTION: str = "documents"

# ── Router ───────────────────────────────────────────────────
ESCALATION_MIN_RETRIEVAL_SCORE: float = float(
    os.getenv("ESCALATION_MIN_RETRIEVAL_SCORE", "0.3")
)

# ── Retrieval ────────────────────────────────────────────────
TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "5"))
CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 64

# ── Persistence ──────────────────────────────────────────────
DB_PATH: str = os.getenv("DB_PATH", str(_PROJECT_ROOT / "db" / "request_logs.db"))

# ── Logging ──────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
