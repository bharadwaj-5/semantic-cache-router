"""Request logging to SQLite — async-safe with aiosqlite."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from contextlib import contextmanager

from app.config import DB_PATH
from app.models import RequestLog

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS request_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    query           TEXT    NOT NULL,
    model_used      TEXT    DEFAULT '',
    tokens_in       INTEGER DEFAULT 0,
    tokens_out      INTEGER DEFAULT 0,
    cost            REAL    DEFAULT 0.0,
    latency_ms      REAL    DEFAULT 0.0,
    cache_hit       INTEGER DEFAULT 0,
    route           TEXT    DEFAULT '',
    quality_score   REAL    DEFAULT 0.0,
    corpus_version  TEXT    DEFAULT '',
    answer          TEXT    DEFAULT ''
);
"""


def _db_path() -> Path:
    p = Path(DB_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def init_db() -> None:
    """Create the database and table if they don't exist."""
    with sqlite3.connect(str(_db_path())) as conn:
        conn.executescript(_SCHEMA)
    logger.info("Initialised DB at %s", _db_path())


@contextmanager
def _connect():
    conn = sqlite3.connect(str(_db_path()))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_request(entry: RequestLog) -> int:
    """Insert a request log and return its row id."""
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO request_logs
                (timestamp, query, model_used, tokens_in, tokens_out,
                 cost, latency_ms, cache_hit, route, quality_score,
                 corpus_version, answer)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.timestamp,
                entry.query,
                entry.model_used,
                entry.tokens_in,
                entry.tokens_out,
                entry.cost,
                entry.latency_ms,
                int(entry.cache_hit),
                entry.route,
                entry.quality_score,
                entry.corpus_version,
                entry.answer,
            ),
        )
        return cur.lastrowid


def get_all_logs() -> list[dict]:
    """Return every row as a dict (for the dashboard)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM request_logs ORDER BY id DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_stats() -> dict:
    """Compute aggregate statistics."""
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM request_logs").fetchall()

    if not rows:
        return {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_hit_rate": 0.0,
            "avg_cost": 0.0,
            "total_cost": 0.0,
            "avg_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "route_split": {},
        }

    total = len(rows)
    hits = sum(1 for r in rows if r["cache_hit"])
    costs = [r["cost"] for r in rows]
    latencies = sorted(r["latency_ms"] for r in rows)

    route_split: dict[str, int] = {}
    for r in rows:
        route = r["route"] or "unknown"
        route_split[route] = route_split.get(route, 0) + 1

    p50_idx = int(len(latencies) * 0.50)
    p95_idx = min(int(len(latencies) * 0.95), len(latencies) - 1)

    return {
        "total_requests": total,
        "cache_hits": hits,
        "cache_hit_rate": round(hits / total, 4) if total else 0.0,
        "avg_cost": round(sum(costs) / total, 6) if total else 0.0,
        "total_cost": round(sum(costs), 6),
        "avg_latency_ms": round(sum(latencies) / total, 2) if total else 0.0,
        "p50_latency_ms": round(latencies[p50_idx], 2),
        "p95_latency_ms": round(latencies[p95_idx], 2),
        "route_split": route_split,
    }
