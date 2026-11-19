"""Semantic cache backed by Qdrant (in-memory mode — no server needed)."""

from __future__ import annotations

import time
import uuid
import logging
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from app.config import (
    CACHE_SIMILARITY_THRESHOLD,
    CACHE_TTL_SECONDS,
    CORPUS_VERSION,
    EMBEDDING_DIM,
    QDRANT_CACHE_COLLECTION,
)
from app.models import CacheEntry, SourceChunk

logger = logging.getLogger(__name__)


class SemanticCache:
    """Vector-similarity cache with corpus-version gating and TTL."""

    def __init__(self, client: QdrantClient | None = None):
        self.client = client or QdrantClient(location=":memory:")
        self._ensure_collection()

    # ── Collection bootstrap ─────────────────────────────────
    def _ensure_collection(self) -> None:
        collections = [c.name for c in self.client.get_collections().collections]
        if QDRANT_CACHE_COLLECTION not in collections:
            self.client.create_collection(
                collection_name=QDRANT_CACHE_COLLECTION,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection: %s", QDRANT_CACHE_COLLECTION)

    # ── Lookup ───────────────────────────────────────────────
    def lookup(
        self,
        query_vec: list[float],
        corpus_version: str = CORPUS_VERSION,
    ) -> Optional[dict]:
        """Return the best cached entry if similarity ≥ threshold, else None.

        Returns a dict with keys: score, entry (CacheEntry dict).
        """
        response = self.client.query_points(
            collection_name=QDRANT_CACHE_COLLECTION,
            query=query_vec,
            limit=1,
            score_threshold=CACHE_SIMILARITY_THRESHOLD,
        )
        if not response.points:
            return None

        hit = response.points[0]
        payload = hit.payload or {}

        # ── Version gate ─────────────────────────────────────
        if payload.get("corpus_version") != corpus_version:
            logger.debug("Cache hit rejected: version mismatch (%s vs %s)",
                         payload.get("corpus_version"), corpus_version)
            return None

        # ── TTL gate ─────────────────────────────────────────
        created = payload.get("created_at_ts", 0)
        if time.time() - created > CACHE_TTL_SECONDS:
            logger.debug("Cache hit rejected: TTL expired")
            return None

        return {"score": hit.score, "entry": payload}

    # ── Store ────────────────────────────────────────────────
    def store(
        self,
        query_vec: list[float],
        entry: CacheEntry,
    ) -> None:
        """Persist a new cache entry. Never cache empty answers."""
        if not entry.answer or not entry.answer.strip():
            logger.debug("Skipping cache store: empty answer")
            return

        payload = entry.model_dump()
        payload["created_at_ts"] = time.time()

        self.client.upsert(
            collection_name=QDRANT_CACHE_COLLECTION,
            points=[
                PointStruct(
                    id=uuid.uuid4().hex,
                    vector=query_vec,
                    payload=payload,
                )
            ],
        )
        logger.debug("Cached answer for: %s", entry.query[:80])

    # ── Invalidation ─────────────────────────────────────────
    def invalidate(self, corpus_version: str | None = None) -> int:
        """Delete entries for a given corpus version (or all)."""
        if corpus_version:
            # Scroll and delete matching points
            points, _ = self.client.scroll(
                collection_name=QDRANT_CACHE_COLLECTION,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="corpus_version",
                            match=MatchValue(value=corpus_version),
                        )
                    ]
                ),
                limit=10_000,
            )
            if points:
                ids = [p.id for p in points]
                self.client.delete(
                    collection_name=QDRANT_CACHE_COLLECTION,
                    points_selector=ids,
                )
                return len(ids)
            return 0
        else:
            # Nuke everything
            self.client.delete_collection(QDRANT_CACHE_COLLECTION)
            self._ensure_collection()
            return -1  # signals full wipe

    @property
    def count(self) -> int:
        info = self.client.get_collection(QDRANT_CACHE_COLLECTION)
        return info.points_count
