"""Embedding service using sentence-transformers."""

from __future__ import annotations

import re
import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# ── Singleton model loader ───────────────────────────────────
@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    return SentenceTransformer(EMBEDDING_MODEL)


# ── Text normalisation ───────────────────────────────────────
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(text: str) -> str:
    """Lower-case, collapse whitespace, strip punctuation."""
    text = text.lower().strip()
    text = _PUNCT.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Embed ────────────────────────────────────────────────────
def embed(text: str) -> list[float]:
    """Return the embedding vector for a single string."""
    model = _load_model()
    vec = model.encode(normalize(text), normalize_embeddings=True)
    return vec.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings efficiently."""
    model = _load_model()
    normalised = [normalize(t) for t in texts]
    vecs = model.encode(normalised, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vecs]
