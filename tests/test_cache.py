"""Tests for the semantic cache and router components."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from qdrant_client import QdrantClient

from app.cache import SemanticCache
from app.embeddings import embed, normalize
from app.router import classify_query, needs_escalation
from app.models import CacheEntry
from app.config import CORPUS_VERSION


# ── Normalisation tests ──────────────────────────────────────
class TestNormalize:
    def test_lowercase(self):
        assert normalize("Hello World") == "hello world"

    def test_strip_punctuation(self):
        assert normalize("What's up?!") == "what s up"

    def test_collapse_whitespace(self):
        assert normalize("too   many    spaces") == "too many spaces"

    def test_strip_leading_trailing(self):
        assert normalize("  hello  ") == "hello"


# ── Embedding tests ──────────────────────────────────────────
class TestEmbeddings:
    def test_embed_returns_list(self):
        vec = embed("test query")
        assert isinstance(vec, list)
        assert len(vec) == 384  # all-MiniLM-L6-v2

    def test_similar_queries_high_score(self):
        import numpy as np
        v1 = np.array(embed("How do I reset my password?"))
        v2 = np.array(embed("Password reset steps"))
        sim = float(np.dot(v1, v2))
        assert sim > 0.7  # Should be semantically similar

    def test_different_queries_lower_score(self):
        import numpy as np
        v1 = np.array(embed("What is the refund policy?"))
        v2 = np.array(embed("How do I set up VPN?"))
        sim = float(np.dot(v1, v2))
        assert sim < 0.7  # Should be less similar


# ── Cache tests ──────────────────────────────────────────────
class TestSemanticCache:
    def setup_method(self):
        self.client = QdrantClient(location=":memory:")
        self.cache = SemanticCache(client=self.client)

    def test_store_and_lookup(self):
        query = "What is the refund policy?"
        vec = embed(query)
        entry = CacheEntry(
            query=query,
            answer="You can return items within 30 days.",
            sources=[],
            model_used="gpt-4o-mini",
            corpus_version=CORPUS_VERSION,
        )
        self.cache.store(vec, entry)
        assert self.cache.count == 1

        hit = self.cache.lookup(vec, corpus_version=CORPUS_VERSION)
        assert hit is not None
        assert hit["entry"]["answer"] == "You can return items within 30 days."

    def test_version_mismatch_no_hit(self):
        query = "test query"
        vec = embed(query)
        entry = CacheEntry(
            query=query,
            answer="test answer",
            sources=[],
            model_used="gpt-4o-mini",
            corpus_version="old_version",
        )
        self.cache.store(vec, entry)
        hit = self.cache.lookup(vec, corpus_version="new_version")
        assert hit is None

    def test_empty_answer_not_cached(self):
        vec = embed("test")
        entry = CacheEntry(
            query="test",
            answer="",
            sources=[],
            model_used="gpt-4o-mini",
            corpus_version=CORPUS_VERSION,
        )
        self.cache.store(vec, entry)
        assert self.cache.count == 0

    def test_invalidate(self):
        vec = embed("test query")
        entry = CacheEntry(
            query="test",
            answer="answer",
            sources=[],
            model_used="gpt-4o-mini",
            corpus_version=CORPUS_VERSION,
        )
        self.cache.store(vec, entry)
        assert self.cache.count == 1

        self.cache.invalidate()
        assert self.cache.count == 0


# ── Router tests ─────────────────────────────────────────────
class TestRouter:
    def test_simple_query(self):
        assert classify_query("What is the dress code?") == "simple"

    def test_complex_query(self):
        result = classify_query(
            "Compare the health insurance plans and explain which is better "
            "for a family of four considering premiums, deductibles, and coverage"
        )
        assert result == "complex"

    def test_short_query_simple(self):
        assert classify_query("List holidays") == "simple"

    def test_needs_escalation_low_scores(self):
        assert needs_escalation("Some answer", [0.1, 0.2], min_score=0.3) is True

    def test_no_escalation_good_scores(self):
        result = needs_escalation(
            "The refund policy allows returns within 30 days of purchase. "
            "Contact customer support for assistance.",
            [0.7, 0.6, 0.5],
            min_score=0.3,
        )
        assert result is False

    def test_escalation_on_hedge(self):
        assert needs_escalation(
            "I don't know the answer to that.",
            [0.8, 0.7],
            min_score=0.3,
        ) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
