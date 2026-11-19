"""Query complexity router — classifies queries as 'simple' or 'complex'.

Rule-based v1 that checks surface signals:
  • query length (token count)
  • presence of reasoning keywords
  • entity / number density
  • spread of retrieval scores (if available)

A tiny LLM-based classifier can be swapped in later.
"""

from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)

# ── Signal keywords ──────────────────────────────────────────
_COMPLEX_KEYWORDS: set[str] = {
    "compare", "contrast", "why", "explain", "analyse", "analyze",
    "summarise", "summarize", "evaluate", "trade-off", "tradeoff",
    "pros and cons", "differences", "implications", "reasoning",
    "step by step", "in detail", "how does", "what are the advantages",
    "argue", "justify", "critically",
}

_SIMPLE_KEYWORDS: set[str] = {
    "what is", "define", "who is", "when", "where",
    "list", "name", "how to", "how do i",
}


def _token_count(text: str) -> int:
    """Rough whitespace tokeniser — good enough for routing."""
    return len(text.split())


def _entity_density(text: str) -> float:
    """Fraction of tokens that look like proper nouns, numbers, or acronyms."""
    tokens = text.split()
    if not tokens:
        return 0.0
    entities = sum(
        1 for t in tokens
        if t[0].isupper() or t.isnumeric() or t.isupper()
    )
    return entities / len(tokens)


def _has_complex_keyword(text: str) -> bool:
    low = text.lower()
    return any(kw in low for kw in _COMPLEX_KEYWORDS)


def _has_simple_keyword(text: str) -> bool:
    low = text.lower()
    return any(kw in low for kw in _SIMPLE_KEYWORDS)


# ── Main classifier ─────────────────────────────────────────
def classify_query(
    query: str,
    retrieval_scores: list[float] | None = None,
) -> str:
    """Return ``'simple'`` or ``'complex'``.

    Parameters
    ----------
    query : str
        The raw user query.
    retrieval_scores : list[float] | None
        Top-K retrieval similarity scores (if already computed). A wide spread
        suggests the query is ambiguous → complex.
    """
    score = 0  # positive = complex

    # 1️⃣ Length signal
    toks = _token_count(query)
    if toks > 30:
        score += 2
    elif toks > 15:
        score += 1

    # 2️⃣ Keyword signal
    if _has_complex_keyword(query):
        score += 3
    if _has_simple_keyword(query):
        score -= 2

    # 3️⃣ Entity density
    if _entity_density(query) > 0.4:
        score += 1

    # 4️⃣ Multi-part question (presence of "and", semicolons, commas)
    if re.search(r"\band\b.*\?|;", query):
        score += 1

    # 5️⃣ Retrieval-score spread
    if retrieval_scores and len(retrieval_scores) >= 2:
        spread = max(retrieval_scores) - min(retrieval_scores)
        if spread > 0.3:
            score += 2  # results are scattered → ambiguous

    route = "complex" if score >= 3 else "simple"
    logger.debug("Router score %d → %s  |  query=%s", score, route, query[:60])
    return route


# ── Escalation check ────────────────────────────────────────
def needs_escalation(
    answer: str,
    retrieval_scores: list[float],
    min_score: float = 0.3,
) -> bool:
    """Decide whether a small-model answer should be re-run on the large model.

    Triggers when:
    - best retrieval score is below ``min_score``
    - the answer contains hedging / refusal language
    - the answer is suspiciously short
    """
    if not retrieval_scores or max(retrieval_scores) < min_score:
        return True

    hedges = ["i don't know", "i'm not sure", "i cannot", "unclear",
              "not enough information", "no relevant"]
    low = answer.lower()
    if any(h in low for h in hedges):
        return True

    if len(answer.split()) < 8:
        return True

    return False
