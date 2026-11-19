"""Threshold tuner — sweep the cache similarity threshold on paraphrase / near-miss pairs.

Usage:
    python -m eval.threshold_tuner
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.embeddings import embed
from tqdm import tqdm


def load_pairs(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def compute_similarities(pairs: list[dict]) -> list[float]:
    """Compute cosine similarity for each pair."""
    sims = []
    for pair in tqdm(pairs, desc="Embedding pairs"):
        v1 = np.array(embed(pair["q1"]))
        v2 = np.array(embed(pair["q2"]))
        sim = float(np.dot(v1, v2))  # vectors are already normalised
        sims.append(sim)
    return sims


def main():
    eval_dir = Path(__file__).resolve().parent / "data"

    print("=" * 70)
    print("SEMANTIC CACHE THRESHOLD TUNER")
    print("=" * 70)

    # Load pairs
    para_pairs = load_pairs(str(eval_dir / "paraphrase_pairs.json"))
    miss_pairs = load_pairs(str(eval_dir / "near_miss_pairs.json"))

    print(f"\n📦 Loaded {len(para_pairs)} paraphrase pairs (should HIT cache)")
    print(f"📦 Loaded {len(miss_pairs)} near-miss pairs (must NOT hit cache)")

    # Compute similarities
    print("\n🔄 Computing embeddings for paraphrase pairs...")
    para_sims = compute_similarities(para_pairs)

    print("🔄 Computing embeddings for near-miss pairs...")
    miss_sims = compute_similarities(miss_pairs)

    print(f"\n📊 Paraphrase similarity range: {min(para_sims):.4f} – {max(para_sims):.4f}")
    print(f"   Mean: {np.mean(para_sims):.4f}, Median: {np.median(para_sims):.4f}")
    print(f"📊 Near-miss similarity range:  {min(miss_sims):.4f} – {max(miss_sims):.4f}")
    print(f"   Mean: {np.mean(miss_sims):.4f}, Median: {np.median(miss_sims):.4f}")

    # Sweep thresholds
    print("\n" + "=" * 70)
    print("THRESHOLD SWEEP")
    print("=" * 70)
    print(f"{'Threshold':>10}  {'Para Hits':>10}  {'Miss Hits':>10}  {'False Hits':>11}  {'Verdict':>10}")
    print("-" * 70)

    best_threshold = None
    for t in np.arange(0.85, 0.98, 0.01):
        para_hits = sum(1 for s in para_sims if s >= t)
        miss_hits = sum(1 for s in miss_sims if s >= t)
        verdict = "✅ SAFE" if miss_hits == 0 else "❌ UNSAFE"
        print(f"{t:>10.2f}  {para_hits:>10}  {miss_hits:>10}  {miss_hits:>11}  {verdict:>10}")

        if miss_hits == 0 and best_threshold is None:
            best_threshold = t

    print("-" * 70)

    if best_threshold is not None:
        # Find the best — lowest threshold with zero false hits
        # But re-sweep more finely
        fine_best = None
        for t in np.arange(0.85, 0.98, 0.005):
            miss_hits = sum(1 for s in miss_sims if s >= t)
            if miss_hits == 0:
                fine_best = t
                break

        if fine_best:
            para_hits = sum(1 for s in para_sims if s >= fine_best)
            print(f"\n🎯 Recommended threshold: {fine_best:.3f}")
            print(f"   Cache hit rate on paraphrases: {para_hits}/{len(para_sims)} ({para_hits/len(para_sims)*100:.1f}%)")
            print(f"   False hits on near-misses: 0/{len(miss_sims)} (0.0%)")
        else:
            print(f"\n🎯 Recommended threshold: {best_threshold:.2f}")
    else:
        print("\n⚠️  No threshold achieves zero false hits in this range.")
        print("    Consider using a different embedding model or adding more near-miss pairs.")

    # Detailed breakdown of near-misses close to threshold
    print("\n" + "=" * 70)
    print("TOP 10 NEAR-MISS PAIRS BY SIMILARITY (most dangerous)")
    print("=" * 70)
    indexed = list(zip(miss_sims, miss_pairs))
    indexed.sort(key=lambda x: x[0], reverse=True)
    for i, (sim, pair) in enumerate(indexed[:10]):
        print(f"  {i+1}. sim={sim:.4f}")
        print(f"     Q1: {pair['q1']}")
        print(f"     Q2: {pair['q2']}")

    # Bottom paraphrase pairs (hardest to cache)
    print("\n" + "=" * 70)
    print("BOTTOM 10 PARAPHRASE PAIRS BY SIMILARITY (hardest to cache)")
    print("=" * 70)
    para_indexed = list(zip(para_sims, para_pairs))
    para_indexed.sort(key=lambda x: x[0])
    for i, (sim, pair) in enumerate(para_indexed[:10]):
        print(f"  {i+1}. sim={sim:.4f}")
        print(f"     Q1: {pair['q1']}")
        print(f"     Q2: {pair['q2']}")


if __name__ == "__main__":
    main()
