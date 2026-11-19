"""Benchmark — compare the baseline (always large model) against smart routing + caching.

Usage:
    # Start the server first:  uvicorn app.main:app --reload
    # Then run:                 python -m eval.benchmark
"""

from __future__ import annotations

import json
import time
import sys
import random
from pathlib import Path

import httpx
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

API_BASE = "http://localhost:8000"
EVAL_DIR = Path(__file__).resolve().parent / "data"


def load_eval_queries() -> list[dict]:
    with open(EVAL_DIR / "eval_queries.json") as f:
        return json.load(f)


def load_paraphrase_pairs() -> list[dict]:
    with open(EVAL_DIR / "paraphrase_pairs.json") as f:
        return json.load(f)


def send_query(query: str) -> dict | None:
    try:
        r = httpx.post(
            f"{API_BASE}/query",
            json={"query": query},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  ⚠️  Error: {e}")
        return None


def run_benchmark():
    print("=" * 70)
    print("BENCHMARK: Baseline vs Smart Routing + Caching")
    print("=" * 70)

    # Check server health
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=5)
        r.raise_for_status()
        health = r.json()
        print(f"\n✅ Server healthy | Cache entries: {health['cache_entries']}")
        print(f"   Small model: {health['small_model']} | Large model: {health['large_model']}")
    except Exception:
        print(f"\n❌ Server not reachable at {API_BASE}")
        print("   Start it with: uvicorn app.main:app --port 8000")
        sys.exit(1)

    # Load queries
    eval_queries = load_eval_queries()
    para_pairs = load_paraphrase_pairs()

    # Build query list: original queries + paraphrased versions (simulate repeat traffic)
    queries = []

    # Phase 1: unique queries
    unique_qs = [q["query"] for q in eval_queries[:30]]  # Use first 30 eval queries
    queries.extend(unique_qs)

    # Phase 2: repeat some queries (exact duplicates — should cache-hit)
    queries.extend(random.sample(unique_qs, min(10, len(unique_qs))))

    # Phase 3: paraphrased versions (should cache-hit via semantic similarity)
    para_qs = [p["q2"] for p in para_pairs[:15]]
    # First send originals
    original_qs = [p["q1"] for p in para_pairs[:15]]
    queries.extend(original_qs)
    # Then paraphrases
    queries.extend(para_qs)

    print(f"\n📋 Benchmark plan: {len(queries)} total queries")
    print(f"   - {len(unique_qs)} unique eval queries")
    print(f"   - 10 exact repeats")
    print(f"   - {len(original_qs)} paraphrase originals + {len(para_qs)} paraphrased versions")

    # Run all queries
    print("\n🚀 Running queries...\n")
    results = []
    for i, q in enumerate(tqdm(queries, desc="Queries")):
        result = send_query(q)
        if result:
            results.append(result)

    if not results:
        print("❌ No results collected.")
        return

    # Analyse results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    total = len(results)
    cache_hits = sum(1 for r in results if r["cache_hit"])
    cache_misses = total - cache_hits

    costs = [r["cost"] for r in results]
    latencies = [r["latency_ms"] for r in results]
    quality_scores = [r["quality_score"] for r in results if not r["cache_hit"]]

    # Route breakdown
    routes = {}
    for r in results:
        route = r["route"]
        routes[route] = routes.get(route, 0) + 1

    print(f"\n📊 Total Requests:    {total}")
    print(f"   Cache Hits:        {cache_hits} ({cache_hits/total*100:.1f}%)")
    print(f"   Cache Misses:      {cache_misses}")

    print(f"\n🔀 Route Breakdown:")
    for route, count in sorted(routes.items()):
        print(f"   {route:>12}: {count}")

    print(f"\n💰 Cost Analysis:")
    actual_total_cost = sum(costs)
    avg_cost = actual_total_cost / total if total else 0
    print(f"   Total Cost (actual):    ${actual_total_cost:.6f}")
    print(f"   Avg Cost / request:     ${avg_cost:.6f}")

    # Estimate baseline cost (if all went through large model, no cache)
    non_cache = [r for r in results if not r["cache_hit"]]
    if non_cache:
        large_in_rate = 2.50 / 1_000_000
        large_out_rate = 10.00 / 1_000_000
        baseline_cost = sum(
            r["tokens_in"] * large_in_rate + r["tokens_out"] * large_out_rate
            for r in non_cache
        )
        # Scale to total (assume cache hits would have same avg token counts)
        avg_tokens_in = np.mean([r["tokens_in"] for r in non_cache])
        avg_tokens_out = np.mean([r["tokens_out"] for r in non_cache])
        baseline_total = total * (avg_tokens_in * large_in_rate + avg_tokens_out * large_out_rate)

        savings = ((baseline_total - actual_total_cost) / baseline_total * 100) if baseline_total > 0 else 0

        print(f"   Baseline Cost (est):    ${baseline_total:.6f}")
        print(f"   💡 Savings:              {savings:.1f}%")

    print(f"\n⏱️  Latency Analysis:")
    print(f"   Avg Latency:        {np.mean(latencies):.0f} ms")
    print(f"   p50 Latency:        {np.percentile(latencies, 50):.0f} ms")
    print(f"   p95 Latency:        {np.percentile(latencies, 95):.0f} ms")
    print(f"   Min Latency:        {min(latencies):.0f} ms")
    print(f"   Max Latency:        {max(latencies):.0f} ms")

    cache_latencies = [r["latency_ms"] for r in results if r["cache_hit"]]
    miss_latencies = [r["latency_ms"] for r in results if not r["cache_hit"]]
    if cache_latencies:
        print(f"   Cache Hit p50:      {np.percentile(cache_latencies, 50):.0f} ms")
    if miss_latencies:
        print(f"   Cache Miss p50:     {np.percentile(miss_latencies, 50):.0f} ms")

    if quality_scores:
        print(f"\n📈 Quality (non-cached):")
        print(f"   Avg Quality Score:  {np.mean(quality_scores):.4f}")
        print(f"   Min Quality Score:  {min(quality_scores):.4f}")
        print(f"   Max Quality Score:  {max(quality_scores):.4f}")

    # Save results
    output_path = EVAL_DIR / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "total_queries": total,
                "cache_hits": cache_hits,
                "cache_hit_rate": round(cache_hits / total, 4),
                "total_cost": round(actual_total_cost, 8),
                "avg_cost": round(avg_cost, 8),
                "baseline_cost_estimate": round(baseline_total, 8) if non_cache else 0,
                "savings_pct": round(savings, 2) if non_cache else 0,
                "avg_latency_ms": round(float(np.mean(latencies)), 2),
                "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2),
                "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
                "route_split": routes,
                "avg_quality_score": round(float(np.mean(quality_scores)), 4) if quality_scores else 0,
            },
            f,
            indent=2,
        )
    print(f"\n💾 Results saved to {output_path}")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
