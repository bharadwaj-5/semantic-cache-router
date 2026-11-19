"""Router accuracy evaluator — tests the rule-based classifier against labelled queries.

Usage:
    python -m eval.router_eval
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.router import classify_query


def main():
    eval_path = Path(__file__).resolve().parent / "data" / "eval_queries.json"
    with open(eval_path) as f:
        queries = json.load(f)

    correct = 0
    total = len(queries)
    errors = []

    for item in queries:
        predicted = classify_query(item["query"])
        expected = item["expected_route"]
        if predicted == expected:
            correct += 1
        else:
            errors.append({
                "query": item["query"],
                "expected": expected,
                "predicted": predicted,
            })

    accuracy = correct / total * 100 if total else 0

    print("=" * 70)
    print("ROUTER ACCURACY EVALUATION")
    print("=" * 70)
    print(f"\n📊 Results: {correct}/{total} correct ({accuracy:.1f}%)")
    print(f"   Errors: {len(errors)}")

    if errors:
        print(f"\n❌ Misclassified Queries:")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. [{err['expected']} → {err['predicted']}] {err['query'][:80]}")

    print("\n" + "=" * 70)

    return accuracy


if __name__ == "__main__":
    main()
