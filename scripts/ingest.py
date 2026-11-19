"""Ingest documents from a directory into the Qdrant vector store."""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from app.rag import ingest_documents
from app.cache import SemanticCache


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into vector store")
    parser.add_argument(
        "--dir",
        default="data/documents",
        help="Directory containing .md / .txt files (default: data/documents)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the semantic cache after ingestion",
    )
    args = parser.parse_args()

    client = QdrantClient(location=":memory:")
    count = ingest_documents(args.dir, client)
    print(f"✅ Ingested {count} chunks from {args.dir}")

    if args.clear_cache:
        cache = SemanticCache(client=client)
        cache.invalidate()
        print("🗑️  Semantic cache cleared")


if __name__ == "__main__":
    main()
