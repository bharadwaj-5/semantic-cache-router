"""RAG pipeline — retrieval from Qdrant + generation via LiteLLM."""

from __future__ import annotations

import uuid
import time
import logging
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

import litellm

from app.config import (
    CORPUS_VERSION,
    EMBEDDING_DIM,
    LARGE_MODEL,
    MODEL_COSTS,
    QDRANT_DOCS_COLLECTION,
    SMALL_MODEL,
    TOP_K_RETRIEVAL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from app.embeddings import embed, embed_batch
from app.models import SourceChunk

logger = logging.getLogger(__name__)

# Silence litellm's noisy logging
litellm.suppress_debug_info = True


# ── Document ingestion ───────────────────────────────────────
def _chunk_text(text: str, source: str) -> list[dict]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks: list[dict] = []
    start = 0
    while start < len(words):
        end = start + CHUNK_SIZE
        chunk_text = " ".join(words[start:end])
        if chunk_text.strip():
            chunks.append({"text": chunk_text, "document": source})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def ingest_documents(
    directory: str,
    client: QdrantClient,
) -> int:
    """Read .md and .txt files from *directory*, chunk, embed, and upsert.

    Returns the number of chunks ingested.
    """
    doc_dir = Path(directory)
    if not doc_dir.exists():
        raise FileNotFoundError(f"Document directory not found: {doc_dir}")

    # Ensure collection
    collections = [c.name for c in client.get_collections().collections]
    if QDRANT_DOCS_COLLECTION in collections:
        client.delete_collection(QDRANT_DOCS_COLLECTION)
    client.create_collection(
        collection_name=QDRANT_DOCS_COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    # Gather chunks from all files
    all_chunks: list[dict] = []
    for fpath in sorted(doc_dir.iterdir()):
        if fpath.suffix in (".md", ".txt"):
            text = fpath.read_text(encoding="utf-8")
            chunks = _chunk_text(text, source=fpath.name)
            all_chunks.extend(chunks)
            logger.info("Chunked %s → %d chunks", fpath.name, len(chunks))

    if not all_chunks:
        logger.warning("No documents found in %s", directory)
        return 0

    # Embed in batch
    texts = [c["text"] for c in all_chunks]
    vectors = embed_batch(texts)

    # Upsert
    points = [
        PointStruct(
            id=uuid.uuid4().hex,
            vector=vec,
            payload={"text": chunk["text"], "document": chunk["document"]},
        )
        for vec, chunk in zip(vectors, all_chunks)
    ]
    # Upsert in batches of 100
    for i in range(0, len(points), 100):
        client.upsert(
            collection_name=QDRANT_DOCS_COLLECTION,
            points=points[i : i + 100],
        )

    logger.info("Ingested %d chunks from %s", len(points), directory)
    return len(points)


# ── Retrieval ────────────────────────────────────────────────
def retrieve(
    query: str,
    client: QdrantClient,
    top_k: int = TOP_K_RETRIEVAL,
) -> tuple[list[SourceChunk], list[float]]:
    """Retrieve the top-K chunks and return (chunks, scores)."""
    q_vec = embed(query)
    response = client.query_points(
        collection_name=QDRANT_DOCS_COLLECTION,
        query=q_vec,
        limit=top_k,
    )
    chunks = [
        SourceChunk(
            text=hit.payload["text"],
            document=hit.payload["document"],
            score=round(hit.score, 4),
        )
        for hit in response.points
    ]
    scores = [hit.score for hit in response.points]
    return chunks, scores


# ── Generation ───────────────────────────────────────────────
_SYSTEM_PROMPT = (
    "You are a helpful enterprise assistant. Answer the user's question "
    "using ONLY the provided context. If the context does not contain the "
    "answer, say so clearly. Cite the source document name when possible."
)


def generate(
    query: str,
    context_chunks: list[SourceChunk],
    model: str = SMALL_MODEL,
) -> dict:
    """Call the LLM and return answer, token counts, cost and latency.

    Returns
    -------
    dict with keys: answer, tokens_in, tokens_out, cost, latency_ms
    """
    # Build context block
    context_block = "\n\n---\n\n".join(
        f"[Source: {c.document}]\n{c.text}" for c in context_chunks
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context:\n{context_block}\n\n"
                f"Question: {query}\n\n"
                "Answer:"
            ),
        },
    ]

    t0 = time.time()
    try:
        response = litellm.completion(model=model, messages=messages, temperature=0.1)
    except Exception as e:
        logger.error("LLM call failed (%s): %s", model, e)
        return {
            "answer": f"Error: LLM call failed — {e}",
            "tokens_in": 0,
            "tokens_out": 0,
            "cost": 0.0,
            "latency_ms": round((time.time() - t0) * 1000, 2),
        }
    latency_ms = round((time.time() - t0) * 1000, 2)

    answer = response.choices[0].message.content.strip()
    usage = response.usage
    tokens_in = usage.prompt_tokens if usage else 0
    tokens_out = usage.completion_tokens if usage else 0

    # Compute cost
    cost_table = MODEL_COSTS.get(model, {"input": 0.0, "output": 0.0})
    cost = (tokens_in * cost_table["input"] + tokens_out * cost_table["output"]) / 1_000_000

    return {
        "answer": answer,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost": round(cost, 8),
        "latency_ms": latency_ms,
    }


# ── Full RAG pipeline (single call) ─────────────────────────
def rag_pipeline(
    query: str,
    model: str,
    client: QdrantClient,
) -> dict:
    """Retrieve → Generate. Returns dict with answer, sources, scores, tokens, cost, latency."""
    chunks, scores = retrieve(query, client)
    gen = generate(query, chunks, model=model)

    quality_score = round(sum(scores) / len(scores), 4) if scores else 0.0

    return {
        "answer": gen["answer"],
        "sources": chunks,
        "scores": scores,
        "tokens_in": gen["tokens_in"],
        "tokens_out": gen["tokens_out"],
        "cost": gen["cost"],
        "latency_ms": gen["latency_ms"],
        "quality_score": quality_score,
    }
