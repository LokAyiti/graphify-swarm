"""
qdrant_store.py — local Qdrant vector store (no Docker required).

Uses QdrantClient(path=...) which runs an embedded storage engine
directly on disk.  All data survives process restarts.

Collection schema
-----------------
  vector   : float32[384]   cosine similarity
  payload  : repo, file_path, language, chunk_type, name,
             content, start_line, end_line
"""
from __future__ import annotations

from pathlib import Path
from typing import List
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from graphify.indexer.chunker import Chunk

_COLLECTION = "graphify_chunks"
_VECTOR_DIM = 384  # matches all-MiniLM-L6-v2


class QdrantStore:
    """Thin wrapper around a local Qdrant collection."""

    def __init__(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(data_dir))
        self._ensure_collection()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if _COLLECTION not in existing:
            self._client.create_collection(
                collection_name=_COLLECTION,
                vectors_config=VectorParams(
                    size=_VECTOR_DIM,
                    distance=Distance.COSINE,
                ),
            )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """Insert chunks with their embeddings into the collection."""
        if not chunks:
            return
        points = [
            PointStruct(
                id=str(uuid4()),
                vector=emb,
                payload={
                    "repo":       chunk.repo,
                    "file_path":  chunk.file_path,
                    "language":   chunk.language,
                    "chunk_type": chunk.chunk_type,
                    "name":       chunk.name,
                    "content":    chunk.content,
                    "start_line": chunk.start_line,
                    "end_line":   chunk.end_line,
                },
            )
            for chunk, emb in zip(chunks, embeddings)
        ]
        self._client.upsert(collection_name=_COLLECTION, points=points)

    def delete_repo(self, repo_name: str) -> None:
        """Delete all vectors belonging to *repo_name*."""
        self._client.delete(
            collection_name=_COLLECTION,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="repo", match=MatchValue(value=repo_name))]
                )
            ),
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        repo_filter: str | None = None,
        language_filter: str | None = None,
    ) -> List[dict]:
        """Return top-k semantically similar chunks.

        Optionally filter by repo name and/or language.
        """
        must_clauses = []
        if repo_filter:
            must_clauses.append(
                FieldCondition(key="repo", match=MatchValue(value=repo_filter))
            )
        if language_filter:
            must_clauses.append(
                FieldCondition(key="language", match=MatchValue(value=language_filter))
            )

        query_filter = Filter(must=must_clauses) if must_clauses else None

        response = self._client.query_points(
            collection_name=_COLLECTION,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [{"score": h.score, **h.payload} for h in response.points]

    def count(self) -> int:
        """Total number of vectors stored."""
        return self._client.count(collection_name=_COLLECTION).count

    def export_payloads(self) -> list[dict]:
        """Return all stored chunk payloads (no vectors) for persistence."""
        payloads: list[dict] = []
        offset = None
        while True:
            records, offset = self._client.scroll(
                collection_name=_COLLECTION,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for r in records:
                if r.payload:
                    payloads.append(dict(r.payload))
            if offset is None:
                break
        return payloads

    def repos(self) -> List[str]:
        """Return the distinct repo names in the collection."""
        # Qdrant doesn't have a GROUP BY, so we scroll and collect unique values
        seen: set[str] = set()
        offset = None
        while True:
            records, offset = self._client.scroll(
                collection_name=_COLLECTION,
                scroll_filter=None,
                limit=256,
                offset=offset,
                with_payload=["repo"],
            )
            for r in records:
                if r.payload:
                    seen.add(r.payload.get("repo", ""))
            if offset is None:
                break
        seen.discard("")
        return sorted(seen)
