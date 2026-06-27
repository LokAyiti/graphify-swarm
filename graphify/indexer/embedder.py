"""
embedder.py — generates sentence embeddings using sentence-transformers.

Model: all-MiniLM-L6-v2
  • 22M parameters, 384-dimensional output
  • Downloads ~90 MB on first use, then cached by HuggingFace locally
  • No API key required — fully offline after first download

Embedding cache
---------------
Each unique piece of text is hashed (SHA-256) and its embedding stored as a
small .npy file under <cache_dir>/embeddings/.  Re-indexing an unchanged file
costs zero inference time.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

import numpy as np

_MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    """Lazy-loading sentence-transformer wrapper with on-disk cache."""

    def __init__(self, cache_dir: Path) -> None:
        self._emb_cache = cache_dir / "embeddings"
        self._emb_cache.mkdir(parents=True, exist_ok=True)
        self._model = None  # loaded on first embed() call

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            self._model = SentenceTransformer(_MODEL_NAME)

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        # Shard into 256 subdirs to avoid huge flat directories
        return self._emb_cache / key[:2] / f"{key}.npy"

    def _read_cache(self, key: str) -> List[float] | None:
        p = self._cache_path(key)
        if p.exists():
            try:
                return np.load(str(p)).tolist()
            except Exception:
                p.unlink(missing_ok=True)
        return None

    def _write_cache(self, key: str, vec: np.ndarray) -> None:
        p = self._cache_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), vec)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return one embedding vector per input text.

        Hits the on-disk cache first; only runs model inference for
        texts not yet seen.
        """
        self._load_model()

        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []

        for i, text in enumerate(texts):
            cached = self._read_cache(self._key(text))
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)

        if uncached_indices:
            batch_texts = [texts[i] for i in uncached_indices]
            embeddings: np.ndarray = self._model.encode(  # type: ignore[union-attr]
                batch_texts,
                show_progress_bar=False,
                batch_size=64,
                normalize_embeddings=True,
            )
            for list_pos, original_idx in enumerate(uncached_indices):
                vec = embeddings[list_pos]
                results[original_idx] = vec.tolist()
                self._write_cache(self._key(texts[original_idx]), vec)

        return results  # type: ignore[return-value]

    @property
    def dim(self) -> int:
        """Embedding dimension (384 for all-MiniLM-L6-v2)."""
        return 384
