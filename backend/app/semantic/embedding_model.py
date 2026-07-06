from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors.astype(np.float32, copy=False)

    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (matrix / norms).astype(np.float32, copy=False)


class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: Any | None = None

    def embed_text(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.empty((0,), dtype=np.float32)

        vectors = self.embed_texts([text], batch_size=1)
        if vectors.shape[0] == 0:
            return np.empty((0,), dtype=np.float32)
        return vectors[0]

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        cleaned_texts = [text.strip() for text in texts if text.strip()]
        if not cleaned_texts:
            return np.empty((0, 0), dtype=np.float32)

        model = self._get_model()
        vectors = model.encode(
            cleaned_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return _normalize_rows(np.asarray(vectors, dtype=np.float32))

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required to build or query MeSH embeddings."
                ) from exc
            model_source = Path(self.model_name)
            kwargs: dict[str, object] = {}
            if not model_source.exists():
                kwargs["local_files_only"] = False
            try:
                logger.info("Loading embedding model %s...", self.model_name)
                self._model = SentenceTransformer(self.model_name, **kwargs)
            except Exception as exc:
                raise RuntimeError(
                    "Unable to load embedding model "
                    f"'{self.model_name}'. Provide a local model path or use a cached "
                    "sentence-transformers model in this environment."
                ) from exc
        return self._model
