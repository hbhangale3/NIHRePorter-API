from __future__ import annotations

from pathlib import Path

import numpy as np


class FaissVectorStore:
    def __init__(self) -> None:
        self.index = None
        self.dimension = 0

    def build(self, vectors: np.ndarray) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("Vectors must be a non-empty 2D float32 array.")

        faiss = self._require_faiss()
        self.dimension = int(matrix.shape[1])
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(matrix)

    def save(self, path: Path) -> None:
        if self.index is None:
            raise ValueError("FAISS index has not been built.")
        faiss = self._require_faiss()
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    def load(self, path: Path) -> None:
        faiss = self._require_faiss()
        if not path.exists():
            raise FileNotFoundError(f"FAISS index not found: {path}")
        self.index = faiss.read_index(str(path))
        self.dimension = int(self.index.d)

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[tuple[int, float]]:
        if self.index is None:
            raise ValueError("FAISS index has not been loaded.")
        if top_k <= 0:
            return []

        vector = np.asarray(query_vector, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        if vector.ndim != 2 or vector.shape[0] == 0 or vector.shape[1] == 0:
            return []

        scores, indices = self.index.search(vector, top_k)
        results: list[tuple[int, float]] = []
        for index, score in zip(indices[0], scores[0], strict=False):
            if index < 0:
                continue
            results.append((int(index), float(score)))
        return results

    def _require_faiss(self):  # type: ignore[no-untyped-def]
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("faiss is required to build or query MeSH vector indexes.") from exc
        return faiss

