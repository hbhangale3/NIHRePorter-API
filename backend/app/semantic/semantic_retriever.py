from __future__ import annotations

import json
from pathlib import Path

from .embedding_model import EmbeddingModel
from .semantic_cache import get_embedding_model
from .semantic_models import MeshVectorMetadata, SemanticMeshResult
from .vector_store import FaissVectorStore


class MeshSemanticRetriever:
    def __init__(
        self,
        *,
        embeddings_dir: str | Path | None = None,
        embedding_model: EmbeddingModel | None = None,
        vector_store: FaissVectorStore | None = None,
        metadata: list[MeshVectorMetadata] | None = None,
        model_name: str | None = None,
    ) -> None:
        backend_dir = Path(__file__).resolve().parents[2]
        self.embeddings_dir = (
            Path(embeddings_dir) if embeddings_dir is not None else backend_dir / "knowledge" / "embeddings"
        )
        self.index_path = self.embeddings_dir / "mesh_vectors.faiss"
        self.metadata_path = self.embeddings_dir / "mesh_vector_metadata.json"
        self.config_path = self.embeddings_dir / "embedding_config.json"
        self.embedding_model = embedding_model
        self.vector_store = vector_store or FaissVectorStore()
        self.metadata = metadata or []
        self._index_loaded = False
        self._config = self._load_config()
        self.model_name = model_name or self._config.get("model_name") or "sentence-transformers/all-MiniLM-L6-v2"

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        min_score: float | None = None,
    ) -> list[SemanticMeshResult]:
        if not query.strip() or top_k <= 0:
            return []

        self._ensure_loaded()
        model = self._get_embedding_model()
        query_vector = model.embed_text(query)
        if query_vector.size == 0:
            return []

        matches = self.vector_store.search(query_vector, top_k=top_k)
        results: list[SemanticMeshResult] = []
        for index, score in matches:
            if min_score is not None and score < min_score:
                continue
            if index < 0 or index >= len(self.metadata):
                continue
            item = self.metadata[index]
            results.append(
                SemanticMeshResult(
                    mesh_id=item.mesh_id,
                    preferred_name=item.preferred_name,
                    score=float(score),
                    synonyms=list(item.synonyms),
                    tree_numbers=list(item.tree_numbers),
                    scope_note=item.scope_note,
                    source_text_preview=item.source_text_preview,
                )
            )
        return results

    def expand_query(
        self,
        query: str,
        top_k: int = 10,
        include_synonyms: bool = True,
        max_terms: int = 30,
        min_score: float | None = None,
    ) -> dict[str, object]:
        results = self.retrieve(query, top_k=top_k, min_score=min_score)

        expanded_terms: list[str] = []
        seen_terms: set[str] = set()
        if max_terms > 0:
            for result in results:
                self._append_term(expanded_terms, seen_terms, result.preferred_name, max_terms=max_terms)
                if include_synonyms:
                    for synonym in result.synonyms:
                        self._append_term(expanded_terms, seen_terms, synonym, max_terms=max_terms)

        return {
            "query": query,
            "semantic_concepts": [
                {
                    "mesh_id": item.mesh_id,
                    "preferred_name": item.preferred_name,
                    "score": item.score,
                    "synonyms": list(item.synonyms),
                    "tree_numbers": list(item.tree_numbers),
                    "scope_note": item.scope_note,
                }
                for item in results
            ],
            "expanded_terms": expanded_terms,
        }

    def _ensure_loaded(self) -> None:
        if not self.metadata:
            self.metadata = self._load_metadata()
        if not self._index_loaded:
            self.vector_store.load(self.index_path)
            self._index_loaded = True

    def _load_config(self) -> dict[str, object]:
        if not self.config_path.exists():
            return {}
        return json.loads(self.config_path.read_text(encoding="utf-8"))

    def _load_metadata(self) -> list[MeshVectorMetadata]:
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Vector metadata not found: {self.metadata_path}")
        payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        items = payload.get("vectors", [])
        return [MeshVectorMetadata.from_dict(item) for item in items]

    def _get_embedding_model(self) -> EmbeddingModel:
        if self.embedding_model is None:
            self.embedding_model = get_embedding_model(self.model_name)
        return self.embedding_model

    def _append_term(
        self,
        terms: list[str],
        seen_terms: set[str],
        term: str,
        *,
        max_terms: int,
    ) -> None:
        normalized = term.strip()
        if not normalized or len(terms) >= max_terms:
            return
        lowered = normalized.lower()
        if lowered in seen_terms:
            return
        seen_terms.add(lowered)
        terms.append(normalized)
