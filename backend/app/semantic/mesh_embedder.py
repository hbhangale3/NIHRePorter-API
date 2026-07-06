from __future__ import annotations

import json
from pathlib import Path

from app.mesh import MeshDescriptor, MeshKnowledgeBase

from .embedding_model import EmbeddingModel
from .semantic_models import MeshVectorMetadata
from .vector_store import FaissVectorStore


class MeshConceptTextBuilder:
    def build_text(self, descriptor: MeshDescriptor) -> str:
        parts = [f"Preferred term: {descriptor.preferred_name}."]

        if descriptor.entry_terms:
            parts.append(f"Synonyms: {'; '.join(descriptor.entry_terms)}.")
        if descriptor.scope_note:
            parts.append(f"Scope note: {descriptor.scope_note}")
        if descriptor.tree_numbers:
            parts.append(f"Tree numbers: {'; '.join(descriptor.tree_numbers)}.")
        if descriptor.see_related:
            parts.append(f"Related terms: {'; '.join(descriptor.see_related)}.")
        if descriptor.pharmacological_actions:
            action_names = [item["name"] for item in descriptor.pharmacological_actions if item.get("name")]
            if action_names:
                parts.append(f"Pharmacological actions: {'; '.join(action_names)}.")
        if descriptor.previous_indexing:
            parts.append(f"Previous indexing: {'; '.join(descriptor.previous_indexing)}.")

        return " ".join(parts)


class MeshEmbeddingBuilder:
    def __init__(
        self,
        *,
        mesh_kb: MeshKnowledgeBase | None = None,
        embedding_model: EmbeddingModel | None = None,
        vector_store: FaissVectorStore | None = None,
        processed_dir: str | Path | None = None,
        embeddings_dir: str | Path | None = None,
        text_builder: MeshConceptTextBuilder | None = None,
    ) -> None:
        backend_dir = Path(__file__).resolve().parents[2]
        self.processed_dir = (
            Path(processed_dir) if processed_dir is not None else backend_dir / "knowledge" / "processed"
        )
        self.embeddings_dir = (
            Path(embeddings_dir) if embeddings_dir is not None else backend_dir / "knowledge" / "embeddings"
        )
        self.mesh_kb = mesh_kb or MeshKnowledgeBase(processed_dir=self.processed_dir)
        self.embedding_model = embedding_model or EmbeddingModel()
        self.vector_store = vector_store or FaissVectorStore()
        self.text_builder = text_builder or MeshConceptTextBuilder()

    def build(self, *, batch_size: int = 64, limit: int | None = None) -> dict[str, object]:
        descriptors = sorted(self.mesh_kb.descriptors.values(), key=lambda item: item.descriptor_ui)
        if limit is not None:
            descriptors = descriptors[:limit]
        if not descriptors:
            raise ValueError("No MeSH descriptors are available to embed.")

        texts = [self.text_builder.build_text(descriptor) for descriptor in descriptors]
        vectors = self.embedding_model.embed_texts(texts, batch_size=batch_size)
        if vectors.shape[0] != len(descriptors):
            raise ValueError("Embedding count does not match descriptor count.")

        self.vector_store.build(vectors)
        metadata = [self._build_metadata(descriptor, text) for descriptor, text in zip(descriptors, texts, strict=False)]

        vector_path = self.embeddings_dir / "mesh_vectors.faiss"
        metadata_path = self.embeddings_dir / "mesh_vector_metadata.json"
        config_path = self.embeddings_dir / "embedding_config.json"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        self.vector_store.save(vector_path)
        metadata_path.write_text(
            json.dumps(
                {
                    "vectors": [item.to_dict() for item in metadata],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        config_path.write_text(
            json.dumps(
                {
                    "model_name": self.embedding_model.model_name,
                    "normalize_embeddings": True,
                    "vector_count": len(metadata),
                    "embedding_dimension": int(vectors.shape[1]),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        return {
            "descriptor_count": len(descriptors),
            "embedding_dimension": int(vectors.shape[1]),
            "model_name": self.embedding_model.model_name,
            "vector_path": vector_path,
            "metadata_path": metadata_path,
            "config_path": config_path,
            "metadata": metadata,
        }

    def _build_metadata(self, descriptor: MeshDescriptor, source_text: str) -> MeshVectorMetadata:
        return MeshVectorMetadata(
            mesh_id=descriptor.descriptor_ui,
            preferred_name=descriptor.preferred_name,
            synonyms=list(descriptor.entry_terms),
            tree_numbers=list(descriptor.tree_numbers),
            scope_note=descriptor.scope_note,
            source_text_preview=source_text[:280],
        )

