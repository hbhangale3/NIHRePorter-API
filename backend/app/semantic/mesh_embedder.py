from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from app.mesh import MeshDescriptor, MeshKnowledgeBase

from .embedding_model import EmbeddingModel
from .semantic_models import MeshVectorMetadata
from .vector_store import FaissVectorStore


@dataclass(slots=True)
class EmbeddingCheckpointState:
    model_name: str
    next_index: int
    total_descriptors: int
    checkpoint_every: int
    batch_size: int
    limit: int | None
    descriptor_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmbeddingCheckpointState":
        return cls(**data)


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
        self.checkpoint_dir = self.embeddings_dir / "checkpoints"
        self.mesh_kb = mesh_kb or MeshKnowledgeBase(processed_dir=self.processed_dir)
        self.embedding_model = embedding_model or EmbeddingModel()
        self.vector_store = vector_store or FaissVectorStore()
        self.text_builder = text_builder or MeshConceptTextBuilder()

    def build(
        self,
        *,
        batch_size: int = 64,
        limit: int | None = None,
        checkpoint_every: int = 25,
        resume: bool = False,
        force: bool = False,
        keep_checkpoints: bool = False,
    ) -> dict[str, object]:
        descriptors = sorted(self.mesh_kb.descriptors.values(), key=lambda item: item.descriptor_ui)
        if limit is not None:
            descriptors = descriptors[:limit]
        if not descriptors:
            raise ValueError("No MeSH descriptors are available to embed.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive.")

        texts = [self.text_builder.build_text(descriptor) for descriptor in descriptors]
        metadata = [self._build_metadata(descriptor, text) for descriptor, text in zip(descriptors, texts, strict=False)]
        descriptor_ids = [descriptor.descriptor_ui for descriptor in descriptors]

        vector_path = self.embeddings_dir / "mesh_vectors.faiss"
        metadata_path = self.embeddings_dir / "mesh_vector_metadata.json"
        config_path = self.embeddings_dir / "embedding_config.json"

        if force and not resume:
            self._clear_checkpoints()
            for artifact_path in (vector_path, metadata_path, config_path):
                if artifact_path.exists():
                    artifact_path.unlink()

        embeddings, start_index = self._load_or_initialize_embeddings(
            metadata=metadata,
            descriptor_ids=descriptor_ids,
            batch_size=batch_size,
            checkpoint_every=checkpoint_every,
            limit=limit,
            resume=resume,
        )

        total_descriptors = len(descriptors)
        total_batches = (total_descriptors + batch_size - 1) // batch_size
        completed_batches = start_index // batch_size

        progress = tqdm(
            total=total_descriptors,
            initial=start_index,
            desc="Embedding MeSH descriptors",
            unit="desc",
        )

        try:
            for batch_index, batch_start in enumerate(range(start_index, total_descriptors, batch_size), start=completed_batches):
                batch_end = min(batch_start + batch_size, total_descriptors)
                batch_vectors = self.embedding_model.embed_texts(texts[batch_start:batch_end], batch_size=batch_size)
                if batch_vectors.shape[0] != (batch_end - batch_start):
                    raise ValueError("Embedding count does not match descriptor batch size.")
                embeddings.append(batch_vectors.astype(np.float32, copy=False))
                progress.update(batch_end - batch_start)
                progress.set_postfix(
                    batch=f"{batch_index + 1}/{total_batches}",
                    refresh=False,
                )

                if (batch_index + 1) % checkpoint_every == 0 or batch_end == total_descriptors:
                    self._save_checkpoint(
                        embeddings=embeddings,
                        metadata=metadata[:batch_end],
                        state=EmbeddingCheckpointState(
                            model_name=self.embedding_model.model_name,
                            next_index=batch_end,
                            total_descriptors=total_descriptors,
                            checkpoint_every=checkpoint_every,
                            batch_size=batch_size,
                            limit=limit,
                            descriptor_ids=descriptor_ids,
                        ),
                    )

                if (batch_index + 1) % max(1, min(checkpoint_every, 5)) == 0 or batch_end == total_descriptors:
                    print(f"[embedding] processed {batch_end} / {total_descriptors} descriptors")
        finally:
            progress.close()

        if not embeddings:
            raise ValueError("No embeddings were generated.")

        vectors = np.vstack(embeddings).astype(np.float32, copy=False)
        if vectors.shape[0] != total_descriptors:
            raise ValueError("Final embedding count does not match descriptor count.")

        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store.build(vectors)
        self.vector_store.save(vector_path)
        self._write_json(
            metadata_path,
            {"vectors": [item.to_dict() for item in metadata]},
        )
        self._write_json(
            config_path,
            {
                "model_name": self.embedding_model.model_name,
                "normalize_embeddings": True,
                "vector_count": len(metadata),
                "embedding_dimension": int(vectors.shape[1]),
            },
        )

        if not keep_checkpoints:
            self._clear_checkpoints()

        return {
            "descriptor_count": len(descriptors),
            "embedding_dimension": int(vectors.shape[1]),
            "model_name": self.embedding_model.model_name,
            "vector_path": vector_path,
            "metadata_path": metadata_path,
            "config_path": config_path,
            "metadata": metadata,
            "checkpoint_dir": self.checkpoint_dir,
        }

    def _load_or_initialize_embeddings(
        self,
        *,
        metadata: list[MeshVectorMetadata],
        descriptor_ids: list[str],
        batch_size: int,
        checkpoint_every: int,
        limit: int | None,
        resume: bool,
    ) -> tuple[list[np.ndarray], int]:
        if not resume:
            return [], 0

        state_path = self.checkpoint_dir / "checkpoint_state.json"
        embeddings_path = self.checkpoint_dir / "checkpoint_embeddings.npy"
        metadata_path = self.checkpoint_dir / "checkpoint_metadata.json"
        if not state_path.exists() or not embeddings_path.exists() or not metadata_path.exists():
            return [], 0

        state = EmbeddingCheckpointState.from_dict(json.loads(state_path.read_text(encoding="utf-8")))
        if state.model_name != self.embedding_model.model_name:
            raise ValueError("Checkpoint model name does not match the requested embedding model.")
        if state.descriptor_ids != descriptor_ids:
            raise ValueError("Checkpoint descriptor set does not match the current build inputs.")
        if state.total_descriptors != len(descriptor_ids):
            raise ValueError("Checkpoint total_descriptors does not match the current build inputs.")
        if state.limit != limit:
            raise ValueError("Checkpoint limit does not match the requested limit.")
        if state.batch_size != batch_size:
            raise ValueError("Checkpoint batch_size does not match the requested batch_size.")
        if state.checkpoint_every != checkpoint_every:
            raise ValueError("Checkpoint checkpoint_every does not match the requested checkpoint_every.")

        checkpoint_metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        checkpoint_metadata = checkpoint_metadata_payload.get("vectors", [])
        if len(checkpoint_metadata) != state.next_index:
            raise ValueError("Checkpoint metadata length does not match checkpoint state.")
        expected_metadata = [item.to_dict() for item in metadata[: state.next_index]]
        if checkpoint_metadata != expected_metadata:
            raise ValueError("Checkpoint metadata does not match current descriptor metadata.")

        saved_embeddings = np.load(embeddings_path)
        if saved_embeddings.shape[0] != state.next_index:
            raise ValueError("Checkpoint embedding rows do not match checkpoint state.")
        return [saved_embeddings.astype(np.float32, copy=False)], state.next_index

    def _save_checkpoint(
        self,
        *,
        embeddings: list[np.ndarray],
        metadata: list[MeshVectorMetadata],
        state: EmbeddingCheckpointState,
    ) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        combined = np.vstack(embeddings).astype(np.float32, copy=False)
        self._write_numpy(self.checkpoint_dir / "checkpoint_embeddings.npy", combined)
        self._write_json(
            self.checkpoint_dir / "checkpoint_metadata.json",
            {"vectors": [item.to_dict() for item in metadata]},
        )
        self._write_json(
            self.checkpoint_dir / "checkpoint_state.json",
            state.to_dict(),
        )

    def _build_metadata(self, descriptor: MeshDescriptor, source_text: str) -> MeshVectorMetadata:
        return MeshVectorMetadata(
            mesh_id=descriptor.descriptor_ui,
            preferred_name=descriptor.preferred_name,
            synonyms=list(descriptor.entry_terms),
            tree_numbers=list(descriptor.tree_numbers),
            scope_note=descriptor.scope_note,
            source_text_preview=source_text[:280],
        )

    def _clear_checkpoints(self) -> None:
        if not self.checkpoint_dir.exists():
            return
        for path in self.checkpoint_dir.glob("*"):
            if path.is_file():
                path.unlink()

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(f"{path.suffix}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(path)

    def _write_numpy(self, path: Path, array: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(f"{path.suffix}.tmp.npy")
        np.save(temp_path, array)
        temp_path.replace(path)
