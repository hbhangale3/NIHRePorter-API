from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.mesh.mesh_models import MeshDescriptor
from app.semantic.mesh_embedder import MeshConceptTextBuilder, MeshEmbeddingBuilder
from app.semantic.semantic_models import MeshVectorMetadata
from app.semantic.semantic_retriever import MeshSemanticRetriever
from app.semantic.vector_store import FaissVectorStore


class FakeEmbeddingModel:
    def __init__(self, vector: np.ndarray | None = None, *, fail_on_call: int | None = None) -> None:
        self.vector = vector if vector is not None else np.array([1.0, 0.0], dtype=np.float32)
        self.model_name = "fake-model"
        self.fail_on_call = fail_on_call
        self.call_count = 0

    def embed_text(self, text: str) -> np.ndarray:
        return self.vector

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        self.call_count += 1
        if self.fail_on_call is not None and self.call_count == self.fail_on_call:
            raise RuntimeError("simulated interruption")
        return np.vstack([self.vector for _ in texts]).astype(np.float32, copy=False)


class FakeVectorStore:
    def __init__(self, results: list[tuple[int, float]]) -> None:
        self.results = results
        self.loaded_path: Path | None = None
        self.saved_path: Path | None = None
        self.built_vectors: np.ndarray | None = None

    def load(self, path: Path) -> None:
        self.loaded_path = path

    def build(self, vectors: np.ndarray) -> None:
        self.built_vectors = vectors

    def save(self, path: Path) -> None:
        self.saved_path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-index")

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[tuple[int, float]]:
        return self.results[:top_k]


class FakeMeshKnowledgeBase:
    def __init__(self, descriptors: list[MeshDescriptor]) -> None:
        self.descriptors = {descriptor.descriptor_ui: descriptor for descriptor in descriptors}


def test_concept_text_builder_includes_preferred_name_synonyms_and_scope_note() -> None:
    descriptor = MeshDescriptor(
        descriptor_ui="D000003",
        preferred_name="Telemedicine",
        entry_terms=["Telehealth", "Remote Consultation"],
        tree_numbers=["N04.761"],
        scope_note="Delivery of health services via telecommunications technology.",
        see_related=["Mobile Health"],
        pharmacological_actions=[{"ui": "D999001", "name": "Cardioprotective Agents"}],
        previous_indexing=["HEALTH SERVICES, REMOTE"],
    )

    text = MeshConceptTextBuilder().build_text(descriptor)

    assert "Preferred term: Telemedicine." in text
    assert "Synonyms: Telehealth; Remote Consultation." in text
    assert "Scope note: Delivery of health services via telecommunications technology." in text
    assert "Tree numbers: N04.761." in text


@pytest.mark.skipif(importlib.util.find_spec("faiss") is None, reason="faiss is not installed")
def test_faiss_vector_store_builds_and_searches_tiny_index() -> None:
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.2],
        ],
        dtype=np.float32,
    )
    store = FaissVectorStore()
    store.build(vectors)

    results = store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=2)

    assert len(results) == 2
    assert results[0][0] == 0
    assert results[0][1] >= results[1][1]


def test_semantic_retriever_returns_sorted_results_from_mocked_store() -> None:
    metadata = [
        MeshVectorMetadata(
            mesh_id="D1",
            preferred_name="Artificial Intelligence",
            synonyms=["AI"],
            tree_numbers=["L01"],
            scope_note="Computational systems.",
            source_text_preview="Preferred term: Artificial Intelligence.",
        ),
        MeshVectorMetadata(
            mesh_id="D2",
            preferred_name="Diabetes Mellitus",
            synonyms=["Diabetes"],
            tree_numbers=["C18.452"],
            scope_note="Glucose metabolism disorder.",
            source_text_preview="Preferred term: Diabetes Mellitus.",
        ),
    ]
    retriever = MeshSemanticRetriever(
        embedding_model=FakeEmbeddingModel(),
        vector_store=FakeVectorStore(results=[(1, 0.91), (0, 0.72)]),
        metadata=metadata,
    )

    results = retriever.retrieve("AI for diabetes", top_k=2)

    assert [item.mesh_id for item in results] == ["D2", "D1"]
    assert results[0].score == pytest.approx(0.91)
    assert results[1].score == pytest.approx(0.72)


def test_expand_query_deduplicates_terms_and_respects_max_terms() -> None:
    metadata = [
        MeshVectorMetadata(
            mesh_id="D1",
            preferred_name="Artificial Intelligence",
            synonyms=["AI", "Machine Learning"],
            tree_numbers=["L01"],
            scope_note=None,
            source_text_preview="Preferred term: Artificial Intelligence.",
        ),
        MeshVectorMetadata(
            mesh_id="D2",
            preferred_name="Diabetes Mellitus",
            synonyms=["Diabetes", "AI"],
            tree_numbers=["C18.452"],
            scope_note=None,
            source_text_preview="Preferred term: Diabetes Mellitus.",
        ),
    ]
    retriever = MeshSemanticRetriever(
        embedding_model=FakeEmbeddingModel(),
        vector_store=FakeVectorStore(results=[(0, 0.88), (1, 0.81)]),
        metadata=metadata,
    )

    expansion = retriever.expand_query("AI for diabetes", top_k=2, include_synonyms=True, max_terms=3)

    assert expansion["expanded_terms"] == [
        "Artificial Intelligence",
        "AI",
        "Machine Learning",
    ]
    assert len(expansion["semantic_concepts"]) == 2
    assert expansion["semantic_concepts"][0]["preferred_name"] == "Artificial Intelligence"


def test_retriever_loads_metadata_from_disk_when_needed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        embeddings_dir = Path(temp_dir)
        metadata_path = embeddings_dir / "mesh_vector_metadata.json"
        config_path = embeddings_dir / "embedding_config.json"
        metadata_path.write_text(
            """
{
  "vectors": [
    {
      "mesh_id": "D1",
      "preferred_name": "Telemedicine",
      "synonyms": ["Telehealth"],
      "tree_numbers": ["N04.761"],
      "scope_note": "Remote care",
      "source_text_preview": "Preferred term: Telemedicine."
    }
  ]
}
""".strip(),
            encoding="utf-8",
        )
        config_path.write_text('{"model_name": "fake-model"}', encoding="utf-8")

        retriever = MeshSemanticRetriever(
            embeddings_dir=embeddings_dir,
            embedding_model=FakeEmbeddingModel(),
            vector_store=FakeVectorStore(results=[(0, 0.95)]),
        )

        results = retriever.retrieve("telehealth", top_k=1)

        assert len(results) == 1
        assert results[0].preferred_name == "Telemedicine"
        assert retriever.vector_store.loaded_path == embeddings_dir / "mesh_vectors.faiss"


def test_embedding_builder_checkpoint_resume_recovers_after_interruption() -> None:
    descriptors = [
        MeshDescriptor(descriptor_ui=f"D{i:06d}", preferred_name=f"Concept {i}")
        for i in range(1, 6)
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        embeddings_dir = Path(temp_dir) / "embeddings"
        processed_dir = Path(temp_dir) / "processed"
        mesh_kb = FakeMeshKnowledgeBase(descriptors)

        interrupted_builder = MeshEmbeddingBuilder(
            mesh_kb=mesh_kb,
            processed_dir=processed_dir,
            embeddings_dir=embeddings_dir,
            embedding_model=FakeEmbeddingModel(fail_on_call=2),
            vector_store=FakeVectorStore(results=[]),
        )

        with pytest.raises(RuntimeError, match="simulated interruption"):
            interrupted_builder.build(
                batch_size=2,
                checkpoint_every=1,
                resume=False,
            )

        checkpoint_dir = embeddings_dir / "checkpoints"
        assert (checkpoint_dir / "checkpoint_embeddings.npy").exists()
        assert (checkpoint_dir / "checkpoint_metadata.json").exists()
        assert (checkpoint_dir / "checkpoint_state.json").exists()

        resumed_store = FakeVectorStore(results=[])
        resumed_builder = MeshEmbeddingBuilder(
            mesh_kb=mesh_kb,
            processed_dir=processed_dir,
            embeddings_dir=embeddings_dir,
            embedding_model=FakeEmbeddingModel(),
            vector_store=resumed_store,
        )
        artifacts = resumed_builder.build(
            batch_size=2,
            checkpoint_every=1,
            resume=True,
        )

        assert artifacts["descriptor_count"] == 5
        assert resumed_store.built_vectors is not None
        assert resumed_store.built_vectors.shape == (5, 2)
        assert resumed_store.saved_path == embeddings_dir / "mesh_vectors.faiss"
        assert not (checkpoint_dir / "checkpoint_embeddings.npy").exists()
        assert not (checkpoint_dir / "checkpoint_metadata.json").exists()
        assert not (checkpoint_dir / "checkpoint_state.json").exists()
        metadata_payload = json.loads((embeddings_dir / "mesh_vector_metadata.json").read_text(encoding="utf-8"))
        assert len(metadata_payload["vectors"]) == 5


def test_embedding_builder_keeps_checkpoints_when_requested() -> None:
    descriptors = [
        MeshDescriptor(descriptor_ui="D000001", preferred_name="Concept 1"),
        MeshDescriptor(descriptor_ui="D000002", preferred_name="Concept 2"),
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        embeddings_dir = Path(temp_dir) / "embeddings"
        processed_dir = Path(temp_dir) / "processed"
        builder = MeshEmbeddingBuilder(
            mesh_kb=FakeMeshKnowledgeBase(descriptors),
            processed_dir=processed_dir,
            embeddings_dir=embeddings_dir,
            embedding_model=FakeEmbeddingModel(),
            vector_store=FakeVectorStore(results=[]),
        )

        builder.build(batch_size=1, checkpoint_every=1, keep_checkpoints=True)

        checkpoint_dir = embeddings_dir / "checkpoints"
        assert (checkpoint_dir / "checkpoint_embeddings.npy").exists()
        assert (checkpoint_dir / "checkpoint_metadata.json").exists()
        assert (checkpoint_dir / "checkpoint_state.json").exists()
