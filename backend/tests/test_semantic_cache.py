from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.semantic.embedding_model import EmbeddingModel
from app.semantic.semantic_cache import get_embedding_model, get_mesh_semantic_retriever


class DummyRetriever:
    def __init__(self, *, embeddings_dir=None, embedding_model=None, model_name=None) -> None:
        self.embeddings_dir = embeddings_dir
        self.embedding_model = embedding_model
        self.model_name = model_name


def test_get_embedding_model_returns_same_cached_instance() -> None:
    get_embedding_model.cache_clear()

    first = get_embedding_model()
    second = get_embedding_model()

    assert isinstance(first, EmbeddingModel)
    assert first is second


def test_get_mesh_semantic_retriever_returns_same_cached_instance(monkeypatch) -> None:
    get_embedding_model.cache_clear()
    get_mesh_semantic_retriever.cache_clear()

    import app.semantic.semantic_retriever as semantic_retriever_module

    monkeypatch.setattr(semantic_retriever_module, "MeshSemanticRetriever", DummyRetriever)

    first = get_mesh_semantic_retriever()
    second = get_mesh_semantic_retriever()

    assert first is second
    assert first.embedding_model is get_embedding_model()
