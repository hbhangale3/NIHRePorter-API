from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from .embedding_model import EmbeddingModel


logger = logging.getLogger(__name__)


def _default_embeddings_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "knowledge" / "embeddings"


def semantic_artifacts_exist(*, embeddings_dir: str | Path | None = None) -> bool:
    resolved_dir = Path(embeddings_dir) if embeddings_dir is not None else _default_embeddings_dir()
    return (
        (resolved_dir / "mesh_vectors.faiss").exists()
        and (resolved_dir / "mesh_vector_metadata.json").exists()
    )


@lru_cache(maxsize=None)
def _get_cached_embedding_model(model_name: str) -> EmbeddingModel:
    logger.info("Creating cached embedding model wrapper for %s", model_name)
    return EmbeddingModel(model_name)


def get_embedding_model(model_name: str | None = None) -> EmbeddingModel:
    resolved_model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
    return _get_cached_embedding_model(resolved_model_name)


get_embedding_model.cache_clear = _get_cached_embedding_model.cache_clear  # type: ignore[attr-defined]


@lru_cache(maxsize=None)
def _get_cached_mesh_semantic_retriever(embeddings_dir: str, model_name: str | None):
    from .semantic_retriever import MeshSemanticRetriever

    resolved_dir = Path(embeddings_dir)
    logger.info("Creating cached MeSH semantic retriever from %s", resolved_dir)
    return MeshSemanticRetriever(
        embeddings_dir=resolved_dir,
        embedding_model=get_embedding_model(model_name),
        model_name=model_name,
    )


def get_mesh_semantic_retriever(
    embeddings_dir: str | Path | None = None,
    model_name: str | None = None,
):
    resolved_dir = str(Path(embeddings_dir) if embeddings_dir is not None else _default_embeddings_dir())
    return _get_cached_mesh_semantic_retriever(resolved_dir, model_name)


get_mesh_semantic_retriever.cache_clear = _get_cached_mesh_semantic_retriever.cache_clear  # type: ignore[attr-defined]


def preload_semantic_resources_if_available() -> None:
    if not semantic_artifacts_exist():
        logger.info("Semantic embeddings not found; skipping preload.")
        return

    try:
        retriever = get_mesh_semantic_retriever()
        retriever._ensure_loaded()
        retriever._get_embedding_model()._get_model()
        logger.info("Semantic retriever and embedding model preloaded.")
    except Exception as exc:
        logger.warning("Semantic preload skipped after initialization failure: %s", exc)
