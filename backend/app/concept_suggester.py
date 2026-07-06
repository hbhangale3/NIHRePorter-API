from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .mesh import MeshKnowledgeBase
from .semantic import MeshSemanticRetriever


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "to",
    "using",
    "via",
    "what",
    "when",
    "where",
    "with",
}

CONNECTOR_PATTERN = re.compile(
    r"\b(?:and|or|for|with|without|in|on|to|of|the|a|an|using|via|through|among|across)\b",
    re.IGNORECASE,
)
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9/-]*")


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.strip().split())


class ConceptSuggester:
    def __init__(
        self,
        *,
        semantic_retriever_factory: Callable[[], MeshSemanticRetriever] | None = None,
        mesh_kb_factory: Callable[[], MeshKnowledgeBase] | None = None,
    ) -> None:
        self.semantic_retriever_factory = semantic_retriever_factory or self._default_semantic_retriever_factory
        self.mesh_kb_factory = mesh_kb_factory or self._default_mesh_kb_factory

    def suggest(self, question: str, top_k: int = 8) -> dict[str, Any]:
        normalized_question = _normalize_whitespace(question)
        if not normalized_question or top_k <= 0:
            return {
                "question": normalized_question,
                "concepts": [],
                "fallback_used": False,
                "error": None,
            }

        semantic_concepts = self._suggest_from_semantic_mesh(normalized_question, top_k=top_k)
        if semantic_concepts:
            return {
                "question": normalized_question,
                "concepts": semantic_concepts[:top_k],
                "fallback_used": False,
                "error": None,
            }

        mesh_lookup_concepts = self._suggest_from_mesh_lookup(normalized_question, top_k=top_k)
        if mesh_lookup_concepts:
            return {
                "question": normalized_question,
                "concepts": mesh_lookup_concepts[:top_k],
                "fallback_used": True,
                "error": None,
            }

        fallback_concepts = self._suggest_from_simple_fallback(normalized_question, top_k=top_k)
        return {
            "question": normalized_question,
            "concepts": fallback_concepts[:top_k],
            "fallback_used": True,
            "error": None,
        }

    def _suggest_from_semantic_mesh(self, question: str, *, top_k: int) -> list[dict[str, Any]]:
        try:
            retriever = self.semantic_retriever_factory()
            results = retriever.retrieve(question, top_k=top_k)
        except Exception:
            return []

        concepts: list[dict[str, Any]] = []
        for result in results:
            concepts.append(
                {
                    "label": result.preferred_name,
                    "mesh_id": result.mesh_id,
                    "source": "semantic_mesh",
                    "score": round(float(result.score), 4),
                }
            )
        return self._dedupe_concepts(concepts)[:top_k]

    def _suggest_from_mesh_lookup(self, question: str, *, top_k: int) -> list[dict[str, Any]]:
        try:
            mesh_kb = self.mesh_kb_factory()
        except Exception:
            return []

        concepts: list[dict[str, Any]] = []
        for phrase in self._extract_candidate_phrases(question):
            for descriptor in mesh_kb.lookup_by_term(phrase):
                concepts.append(
                    {
                        "label": descriptor.preferred_name,
                        "mesh_id": descriptor.descriptor_ui,
                        "source": "mesh_lookup",
                        "score": 1.0,
                    }
                )

            for result in mesh_kb.search(phrase, limit=min(5, top_k), score_cutoff=55.0):
                concepts.append(
                    {
                        "label": result.preferred_name,
                        "mesh_id": result.mesh_id,
                        "source": "mesh_lookup",
                        "score": round(float(result.score) / 100.0, 4),
                    }
                )

            deduped = self._dedupe_concepts(concepts)
            if len(deduped) >= top_k:
                return deduped[:top_k]

        return self._dedupe_concepts(concepts)[:top_k]

    def _suggest_from_simple_fallback(self, question: str, *, top_k: int) -> list[dict[str, Any]]:
        concepts = [
            {
                "label": label,
                "mesh_id": None,
                "source": "fallback",
                "score": None,
            }
            for label in self._extract_candidate_phrases(question)
        ]
        return self._dedupe_concepts(concepts)[:top_k]

    def _extract_candidate_phrases(self, question: str) -> list[str]:
        cleaned = _normalize_whitespace(re.sub(r"[^\w\s/-]", " ", question))
        if not cleaned:
            return []

        phrases: list[str] = [cleaned]
        for segment in CONNECTOR_PATTERN.split(cleaned):
            normalized = _normalize_whitespace(segment)
            if normalized:
                phrases.append(normalized)

        tokens = TOKEN_PATTERN.findall(cleaned)
        meaningful_tokens = [token for token in tokens if token.lower() not in STOPWORDS and len(token) > 2]
        for start_index in range(len(meaningful_tokens)):
            for width in (1, 2, 3):
                chunk = meaningful_tokens[start_index : start_index + width]
                if len(chunk) != width:
                    continue
                phrases.append(" ".join(chunk))

        deduped: list[str] = []
        seen: set[str] = set()
        for phrase in phrases:
            normalized = _normalize_whitespace(phrase)
            lowered = normalized.lower()
            if not normalized or lowered in seen:
                continue
            if lowered in STOPWORDS or len(normalized) < 3:
                continue
            seen.add(lowered)
            deduped.append(normalized)
        return deduped

    def _default_semantic_retriever_factory(self) -> MeshSemanticRetriever:
        backend_dir = Path(__file__).resolve().parents[1]
        embeddings_dir = backend_dir / "knowledge" / "embeddings"
        vector_path = embeddings_dir / "mesh_vectors.faiss"
        metadata_path = embeddings_dir / "mesh_vector_metadata.json"
        if not vector_path.exists() or not metadata_path.exists():
            raise FileNotFoundError("Semantic MeSH index is not available.")
        return MeshSemanticRetriever(embeddings_dir=embeddings_dir)

    def _default_mesh_kb_factory(self) -> MeshKnowledgeBase:
        backend_dir = Path(__file__).resolve().parents[1]
        processed_dir = backend_dir / "knowledge" / "processed"
        descriptor_path = processed_dir / "mesh_descriptors.json"
        graph_path = processed_dir / "mesh_graph.json"
        lookup_path = processed_dir / "mesh_lookup.pkl"
        if not descriptor_path.exists() or not graph_path.exists() or not lookup_path.exists():
            raise FileNotFoundError("Processed MeSH knowledge base is not available.")
        kb = MeshKnowledgeBase(processed_dir=processed_dir, auto_build=False)
        kb._load_processed_files(descriptor_path, graph_path, lookup_path)
        return kb

    def _dedupe_concepts(self, concepts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for concept in concepts:
            label = _normalize_whitespace(str(concept.get("label") or ""))
            if not label:
                continue
            lowered = label.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            deduped.append({**concept, "label": label})
        return deduped
