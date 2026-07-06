from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .mesh import MeshKnowledgeBase
from .query_intent import QueryIntent, extract_query_intent
from .semantic import MeshSemanticRetriever, get_mesh_semantic_retriever
from .utils import normalize_text, unique_preserve_order


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
COMPLICATION_PENALTY_TERMS = {"coma", "foot", "insipidus", "retinopathy", "neuropathy", "nephropathy"}
ORG_PENALTY_TERMS = {
    "national institute",
    "national institutes",
    "niddk",
    "nih",
    "institute",
}
CANONICAL_PREFERENCE_BY_DIMENSION = {
    "technology_method": [
        "Artificial Intelligence",
        "Machine Learning",
        "Medical Informatics",
        "Clinical Decision Support Systems",
        "Decision Support Systems, Clinical",
    ],
    "disease_condition": [
        "Diabetes Mellitus",
    ],
    "population_equity_access": [
        "Medically Underserved Area",
        "Healthcare Disparities",
        "Health Equity",
    ],
    "care_delivery_context": [
        "Telemedicine",
        "Digital Health",
        "Patient Care",
    ],
}


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

        intent = extract_query_intent(research_question=normalized_question)

        semantic_concepts = self._suggest_from_semantic_mesh(normalized_question, intent=intent, top_k=top_k)
        if semantic_concepts:
            return {
                "question": normalized_question,
                "concepts": semantic_concepts[:top_k],
                "fallback_used": False,
                "error": None,
            }

        mesh_lookup_concepts = self._suggest_from_mesh_lookup(normalized_question, intent=intent, top_k=top_k)
        if mesh_lookup_concepts:
            return {
                "question": normalized_question,
                "concepts": mesh_lookup_concepts[:top_k],
                "fallback_used": True,
                "error": None,
            }

        fallback_concepts = self._suggest_from_simple_fallback(normalized_question, intent=intent, top_k=top_k)
        return {
            "question": normalized_question,
            "concepts": fallback_concepts[:top_k],
            "fallback_used": True,
            "error": None,
        }

    def _suggest_from_semantic_mesh(self, question: str, *, intent: QueryIntent, top_k: int) -> list[dict[str, Any]]:
        try:
            retriever = self.semantic_retriever_factory()
        except Exception:
            return []

        candidates: list[dict[str, Any]] = []
        budgets = self._dimension_budgets(intent, top_k)
        for dimension_key, dimension_queries in self._dimension_queries(intent).items():
            budget = budgets.get(dimension_key, 0)
            if budget <= 0:
                continue
            for query_text in dimension_queries[:2]:
                results = retriever.retrieve(query_text, top_k=max(4, budget * 2))
                for result in results:
                    candidate = self._semantic_result_to_candidate(
                        result=result,
                        dimension_key=dimension_key,
                        question=question,
                    )
                    if candidate is not None:
                        candidates.append(candidate)

        return self._select_balanced_candidates(candidates, top_k=top_k)

    def _suggest_from_mesh_lookup(self, question: str, *, intent: QueryIntent, top_k: int) -> list[dict[str, Any]]:
        try:
            mesh_kb = self.mesh_kb_factory()
        except Exception:
            return []

        candidates: list[dict[str, Any]] = []
        for dimension_key, dimension_queries in self._dimension_queries(intent).items():
            for phrase in dimension_queries[:3]:
                for descriptor in mesh_kb.lookup_by_term(phrase):
                    candidate = self._lookup_descriptor_to_candidate(
                        preferred_name=descriptor.preferred_name,
                        mesh_id=descriptor.descriptor_ui,
                        dimension_key=dimension_key,
                        question=question,
                        source_score=1.0,
                        tree_numbers=getattr(descriptor, "tree_numbers", []),
                    )
                    if candidate is not None:
                        candidates.append(candidate)

                for result in mesh_kb.search(phrase, limit=min(5, top_k), score_cutoff=55.0):
                    candidate = self._lookup_descriptor_to_candidate(
                        preferred_name=result.preferred_name,
                        mesh_id=result.mesh_id,
                        dimension_key=dimension_key,
                        question=question,
                        source_score=round(float(result.score) / 100.0, 4),
                        tree_numbers=[],
                    )
                    if candidate is not None:
                        candidates.append(candidate)

        return self._select_balanced_candidates(candidates, top_k=top_k)

    def _suggest_from_simple_fallback(self, question: str, *, intent: QueryIntent, top_k: int) -> list[dict[str, Any]]:
        fallback_terms = []
        for _dimension_key, terms in intent.dimension_items():
            fallback_terms.extend(terms)
        if not fallback_terms:
            fallback_terms = self._extract_candidate_phrases(question)
        concepts = [
            {
                "label": label,
                "mesh_id": None,
                "source": "fallback",
                "score": None,
            }
            for label in unique_preserve_order(fallback_terms)
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
        return get_mesh_semantic_retriever()

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
        deduped_by_label: dict[str, dict[str, Any]] = {}
        for concept in concepts:
            label = _normalize_whitespace(str(concept.get("label") or ""))
            if not label:
                continue
            lowered = label.lower()
            normalized_concept = {**concept, "label": label}
            existing = deduped_by_label.get(lowered)
            if existing is None or float(normalized_concept.get("rank_score", normalized_concept.get("score") or 0.0)) > float(
                existing.get("rank_score", existing.get("score") or 0.0)
            ):
                deduped_by_label[lowered] = normalized_concept
        return sorted(
            deduped_by_label.values(),
            key=lambda item: float(item.get("rank_score", item.get("score") or 0.0)),
            reverse=True,
        )

    def _dimension_queries(self, intent: QueryIntent) -> dict[str, list[str]]:
        queries: dict[str, list[str]] = {}
        for dimension_key, terms in intent.dimension_items():
            if not terms or dimension_key == "other":
                continue
            queries[dimension_key] = unique_preserve_order([term for term in terms if len(normalize_text(term)) >= 2])
        return queries

    def _dimension_budgets(self, intent: QueryIntent, top_k: int) -> dict[str, int]:
        non_empty = [key for key, terms in intent.dimension_items() if terms and key != "other"]
        if not non_empty:
            return {"other": top_k}

        base = max(1, top_k // len(non_empty))
        budgets = {key: base for key in non_empty}
        remaining = top_k - sum(budgets.values())
        preferred_order = [
            "technology_method",
            "disease_condition",
            "population_equity_access",
            "care_delivery_context",
        ]
        while remaining > 0:
            for key in preferred_order:
                if key in budgets and remaining > 0:
                    budgets[key] += 1
                    remaining -= 1
        return budgets

    def _semantic_result_to_candidate(
        self,
        *,
        result: Any,
        dimension_key: str,
        question: str,
    ) -> dict[str, Any] | None:
        preferred_name = str(result.preferred_name)
        rank_score = self._concept_rank_score(
            preferred_name=preferred_name,
            question=question,
            dimension_key=dimension_key,
            base_score=float(result.score),
            tree_numbers=list(getattr(result, "tree_numbers", []) or []),
        )
        if rank_score <= 0:
            return None
        return {
            "label": preferred_name,
            "mesh_id": result.mesh_id,
            "source": "semantic_mesh",
            "score": round(float(result.score), 4),
            "rank_score": round(rank_score, 4),
            "dimension": dimension_key,
        }

    def _lookup_descriptor_to_candidate(
        self,
        *,
        preferred_name: str,
        mesh_id: str,
        dimension_key: str,
        question: str,
        source_score: float,
        tree_numbers: list[str],
    ) -> dict[str, Any] | None:
        rank_score = self._concept_rank_score(
            preferred_name=preferred_name,
            question=question,
            dimension_key=dimension_key,
            base_score=source_score,
            tree_numbers=tree_numbers,
        )
        if rank_score <= 0:
            return None
        return {
            "label": preferred_name,
            "mesh_id": mesh_id,
            "source": "mesh_lookup",
            "score": round(source_score, 4),
            "rank_score": round(rank_score, 4),
            "dimension": dimension_key,
        }

    def _concept_rank_score(
        self,
        *,
        preferred_name: str,
        question: str,
        dimension_key: str,
        base_score: float,
        tree_numbers: list[str],
    ) -> float:
        normalized_label = normalize_text(preferred_name)
        normalized_question = normalize_text(question)
        score = float(base_score)

        if normalized_label in {normalize_text(item) for item in CANONICAL_PREFERENCE_BY_DIMENSION.get(dimension_key, [])}:
            score += 0.35
        if normalized_label and re.search(rf"(?<!\w){re.escape(normalized_label)}(?!\w)", normalized_question):
            score += 0.25
        if any(term in normalized_label for term in ORG_PENALTY_TERMS):
            score -= 1.05
        if any(term in normalized_label for term in COMPLICATION_PENALTY_TERMS) and not any(
            term in normalized_question for term in COMPLICATION_PENALTY_TERMS
        ):
            score -= 1.05
        if tree_numbers:
            shortest_tree = min(len(tree.split(".")) for tree in tree_numbers if tree)
            if shortest_tree <= 2:
                score += 0.15
            elif shortest_tree >= 3:
                score -= 0.25
        return score

    def _select_balanced_candidates(self, concepts: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
        deduped = self._dedupe_concepts(concepts)
        if not deduped:
            return []
        deduped = [concept for concept in deduped if float(concept.get("rank_score", 0.0)) >= 0.45]
        if not deduped:
            return []

        by_dimension: dict[str, list[dict[str, Any]]] = {}
        for concept in deduped:
            dimension = str(concept.get("dimension") or "other")
            by_dimension.setdefault(dimension, []).append(concept)

        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        while len(selected) < top_k:
            added_this_round = False
            for dimension in [
                "technology_method",
                "disease_condition",
                "population_equity_access",
                "care_delivery_context",
                "other",
            ]:
                queue = by_dimension.get(dimension, [])
                while queue:
                    candidate = queue.pop(0)
                    lowered = candidate["label"].lower()
                    if lowered in seen:
                        continue
                    seen.add(lowered)
                    selected.append(candidate)
                    added_this_round = True
                    break
                if len(selected) >= top_k:
                    break
            if not added_this_round:
                break

        return [
            {
                "label": concept["label"],
                "mesh_id": concept.get("mesh_id"),
                "source": concept.get("source"),
                "score": concept.get("score"),
            }
            for concept in selected[:top_k]
        ]
