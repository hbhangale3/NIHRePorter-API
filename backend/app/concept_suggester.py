from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .mesh import MeshKnowledgeBase
from .query_intent import (
    CARE_DELIVERY_CONTEXT_TERMS,
    DISEASE_SYNONYM_MAP,
    POPULATION_EQUITY_ACCESS_TERMS,
    TECHNOLOGY_METHOD_TERMS,
    QueryIntent,
    extract_query_intent,
)
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
    "can",
    "for",
    "from",
    "how",
    "improve",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "to",
    "use",
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
SPECIFICITY_PENALTY_TERMS = {"mutation", "gene", "pathway", "receptor", "protein", "syndrome"}

DIAGNOSTIC_IMAGING_TERMS = [
    "early cancer diagnosis",
    "cancer diagnosis",
    "early diagnosis",
    "early detection",
    "early detection of cancer",
    "medical imaging",
    "diagnostic imaging",
    "image processing",
    "image processing computer-assisted",
    "computer-assisted image processing",
    "radiology",
    "imaging",
]

DOMAIN_TERMS_BY_DIMENSION: dict[str, list[str]] = {
    "technology_method": TECHNOLOGY_METHOD_TERMS + ["artificial intelligence"],
    "disease_condition": unique_preserve_order(
        [
            "precision oncology",
            "oncology",
            "cancer",
            "neoplasms",
            "precision medicine",
        ]
        + [synonym for synonyms in DISEASE_SYNONYM_MAP.values() for synonym in synonyms]
    ),
    "diagnostic_imaging": DIAGNOSTIC_IMAGING_TERMS,
    "population_equity_access": POPULATION_EQUITY_ACCESS_TERMS + ["medically underserved", "underserved populations"],
    "care_delivery_context": CARE_DELIVERY_CONTEXT_TERMS + ["disease management", "patient care"],
}

PHRASE_EXPANSIONS: dict[str, list[tuple[str, str]]] = {
    "ai": [("artificial intelligence", "technology_method")],
    "machine learning": [("artificial intelligence", "technology_method")],
    "precision oncology": [
        ("oncology", "disease_condition"),
        ("cancer", "disease_condition"),
        ("precision medicine", "disease_condition"),
    ],
    "early cancer diagnosis": [
        ("cancer diagnosis", "diagnostic_imaging"),
        ("early detection of cancer", "diagnostic_imaging"),
        ("cancer", "disease_condition"),
    ],
    "medical imaging": [
        ("diagnostic imaging", "diagnostic_imaging"),
        ("radiology", "diagnostic_imaging"),
        ("imaging", "diagnostic_imaging"),
    ],
    "diabetes care": [
        ("diabetes", "disease_condition"),
        ("disease management", "care_delivery_context"),
        ("patient care", "care_delivery_context"),
    ],
    "underserved populations": [
        ("medically underserved", "population_equity_access"),
        ("health disparities", "population_equity_access"),
        ("health equity", "population_equity_access"),
    ],
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
        "Neoplasms",
        "Medical Oncology",
        "Precision Medicine",
    ],
    "diagnostic_imaging": [
        "Diagnostic Imaging",
        "Medical Imaging",
        "Radiology",
        "Early Detection of Cancer",
        "Image Processing, Computer-Assisted",
    ],
    "population_equity_access": [
        "Medically Underserved Area",
        "Healthcare Disparities",
        "Health Equity",
    ],
    "care_delivery_context": [
        "Patient Care",
        "Disease Management",
        "Clinical Decision Support Systems",
        "Telemedicine",
    ],
}

DIMENSION_ORDER = [
    "technology_method",
    "disease_condition",
    "diagnostic_imaging",
    "population_equity_access",
    "care_delivery_context",
    "other",
]


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
        phrase_candidates = self._extract_candidate_phrases(normalized_question, intent=intent)

        semantic_concepts = self._suggest_from_semantic_mesh(
            normalized_question,
            intent=intent,
            phrase_candidates=phrase_candidates,
            top_k=top_k,
        )
        if semantic_concepts:
            return {
                "question": normalized_question,
                "concepts": semantic_concepts[:top_k],
                "fallback_used": False,
                "error": None,
            }

        mesh_lookup_concepts = self._suggest_from_mesh_lookup(
            normalized_question,
            intent=intent,
            phrase_candidates=phrase_candidates,
            top_k=top_k,
        )
        if mesh_lookup_concepts:
            return {
                "question": normalized_question,
                "concepts": mesh_lookup_concepts[:top_k],
                "fallback_used": True,
                "error": None,
            }

        fallback_concepts = self._suggest_from_simple_fallback(
            normalized_question,
            intent=intent,
            phrase_candidates=phrase_candidates,
            top_k=top_k,
        )
        return {
            "question": normalized_question,
            "concepts": fallback_concepts[:top_k],
            "fallback_used": True,
            "error": None,
        }

    def _suggest_from_semantic_mesh(
        self,
        question: str,
        *,
        intent: QueryIntent,
        phrase_candidates: list[dict[str, str]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        try:
            retriever = self.semantic_retriever_factory()
        except Exception:
            return []

        candidates: list[dict[str, Any]] = []
        for phrase_candidate in phrase_candidates:
            phrase = phrase_candidate["phrase"]
            dimension = phrase_candidate["dimension"]
            results = retriever.retrieve(phrase, top_k=max(5, min(8, top_k)))
            for result in results:
                candidate = self._semantic_result_to_candidate(
                    result=result,
                    matched_phrase=phrase,
                    dimension_key=dimension,
                    question=question,
                )
                if candidate is not None:
                    candidates.append(candidate)

        return self._select_balanced_candidates(candidates, intent=intent, top_k=top_k)

    def _suggest_from_mesh_lookup(
        self,
        question: str,
        *,
        intent: QueryIntent,
        phrase_candidates: list[dict[str, str]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        try:
            mesh_kb = self.mesh_kb_factory()
        except Exception:
            return []

        candidates: list[dict[str, Any]] = []
        for phrase_candidate in phrase_candidates:
            phrase = phrase_candidate["phrase"]
            dimension = phrase_candidate["dimension"]
            for descriptor in mesh_kb.lookup_by_term(phrase):
                candidate = self._lookup_descriptor_to_candidate(
                    preferred_name=descriptor.preferred_name,
                    mesh_id=descriptor.descriptor_ui,
                    matched_phrase=phrase,
                    dimension_key=dimension,
                    question=question,
                    source_score=1.0,
                    tree_numbers=getattr(descriptor, "tree_numbers", []),
                )
                if candidate is not None:
                    candidates.append(candidate)

            for result in mesh_kb.search(phrase, limit=min(6, top_k), score_cutoff=55.0):
                candidate = self._lookup_descriptor_to_candidate(
                    preferred_name=result.preferred_name,
                    mesh_id=result.mesh_id,
                    matched_phrase=phrase,
                    dimension_key=dimension,
                    question=question,
                    source_score=round(float(result.score) / 100.0, 4),
                    tree_numbers=[],
                )
                if candidate is not None:
                    candidates.append(candidate)

        return self._select_balanced_candidates(candidates, intent=intent, top_k=top_k)

    def _suggest_from_simple_fallback(
        self,
        question: str,
        *,
        intent: QueryIntent,
        phrase_candidates: list[dict[str, str]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        fallback_terms = [
            {
                "label": item["phrase"],
                "mesh_id": None,
                "source": "fallback",
                "matched_phrase": item["phrase"],
                "score": None,
                "dimension": item["dimension"],
            }
            for item in phrase_candidates
        ]
        if not fallback_terms:
            fallback_terms = [
                {
                    "label": label,
                    "mesh_id": None,
                    "source": "fallback",
                    "matched_phrase": label,
                    "score": None,
                    "dimension": "other",
                }
                for label in unique_preserve_order([term for _key, terms in intent.dimension_items() for term in terms])
            ]
        return self._dedupe_final_concepts(fallback_terms)[:top_k]

    def _extract_candidate_phrases(self, question: str, *, intent: QueryIntent) -> list[dict[str, str]]:
        normalized_question = normalize_text(question)
        if not normalized_question:
            return []

        phrases: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()

        def add_phrase(phrase: str, dimension: str | None = None) -> None:
            normalized_phrase = _normalize_whitespace(phrase)
            lowered = normalize_text(normalized_phrase)
            if not lowered or lowered in STOPWORDS or len(lowered) < 2:
                return
            resolved_dimension = dimension or self._classify_phrase(normalized_phrase, intent=intent)
            key = (lowered, resolved_dimension)
            if key in seen:
                return
            seen.add(key)
            phrases.append({"phrase": normalized_phrase, "dimension": resolved_dimension})

        for dimension_key, terms in intent.dimension_items():
            if dimension_key == "other":
                continue
            for term in terms:
                add_phrase(term, dimension_key)

        for dimension_key, dictionary_terms in DOMAIN_TERMS_BY_DIMENSION.items():
            for term in dictionary_terms:
                if _contains_phrase(normalized_question, term):
                    add_phrase(term, dimension_key)
                    for expanded_phrase, expanded_dimension in PHRASE_EXPANSIONS.get(term, []):
                        add_phrase(expanded_phrase, expanded_dimension)

        for segment in CONNECTOR_PATTERN.split(question):
            normalized_segment = _normalize_whitespace(segment)
            if not normalized_segment:
                continue
            tokens = _meaningful_tokens(normalized_segment)
            if 1 <= len(tokens) <= 5:
                add_phrase(" ".join(tokens))

        tokens = _meaningful_tokens(question)
        for width in (4, 3, 2, 1):
            for start_index in range(len(tokens) - width + 1):
                chunk_tokens = tokens[start_index : start_index + width]
                chunk = " ".join(chunk_tokens)
                if self._is_candidate_chunk(chunk_tokens):
                    add_phrase(chunk)

        return phrases

    def _classify_phrase(self, phrase: str, *, intent: QueryIntent) -> str:
        normalized_phrase = normalize_text(phrase)
        if not normalized_phrase:
            return "other"
        for dimension_key in DIMENSION_ORDER:
            if dimension_key == "other":
                continue
            for term in DOMAIN_TERMS_BY_DIMENSION.get(dimension_key, []):
                if _contains_phrase(normalized_phrase, term) or _contains_phrase(term, normalized_phrase):
                    return dimension_key
        for dimension_key, terms in intent.dimension_items():
            if dimension_key == "other":
                continue
            if any(_contains_phrase(normalized_phrase, term) or _contains_phrase(term, normalized_phrase) for term in terms):
                return dimension_key
        return "other"

    def _is_candidate_chunk(self, chunk_tokens: list[str]) -> bool:
        if not chunk_tokens:
            return False
        normalized_chunk = normalize_text(" ".join(chunk_tokens))
        if len(chunk_tokens) == 1:
            token = normalized_chunk
            return any(_contains_phrase(token, term) or _contains_phrase(term, token) for terms in DOMAIN_TERMS_BY_DIMENSION.values() for term in terms)
        return any(token not in STOPWORDS for token in chunk_tokens)

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

    def _dedupe_ranked_candidates(self, concepts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped_by_label: dict[str, dict[str, Any]] = {}
        for concept in concepts:
            label = _normalize_whitespace(str(concept.get("label") or ""))
            if not label:
                continue
            lowered = label.lower()
            normalized_concept = {**concept, "label": label}
            existing = deduped_by_label.get(lowered)
            current_score = float(normalized_concept.get("rank_score", normalized_concept.get("score") or 0.0))
            existing_score = float(existing.get("rank_score", existing.get("score") or 0.0)) if existing else 0.0
            if existing is None or current_score > existing_score:
                deduped_by_label[lowered] = normalized_concept
        return sorted(
            deduped_by_label.values(),
            key=lambda item: float(item.get("rank_score", item.get("score") or 0.0)),
            reverse=True,
        )

    def _semantic_result_to_candidate(
        self,
        *,
        result: Any,
        matched_phrase: str,
        dimension_key: str,
        question: str,
    ) -> dict[str, Any] | None:
        preferred_name = str(result.preferred_name)
        rank_score = self._concept_rank_score(
            preferred_name=preferred_name,
            matched_phrase=matched_phrase,
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
            "source": self._concept_source(preferred_name, matched_phrase, default_source="semantic_mesh"),
            "matched_phrase": matched_phrase,
            "score": round(float(result.score), 4),
            "rank_score": round(rank_score, 4),
            "dimension": dimension_key,
        }

    def _lookup_descriptor_to_candidate(
        self,
        *,
        preferred_name: str,
        mesh_id: str,
        matched_phrase: str,
        dimension_key: str,
        question: str,
        source_score: float,
        tree_numbers: list[str],
    ) -> dict[str, Any] | None:
        rank_score = self._concept_rank_score(
            preferred_name=preferred_name,
            matched_phrase=matched_phrase,
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
            "source": self._concept_source(preferred_name, matched_phrase, default_source="mesh_lookup"),
            "matched_phrase": matched_phrase,
            "score": round(source_score, 4),
            "rank_score": round(rank_score, 4),
            "dimension": dimension_key,
        }

    def _concept_source(self, preferred_name: str, matched_phrase: str, *, default_source: str) -> str:
        normalized_label = normalize_text(preferred_name)
        normalized_phrase = normalize_text(matched_phrase)
        if normalized_label == normalized_phrase:
            return "exact_phrase"
        if normalized_label and normalized_phrase and (
            normalized_label in normalized_phrase or normalized_phrase in normalized_label
        ):
            return "exact_phrase"
        return default_source

    def _concept_rank_score(
        self,
        *,
        preferred_name: str,
        matched_phrase: str,
        question: str,
        dimension_key: str,
        base_score: float,
        tree_numbers: list[str],
    ) -> float:
        normalized_label = normalize_text(preferred_name)
        normalized_phrase = normalize_text(matched_phrase)
        normalized_question = normalize_text(question)
        score = float(base_score)

        phrase_tokens = set(_meaningful_tokens(matched_phrase))
        label_tokens = set(_meaningful_tokens(preferred_name))
        overlap_count = len(phrase_tokens & label_tokens)

        canonical_labels = {normalize_text(item) for item in CANONICAL_PREFERENCE_BY_DIMENSION.get(dimension_key, [])}
        if normalized_label in canonical_labels:
            score += 0.35
        if normalized_label == normalized_phrase:
            score += 0.45
        elif normalized_label and normalized_phrase and (
            normalized_label in normalized_phrase or normalized_phrase in normalized_label
        ):
            score += 0.25
        if overlap_count:
            score += min(0.3, overlap_count * 0.12)

        if "cancer" in normalized_phrase or "oncology" in normalized_phrase:
            if normalized_label in {"neoplasms", "medical oncology", "precision medicine"}:
                score += 0.3
            if "early detection" in normalized_label:
                score += 0.2
        if "diagnosis" in normalized_phrase or "imaging" in normalized_phrase:
            if normalized_label in {
                "diagnostic imaging",
                "medical imaging",
                "radiology",
                "image processing computer-assisted",
                "image processing, computer-assisted",
                "early detection of cancer",
            }:
                score += 0.3
        if "diabetes" in normalized_phrase and normalized_label == "diabetes mellitus":
            score += 0.3
        if ("underserved" in normalized_phrase or "disparities" in normalized_phrase or "equity" in normalized_phrase) and normalized_label in {
            "medically underserved area",
            "healthcare disparities",
            "health equity",
        }:
            score += 0.3

        if any(term in normalized_label for term in ORG_PENALTY_TERMS):
            score -= 1.05
        if any(term in normalized_label for term in COMPLICATION_PENALTY_TERMS) and not any(
            term in normalized_question for term in COMPLICATION_PENALTY_TERMS
        ):
            score -= 1.05
        if any(term in normalized_label for term in SPECIFICITY_PENALTY_TERMS) and not any(
            term in normalized_question for term in SPECIFICITY_PENALTY_TERMS
        ):
            score -= 0.5
        if tree_numbers:
            shortest_tree = min(len(tree.split(".")) for tree in tree_numbers if tree)
            if shortest_tree <= 2:
                score += 0.15
            elif shortest_tree >= 4:
                score -= 0.25

        return score

    def _select_balanced_candidates(
        self,
        concepts: list[dict[str, Any]],
        *,
        intent: QueryIntent,
        top_k: int,
    ) -> list[dict[str, Any]]:
        deduped = self._dedupe_ranked_candidates(concepts)
        if not deduped:
            return []
        deduped = [concept for concept in deduped if float(concept.get("rank_score", 0.0)) >= 0.45]
        if not deduped:
            return []

        by_dimension: dict[str, list[dict[str, Any]]] = {}
        for concept in deduped:
            by_dimension.setdefault(str(concept.get("dimension") or "other"), []).append(concept)

        selected: list[dict[str, Any]] = []
        seen_labels: set[str] = set()
        phrase_counts: dict[str, int] = {}
        dimension_budgets = self._dimension_budgets(top_k=top_k, available_dimensions=list(by_dimension))

        for dimension in DIMENSION_ORDER:
            queue = by_dimension.get(dimension, [])
            budget = dimension_budgets.get(dimension, 0)
            while queue and budget > 0 and len(selected) < top_k:
                candidate = queue.pop(0)
                lowered = candidate["label"].lower()
                phrase_key = normalize_text(str(candidate.get("matched_phrase") or ""))
                if lowered in seen_labels or phrase_counts.get(phrase_key, 0) >= 2:
                    continue
                seen_labels.add(lowered)
                phrase_counts[phrase_key] = phrase_counts.get(phrase_key, 0) + 1
                selected.append(candidate)
                budget -= 1

        for candidate in deduped:
            if len(selected) >= top_k:
                break
            lowered = candidate["label"].lower()
            phrase_key = normalize_text(str(candidate.get("matched_phrase") or ""))
            if lowered in seen_labels or phrase_counts.get(phrase_key, 0) >= 2:
                continue
            seen_labels.add(lowered)
            phrase_counts[phrase_key] = phrase_counts.get(phrase_key, 0) + 1
            selected.append(candidate)

        return self._dedupe_final_concepts(selected[:top_k])

    def _dimension_budgets(self, *, top_k: int, available_dimensions: list[str]) -> dict[str, int]:
        filtered_dimensions = [dimension for dimension in DIMENSION_ORDER if dimension in available_dimensions and dimension != "other"]
        budgets = {dimension: 0 for dimension in available_dimensions}
        if not filtered_dimensions:
            budgets["other"] = top_k
            return budgets

        minimum_per_dimension = 2 if top_k >= len(filtered_dimensions) * 2 else 1
        for dimension in filtered_dimensions:
            budgets[dimension] = minimum_per_dimension
        while sum(budgets.values()) > top_k:
            for dimension in reversed(filtered_dimensions):
                if budgets[dimension] > 1 and sum(budgets.values()) > top_k:
                    budgets[dimension] -= 1

        while sum(budgets.values()) < top_k:
            for dimension in DIMENSION_ORDER:
                if dimension in budgets and sum(budgets.values()) < top_k:
                    budgets[dimension] += 1
        return budgets

    def _dedupe_final_concepts(self, concepts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: dict[str, dict[str, Any]] = {}
        for concept in concepts:
            label = _normalize_whitespace(str(concept.get("label") or ""))
            if not label:
                continue
            lowered = label.lower()
            existing = deduped.get(lowered)
            current_score = float(concept.get("rank_score", concept.get("score") or 0.0) or 0.0)
            existing_score = float(existing.get("rank_score", existing.get("score") or 0.0) or 0.0) if existing else -1.0
            normalized_concept = {
                "label": label,
                "mesh_id": concept.get("mesh_id"),
                "source": concept.get("source"),
                "matched_phrase": concept.get("matched_phrase"),
                "score": concept.get("score"),
                "dimension": concept.get("dimension"),
            }
            if existing is None or current_score > existing_score:
                deduped[lowered] = normalized_concept
        return list(deduped.values())


def _meaningful_tokens(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_PATTERN.findall(normalize_text(text))
        if token and token.lower() not in STOPWORDS and len(token) > 1
    ]


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = normalize_text(text)
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return False
    pattern = re.compile(rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)")
    return bool(pattern.search(normalized_text))
