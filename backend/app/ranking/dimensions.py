from __future__ import annotations

from dataclasses import dataclass
import re

from ..query_intent import QueryIntent, extract_query_intent
from ..utils import normalize_text, unique_preserve_order


DIMENSION_LABELS = {
    "technology_method": "AI / Data Science",
    "disease_condition": "Disease / Condition",
    "population_equity_access": "Population / Equity / Access",
    "care_delivery_context": "Care Delivery / Intervention Context",
    "other": "Query Concepts",
}


@dataclass(slots=True)
class DimensionFamily:
    key: str
    label: str
    terms: list[str]


def build_dimension_families(
    *,
    research_question: str,
    selected_concepts: list[str],
    final_keywords: list[str],
    mesh_terms: list[str],
    semantic_terms: list[str],
    semantic_concepts: list[str],
) -> list[DimensionFamily]:
    intent = extract_query_intent(
        research_question=research_question,
        selected_concepts=selected_concepts,
        final_keywords=final_keywords,
        mesh_terms=mesh_terms,
        semantic_terms=semantic_terms,
        semantic_concepts=semantic_concepts,
    )
    families: list[DimensionFamily] = []
    for key, terms in intent.dimension_items():
        if not terms:
            continue
        families.append(
            DimensionFamily(
                key=key,
                label=DIMENSION_LABELS[key],
                terms=unique_preserve_order(terms),
            )
        )
    return families


def match_family_terms(text: str, family: DimensionFamily) -> list[str]:
    normalized_text = normalize_text(text)
    if not normalized_text:
        return []
    return unique_preserve_order([term for term in family.terms if _contains_phrase(normalized_text, term)])


def dimension_label_for_key(key: str) -> str:
    return DIMENSION_LABELS.get(key, key.replace("_", " ").title())


def requires_technology_match(intent: QueryIntent) -> bool:
    return bool(intent.technology_method)


def build_query_intent(
    *,
    research_question: str,
    selected_concepts: list[str],
    final_keywords: list[str],
    mesh_terms: list[str],
    semantic_terms: list[str],
    semantic_concepts: list[str],
) -> QueryIntent:
    return extract_query_intent(
        research_question=research_question,
        selected_concepts=selected_concepts,
        final_keywords=final_keywords,
        mesh_terms=mesh_terms,
        semantic_terms=semantic_terms,
        semantic_concepts=semantic_concepts,
    )


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return False
    pattern = re.compile(rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)")
    return bool(pattern.search(text))
