from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any

from ..models import MultiQueryRetrievalConfig
from ..query_intent import QueryIntent, extract_query_intent
from ..utils import normalize_text, unique_preserve_order


@dataclass(slots=True)
class QueryPlan:
    query_id: str
    search_terms: list[str]
    operator: str
    reason: str
    covered_dimensions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_query_plans(
    *,
    research_question: str,
    selected_concepts: list[str],
    final_keywords: list[str],
    mesh_terms: list[str],
    semantic_terms: list[str],
    semantic_concepts: list[str],
    retrieval_config: MultiQueryRetrievalConfig,
) -> tuple[QueryIntent, list[QueryPlan]]:
    intent = extract_query_intent(
        research_question=research_question,
        selected_concepts=selected_concepts,
        final_keywords=final_keywords,
        mesh_terms=mesh_terms,
        semantic_terms=semantic_terms,
        semantic_concepts=semantic_concepts,
    )

    dimensions = {
        key: _preferred_terms(terms)
        for key, terms in intent.dimension_items()
        if terms and key != "other"
    }

    plans: list[QueryPlan] = []
    seen_term_sets: set[tuple[str, ...]] = set()
    query_index = 1

    dimension_keys = list(dimensions.keys())
    max_targeted_queries = retrieval_config.max_queries - (1 if retrieval_config.include_original_query else 0)

    for left_key, right_key in combinations(dimension_keys, 2):
        if retrieval_config.require_dimension_overlap and left_key == right_key:
            continue
        left_terms = dimensions[left_key][:2]
        right_terms = dimensions[right_key][:2]
        for left_term in left_terms:
            for right_term in right_terms:
                term_key = tuple(sorted({normalize_text(left_term), normalize_text(right_term)}))
                if len(term_key) < 2 or term_key in seen_term_sets:
                    continue
                seen_term_sets.add(term_key)
                plans.append(
                    QueryPlan(
                        query_id=f"mq-{query_index}",
                        search_terms=[left_term, right_term],
                        operator="and",
                        reason=f"Cross-dimension query for {left_key} + {right_key}",
                        covered_dimensions=[left_key, right_key],
                    )
                )
                query_index += 1
                if len(plans) >= max_targeted_queries:
                    break
            if len(plans) >= max_targeted_queries:
                break
        if len(plans) >= max_targeted_queries:
            break

    if retrieval_config.include_original_query:
        original_terms = unique_preserve_order(final_keywords)[: min(6, len(final_keywords))]
        if original_terms:
            plans.insert(
                0,
                QueryPlan(
                    query_id="mq-original",
                    search_terms=original_terms,
                    operator="or",
                    reason="Original broad query for recall fallback",
                    covered_dimensions=list(dimensions.keys()),
                ),
            )

    return intent, plans[: retrieval_config.max_queries]


def _preferred_terms(terms: list[str]) -> list[str]:
    if not terms:
        return []
    sorted_terms = sorted(
        unique_preserve_order(terms),
        key=lambda term: (0 if " " in term.strip() else 1, len(term)),
    )
    return sorted_terms
