from __future__ import annotations

from typing import Any

from ..models import PIOutreachRow
from ..query_intent import extract_query_intent
from ..utils import normalize_text, unique_preserve_order
from .breakdown import breakdown_text, build_breakdown_entries


def apply_explanations(
    rows: list[PIOutreachRow],
    *,
    research_question: str,
    selected_concepts: list[str],
    final_keywords: list[str],
    mesh_terms: list[str],
    semantic_terms: list[str],
    semantic_concepts: list[str],
    retrieval_trace: dict[str, Any],
) -> list[PIOutreachRow]:
    intent = extract_query_intent(
        research_question=research_question,
        selected_concepts=selected_concepts,
        final_keywords=final_keywords,
        mesh_terms=mesh_terms,
        semantic_terms=semantic_terms,
        semantic_concepts=semantic_concepts,
    )
    expected_concepts = _ordered_expected_concepts(intent)
    query_plan_map = {
        str(plan.get("query_id")): str(plan.get("reason") or "")
        for plan in retrieval_trace.get("query_plans", [])
        if isinstance(plan, dict) and plan.get("query_id")
    }

    explained_rows: list[PIOutreachRow] = []
    for row in rows:
        ordered_matched = _ordered_matched_concepts(row.matched_concepts, expected_concepts)
        missing_concepts = _missing_concepts(expected_concepts, ordered_matched)
        breakdown_entries = build_breakdown_entries(row.score_breakdown)
        retrieved_by = _retrieved_by(row, query_plan_map, retrieval_trace)
        explained_rows.append(
            row.model_copy(
                update={
                    "matched_concepts": ordered_matched,
                    "missing_concepts": missing_concepts,
                    "score_breakdown_entries": [entry.to_dict() for entry in breakdown_entries],
                    "score_breakdown_text": breakdown_text(breakdown_entries, row.relevance_score),
                    "retrieved_by": retrieved_by,
                }
            )
        )
    return explained_rows


def _ordered_expected_concepts(intent: Any) -> list[str]:
    concepts: list[str] = []
    for _dimension_key, terms in intent.dimension_items():
        concepts.extend(terms)
    return unique_preserve_order(concepts)


def _ordered_matched_concepts(matched_concepts: list[str], expected_concepts: list[str]) -> list[str]:
    expected_index = {normalize_text(term): idx for idx, term in enumerate(expected_concepts)}
    deduped = unique_preserve_order(matched_concepts)
    return sorted(
        deduped,
        key=lambda term: (
            expected_index.get(normalize_text(term), 10_000),
            len(term),
            term.lower(),
        ),
    )


def _missing_concepts(expected_concepts: list[str], matched_concepts: list[str]) -> list[str]:
    matched_set = {normalize_text(term) for term in matched_concepts if normalize_text(term)}
    missing = [term for term in expected_concepts if normalize_text(term) not in matched_set]
    return unique_preserve_order(missing)[:8]


def _retrieved_by(
    row: PIOutreachRow,
    query_plan_map: dict[str, str],
    retrieval_trace: dict[str, Any],
) -> list[str]:
    if not retrieval_trace.get("multi_query_enabled"):
        return ["Retrieved by Original Query"]

    items: list[str] = []
    for query_id in row.retrieval_query_matches:
        reason = query_plan_map.get(query_id, "")
        if reason:
            items.append(f"{query_id}: {reason}")
        else:
            items.append(query_id)
    return unique_preserve_order(items) or ["Retrieved by Original Query"]
