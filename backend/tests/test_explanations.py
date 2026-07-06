from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models import PIOutreachRow
from app.ranking.breakdown import breakdown_total, build_breakdown_entries
from app.ranking.explanation import apply_explanations


def _row() -> PIOutreachRow:
    return PIOutreachRow(
        pi_name="Alex Strong",
        project_numbers=["P1"],
        relevance_score=76,
        matched_concepts=[
            "Health Disparities",
            "Diabetes Mellitus",
            "Artificial Intelligence",
            "Machine Learning",
        ],
        matched_dimensions=["AI / Data Science", "Disease / Condition", "Population / Equity / Access"],
        missing_dimensions=["Care Delivery / Intervention Context"],
        reasoning="Moderate match: the project is diabetes-focused and includes underserved/community context, but it does not show clear AI/data-science methods.",
        semantic_similarity=0.72,
        mesh_matches=["Diabetes Mellitus", "Health Disparities"],
        retrieval_query_matches=["mq-2", "mq-5"],
        retrieval_query_reasons=["technology + disease", "technology + population"],
        score_breakdown={
            "exact_topic_match": 20,
            "semantic_similarity": 18,
            "mesh_overlap": 15,
            "dimension_coverage": 15,
            "dimension_bonus": 4,
            "retrieval_multi_hit_bonus": 0,
            "recent_funding": 4,
            "technology_penalty": 0,
        },
    )


def test_breakdown_entries_are_deterministic_and_total_correct() -> None:
    entries = build_breakdown_entries(_row().score_breakdown)

    assert [entry.key for entry in entries][:3] == [
        "exact_topic_match",
        "semantic_similarity",
        "mesh_overlap",
    ]
    assert breakdown_total(entries) == 76


def test_apply_explanations_orders_matched_and_missing_concepts() -> None:
    explained_rows = apply_explanations(
        [_row()],
        research_question="AI for diabetes care in underserved populations",
        selected_concepts=["Artificial Intelligence", "Diabetes Mellitus", "Health Disparities"],
        final_keywords=["Artificial Intelligence", "Machine Learning", "Diabetes Mellitus", "Health Equity"],
        mesh_terms=["Medically Underserved Area"],
        semantic_terms=["Clinical Care"],
        semantic_concepts=["Clinical Decision Support Systems"],
        retrieval_trace={
            "multi_query_enabled": True,
            "query_plans": [
                {"query_id": "mq-2", "reason": "Artificial Intelligence + Diabetes Mellitus"},
                {"query_id": "mq-5", "reason": "Machine Learning + Health Disparities"},
            ],
        },
    )

    row = explained_rows[0]

    assert row.matched_concepts[:2] == ["Artificial Intelligence", "Machine Learning"]
    assert row.missing_concepts
    assert any("clinical" in concept.lower() or "underserved" in concept.lower() for concept in row.missing_concepts)
    assert row.retrieved_by == [
        "mq-2: Artificial Intelligence + Diabetes Mellitus",
        "mq-5: Machine Learning + Health Disparities",
    ]
    assert row.score_breakdown_text is not None
    assert "Total: 76 / 100" in row.score_breakdown_text
