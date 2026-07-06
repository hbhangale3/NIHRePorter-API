from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.query_intent import extract_query_intent


def test_query_intent_extracts_major_dimensions_for_ai_diabetes_underserved_query() -> None:
    intent = extract_query_intent(
        research_question="AI for diabetes care in underserved populations",
        selected_concepts=["artificial intelligence", "diabetes", "underserved populations"],
    )

    assert "ai" in [term.lower() for term in intent.technology_method]
    assert any("diabetes" in term.lower() for term in intent.disease_condition)
    assert any("underserved" in term.lower() or "equity" in term.lower() for term in intent.population_equity_access)
    assert any("care" in term.lower() or "clinical" in term.lower() for term in intent.care_delivery_context)
