from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models import PIOutreachRow
from app.ranking import OutreachRankingContext, OutreachRankingScorer, relevance_badge_for_score


class FakeEmbeddingModel:
    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text], batch_size=1)[0]

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        vectors = []
        for text in texts:
            lowered = text.lower()
            vector = np.asarray(
                [
                    1.0 if "diabetes" in lowered else 0.0,
                    1.0 if "artificial intelligence" in lowered or "machine learning" in lowered or "ai" in lowered else 0.0,
                    1.0 if "underserved" in lowered or "health disparities" in lowered or "equity" in lowered else 0.0,
                ],
                dtype=np.float32,
            )
            norm = np.linalg.norm(vector)
            if norm == 0:
                vectors.append(vector)
            else:
                vectors.append(vector / norm)
        return np.vstack(vectors)


def _row(
    *,
    name: str,
    title: str,
    abstract: str,
    terms: list[str],
    fiscal_years: list[int],
    project_number: str,
) -> PIOutreachRow:
    return PIOutreachRow(
        pi_name=name,
        pi_first_name=name.split()[0],
        pi_last_name=name.split()[-1],
        sample_project_titles=[title],
        project_abstracts=[abstract],
        project_terms=terms,
        fiscal_years=fiscal_years,
        project_numbers=[project_number],
        project_ids=[project_number],
        matched_topics=["AI + Health Disparities"],
    )


def test_ranking_orders_results_and_generates_reasoning() -> None:
    rows = [
        _row(
            name="Alex Strong",
            title="Artificial intelligence for diabetes care in underserved communities",
            abstract="Machine learning models for health disparities in diabetes care.",
            terms=["Diabetes Mellitus", "Artificial Intelligence", "Health Equity"],
            fiscal_years=[2025, 2026],
            project_number="P1",
        ),
        _row(
            name="Bailey Mid",
            title="Telemedicine support for diabetes management",
            abstract="Community-focused diabetes self-management research.",
            terms=["Diabetes Mellitus", "Telemedicine"],
            fiscal_years=[2024],
            project_number="P2",
        ),
        _row(
            name="Casey Weak",
            title="Imaging biomarkers in mouse models",
            abstract="Preclinical imaging study.",
            terms=["Imaging"],
            fiscal_years=[2022],
            project_number="P3",
        ),
    ]
    scorer = OutreachRankingScorer(embedding_model=FakeEmbeddingModel())
    context = OutreachRankingContext(
        research_question="AI for diabetes care in underserved populations",
        expanded_terms=["Diabetes Mellitus", "Artificial Intelligence", "Health Equity"],
    )

    ranked_rows, ranking_summary = scorer.rank_rows(rows, context)

    assert [row.pi_name for row in ranked_rows] == ["Alex Strong", "Bailey Mid", "Casey Weak"]
    assert ranked_rows[0].relevance_score > ranked_rows[1].relevance_score > ranked_rows[2].relevance_score
    assert ranked_rows[0].relevance_badge == "Highly Relevant"
    assert ranked_rows[1].relevance_badge == "Low Match"
    assert "AI concepts matched" in ranked_rows[0].reasoning
    assert "Diabetes concepts matched" in ranked_rows[0].reasoning
    assert ranked_rows[0].mesh_matches == ["Diabetes Mellitus", "Artificial Intelligence", "Health Equity"]
    assert ranking_summary["highly_relevant_count"] == 1


def test_ranking_is_deterministic() -> None:
    rows = [
        _row(
            name="Alex Strong",
            title="Artificial intelligence for diabetes care in underserved communities",
            abstract="Machine learning models for health disparities in diabetes care.",
            terms=["Diabetes Mellitus", "Artificial Intelligence", "Health Equity"],
            fiscal_years=[2025, 2026],
            project_number="P1",
        ),
        _row(
            name="Bailey Mid",
            title="Telemedicine support for diabetes management",
            abstract="Community-focused diabetes self-management research.",
            terms=["Diabetes Mellitus", "Telemedicine"],
            fiscal_years=[2024],
            project_number="P2",
        ),
    ]
    context = OutreachRankingContext(
        research_question="AI for diabetes care in underserved populations",
        expanded_terms=["Diabetes Mellitus", "Artificial Intelligence", "Health Equity"],
    )
    scorer = OutreachRankingScorer(embedding_model=FakeEmbeddingModel())

    first_run, _ = scorer.rank_rows(rows, context)
    second_run, _ = scorer.rank_rows(rows, context)

    assert [row.relevance_score for row in first_run] == [row.relevance_score for row in second_run]
    assert [row.reasoning for row in first_run] == [row.reasoning for row in second_run]


def test_relevance_badge_thresholds() -> None:
    assert relevance_badge_for_score(95) == "Highly Relevant"
    assert relevance_badge_for_score(80) == "Moderately Relevant"
    assert relevance_badge_for_score(60) == "Weak Match"
    assert relevance_badge_for_score(59) == "Low Match"
