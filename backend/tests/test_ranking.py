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
                    1.0 if any(term in lowered for term in ["diabetes", "glycemic", "glucose", "insulin", "a1c"]) else 0.0,
                    1.0 if any(term in lowered for term in ["artificial intelligence", "machine learning", "ai", "deep learning"]) else 0.0,
                    1.0 if any(term in lowered for term in ["underserved", "health disparities", "equity", "community health"]) else 0.0,
                    1.0 if any(term in lowered for term in ["telemedicine", "telehealth", "digital health"]) else 0.0,
                ],
                dtype=np.float32,
            )
            norm = np.linalg.norm(vector)
            vectors.append(vector if norm == 0 else vector / norm)
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


def _context() -> OutreachRankingContext:
    return OutreachRankingContext(
        research_question="AI for diabetes care in underserved populations",
        expanded_terms=[
            "Diabetes Mellitus",
            "Artificial Intelligence",
            "Health Equity",
            "Underserved",
            "Community Health",
        ],
        selected_concepts=[
            "artificial intelligence",
            "diabetes",
            "underserved populations",
        ],
        final_keywords=[
            "artificial intelligence",
            "machine learning",
            "diabetes",
            "health disparities",
            "underserved",
        ],
        mesh_terms=[
            "Artificial Intelligence",
            "Diabetes Mellitus",
            "Health Equity",
        ],
        semantic_terms=[
            "machine learning",
            "community health",
            "glycemic control",
        ],
        semantic_concepts=[
            "Artificial Intelligence",
            "Diabetes Mellitus",
            "Health Equity",
        ],
    )


def test_ranking_spreads_scores_and_orders_results() -> None:
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
            title="Community diabetes care coordination program",
            abstract="Diabetes intervention for underserved adults in community health settings.",
            terms=["Diabetes Mellitus", "Community Health"],
            fiscal_years=[2024],
            project_number="P2",
        ),
        _row(
            name="Casey Disease",
            title="Diabetes management biomarkers",
            abstract="Study of diabetes progression and glycemic control.",
            terms=["Diabetes Mellitus"],
            fiscal_years=[2024],
            project_number="P3",
        ),
        _row(
            name="Dana Weak",
            title="Imaging biomarkers in mouse models",
            abstract="Preclinical imaging study.",
            terms=["Imaging"],
            fiscal_years=[2022],
            project_number="P4",
        ),
    ]
    scorer = OutreachRankingScorer(embedding_model=FakeEmbeddingModel())

    ranked_rows, ranking_summary = scorer.rank_rows(rows, _context())

    assert [row.pi_name for row in ranked_rows] == [
        "Alex Strong",
        "Bailey Mid",
        "Casey Disease",
        "Dana Weak",
    ]
    assert ranked_rows[0].relevance_score >= 80
    assert 60 <= ranked_rows[1].relevance_score < 80
    assert ranked_rows[2].relevance_score < ranked_rows[1].relevance_score
    assert ranked_rows[2].relevance_badge != "Highly Relevant"
    assert ranked_rows[3].relevance_badge == "Low Relevance"
    assert ranking_summary["highly_relevant_count"] >= 1
    assert ranking_summary["moderately_relevant_count"] >= 1
    assert ranking_summary["weak_match_count"] + ranking_summary["low_match_count"] >= 1


def test_dimension_coverage_and_missing_dimensions_are_reported() -> None:
    row = _row(
        name="Bailey Mid",
        title="Community diabetes care coordination program",
        abstract="Diabetes intervention for underserved adults in community health settings.",
        terms=["Diabetes Mellitus", "Community Health"],
        fiscal_years=[2024],
        project_number="P2",
    )
    scorer = OutreachRankingScorer(embedding_model=FakeEmbeddingModel())

    ranked_rows, _ = scorer.rank_rows([row], _context())
    ranked = ranked_rows[0]

    assert ranked.dimension_match_count >= 2
    assert ranked.dimension_coverage_ratio > 0
    assert "AI / Data Science" in ranked.missing_dimensions
    assert any("Disease" in dimension for dimension in ranked.matched_dimensions)
    assert "Missing dimensions:" in ranked.reasoning
    assert "Matched dimensions:" in ranked.reasoning


def test_ai_disease_population_match_scores_above_disease_only() -> None:
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
            name="Casey Disease",
            title="Diabetes management biomarkers",
            abstract="Study of diabetes progression and glycemic control.",
            terms=["Diabetes Mellitus"],
            fiscal_years=[2024],
            project_number="P3",
        ),
    ]
    scorer = OutreachRankingScorer(embedding_model=FakeEmbeddingModel())

    ranked_rows, _ = scorer.rank_rows(rows, _context())

    assert ranked_rows[0].pi_name == "Alex Strong"
    assert ranked_rows[0].relevance_score > ranked_rows[1].relevance_score
    assert ranked_rows[1].relevance_badge != "Highly Relevant"
    assert ranked_rows[1].dimension_match_count < ranked_rows[0].dimension_match_count
    assert "AI / Data Science" in ranked_rows[1].missing_dimensions
    assert "AI / Data Science" in ranked_rows[1].reasoning


def test_high_relevance_requires_technology_dimension_when_query_includes_ai() -> None:
    row = _row(
        name="Bailey Mid",
        title="Community diabetes care coordination program",
        abstract="Diabetes intervention for underserved adults in community health settings.",
        terms=["Diabetes Mellitus", "Community Health"],
        fiscal_years=[2025],
        project_number="P2",
    )
    scorer = OutreachRankingScorer(embedding_model=FakeEmbeddingModel())

    ranked_rows, _ = scorer.rank_rows([row], _context())

    assert ranked_rows[0].ai_match is False
    assert ranked_rows[0].relevance_score < 80
    assert ranked_rows[0].relevance_badge != "Highly Relevant"


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
            name="Casey Disease",
            title="Diabetes management biomarkers",
            abstract="Study of diabetes progression and glycemic control.",
            terms=["Diabetes Mellitus"],
            fiscal_years=[2024],
            project_number="P3",
        ),
    ]
    scorer = OutreachRankingScorer(embedding_model=FakeEmbeddingModel())
    context = _context()

    first_run, _ = scorer.rank_rows(rows, context)
    second_run, _ = scorer.rank_rows(rows, context)

    assert [row.relevance_score for row in first_run] == [row.relevance_score for row in second_run]
    assert [row.reasoning for row in first_run] == [row.reasoning for row in second_run]
    assert [row.missing_dimensions for row in first_run] == [row.missing_dimensions for row in second_run]


def test_relevance_badge_thresholds() -> None:
    assert relevance_badge_for_score(95) == "Highly Relevant"
    assert relevance_badge_for_score(70) == "Moderately Relevant"
    assert relevance_badge_for_score(45) == "Weak Match"
    assert relevance_badge_for_score(39) == "Low Relevance"


def test_ranking_accepts_retrieval_query_matches_without_breaking() -> None:
    row = _row(
        name="Alex Strong",
        title="Artificial intelligence for diabetes care in underserved communities",
        abstract="Machine learning models for health disparities in diabetes care.",
        terms=["Diabetes Mellitus", "Artificial Intelligence", "Health Equity"],
        fiscal_years=[2025, 2026],
        project_number="P1",
    )
    row.retrieval_query_matches = ["mq-1", "mq-2", "mq-3"]
    row.retrieval_query_reasons = ["tech + disease", "tech + equity", "disease + equity"]

    scorer = OutreachRankingScorer(embedding_model=FakeEmbeddingModel())
    ranked_rows, _ = scorer.rank_rows([row], _context())

    assert ranked_rows[0].relevance_score >= 80
    assert ranked_rows[0].score_breakdown["retrieval_multi_hit_bonus"] > 0
