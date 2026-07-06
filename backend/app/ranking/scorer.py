from __future__ import annotations

from dataclasses import dataclass

from ..models import PIOutreachRow
from ..semantic.embedding_model import EmbeddingModel
from ..utils import unique_preserve_order
from .keyword_score import compute_keyword_score
from .mesh_score import compute_mesh_score
from .reasoning import build_reasoning, relevance_badge_for_score
from .semantic_score import SemanticSimilarityScorer


@dataclass(slots=True)
class OutreachRankingContext:
    research_question: str
    expanded_terms: list[str]


class OutreachRankingScorer:
    def __init__(self, embedding_model: EmbeddingModel | None = None) -> None:
        self.semantic_scorer = SemanticSimilarityScorer(embedding_model=embedding_model)

    def rank_rows(
        self,
        rows: list[PIOutreachRow],
        context: OutreachRankingContext,
    ) -> tuple[list[PIOutreachRow], dict[str, object]]:
        semantic_results = self.semantic_scorer.score_rows(context.research_question, rows)
        ranked_rows: list[PIOutreachRow] = []

        for index, row in enumerate(rows):
            title_text = " ".join(row.sample_project_titles)
            full_text = " ".join(
                [title_text, " ".join(row.project_abstracts), " ".join(row.project_terms)]
            ).strip()
            keyword_result = compute_keyword_score(
                research_question=context.research_question,
                title_text=title_text,
                full_text=full_text,
                fiscal_years=row.fiscal_years,
            )
            mesh_result = compute_mesh_score(row.project_terms, context.expanded_terms)
            row_key = row.pi_profile_id or "|".join(row.project_numbers) or f"row-{index}"
            semantic_result = semantic_results.get(row_key)
            semantic_similarity = semantic_result.similarity if semantic_result else 0.0
            semantic_score = semantic_result.score if semantic_result else 0

            total_score = min(
                100,
                keyword_result.exact_topic_score
                + mesh_result.score
                + semantic_score
                + (10 if keyword_result.population_match else 0)
                + (10 if keyword_result.ai_match else 0)
                + (10 if keyword_result.disease_match else 0)
                + keyword_result.recent_funding_score,
            )

            matched_dimensions, reasoning = build_reasoning(
                exact_topic_score=keyword_result.exact_topic_score,
                mesh_matches=mesh_result.mesh_matches,
                semantic_score=semantic_score,
                population_match=keyword_result.population_match,
                ai_match=keyword_result.ai_match,
                disease_match=keyword_result.disease_match,
                recent_funding_score=keyword_result.recent_funding_score,
            )

            matched_concepts = unique_preserve_order(
                mesh_result.mesh_matches
                + keyword_result.ai_terms
                + keyword_result.disease_terms
                + keyword_result.population_terms
                + keyword_result.exact_topic_matches
            )

            ranked_rows.append(
                row.model_copy(
                    update={
                        "relevance_score": total_score,
                        "matched_dimensions": matched_dimensions,
                        "reasoning": reasoning,
                        "semantic_similarity": round(semantic_similarity, 4),
                        "mesh_matches": mesh_result.mesh_matches,
                        "matched_concepts": matched_concepts,
                        "ai_match": keyword_result.ai_match,
                        "disease_match": keyword_result.disease_match,
                        "population_match": keyword_result.population_match,
                        "relevance_badge": relevance_badge_for_score(total_score),
                        "score_breakdown": {
                            "exact_topic_match": keyword_result.exact_topic_score,
                            "mesh_overlap": mesh_result.score,
                            "semantic_similarity": semantic_score,
                            "population_match": 10 if keyword_result.population_match else 0,
                            "ai_relevance": 10 if keyword_result.ai_match else 0,
                            "disease_relevance": 10 if keyword_result.disease_match else 0,
                            "recent_funding": keyword_result.recent_funding_score,
                        },
                    }
                )
            )

        ranked_rows.sort(
            key=lambda row: (
                -(row.relevance_score or 0),
                -(row.semantic_similarity or 0.0),
                row.pi_last_name or "",
                row.pi_name or "",
            )
        )
        return ranked_rows, _ranking_summary(ranked_rows)


def _ranking_summary(rows: list[PIOutreachRow]) -> dict[str, object]:
    if not rows:
        return {
            "average_relevance_score": 0.0,
            "highly_relevant_count": 0,
            "moderately_relevant_count": 0,
            "weak_match_count": 0,
            "low_match_count": 0,
        }

    scores = [row.relevance_score or 0 for row in rows]
    return {
        "average_relevance_score": round(sum(scores) / len(scores), 1),
        "highly_relevant_count": sum(1 for score in scores if score >= 90),
        "moderately_relevant_count": sum(1 for score in scores if 75 <= score < 90),
        "weak_match_count": sum(1 for score in scores if 60 <= score < 75),
        "low_match_count": sum(1 for score in scores if score < 60),
    }
