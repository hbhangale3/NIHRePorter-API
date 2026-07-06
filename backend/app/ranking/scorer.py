from __future__ import annotations

from dataclasses import dataclass

from ..models import PIOutreachRow
from ..semantic.embedding_model import EmbeddingModel
from ..utils import normalize_text, unique_preserve_order
from .dimensions import DimensionFamily, build_dimension_families, build_query_intent, match_family_terms, requires_technology_match
from .keyword_score import compute_keyword_score
from .mesh_score import compute_mesh_score
from .reasoning import build_reasoning, relevance_badge_for_score
from .semantic_score import SemanticSimilarityScorer


@dataclass(slots=True)
class OutreachRankingContext:
    research_question: str
    expanded_terms: list[str]
    selected_concepts: list[str]
    final_keywords: list[str]
    mesh_terms: list[str]
    semantic_terms: list[str]
    semantic_concepts: list[str]


@dataclass(slots=True)
class RowSignals:
    row: PIOutreachRow
    title_text: str
    full_text: str
    keyword_exact_score: int
    exact_topic_matches: list[str]
    recent_funding_score: int
    semantic_similarity: float
    mesh_matches: list[str]
    raw_mesh_overlap_count: int
    matched_dimensions: list[str]
    missing_dimensions: list[str]
    matched_concepts: list[str]
    ai_match: bool
    disease_match: bool
    population_match: bool
    dimension_match_count: int
    dimension_coverage_ratio: float
    technology_penalty: int


class OutreachRankingScorer:
    def __init__(self, embedding_model: EmbeddingModel | None = None) -> None:
        self.semantic_scorer = SemanticSimilarityScorer(embedding_model=embedding_model)

    def rank_rows(
        self,
        rows: list[PIOutreachRow],
        context: OutreachRankingContext,
    ) -> tuple[list[PIOutreachRow], dict[str, object]]:
        if not rows:
            return [], _ranking_summary([])

        query_intent = build_query_intent(
            research_question=context.research_question,
            selected_concepts=context.selected_concepts,
            final_keywords=context.final_keywords,
            mesh_terms=context.mesh_terms,
            semantic_terms=context.semantic_terms,
            semantic_concepts=context.semantic_concepts,
        )
        dimension_families = build_dimension_families(
            research_question=context.research_question,
            selected_concepts=context.selected_concepts,
            final_keywords=context.final_keywords,
            mesh_terms=context.mesh_terms,
            semantic_terms=context.semantic_terms,
            semantic_concepts=context.semantic_concepts,
        )
        technology_required_for_high = requires_technology_match(query_intent)
        semantic_results = self.semantic_scorer.score_rows(context.research_question, rows)

        row_signals = [
            self._collect_row_signals(
                row=row,
                index=index,
                context=context,
                semantic_results=semantic_results,
                dimension_families=dimension_families,
                technology_required_for_high=technology_required_for_high,
            )
            for index, row in enumerate(rows)
        ]

        semantic_points = _normalize_component(
            [signal.semantic_similarity for signal in row_signals],
            max_points=28,
        )
        mesh_points = _normalize_component(
            [float(signal.raw_mesh_overlap_count) for signal in row_signals],
            max_points=18,
        )

        ranked_rows: list[PIOutreachRow] = []
        for index, signal in enumerate(row_signals):
            coverage_points = round(24 * signal.dimension_coverage_ratio)
            dimension_bonus = _dimension_bonus(signal.dimension_match_count, len(dimension_families))
            total_score = min(
                100,
                signal.keyword_exact_score
                + semantic_points[index]
                + mesh_points[index]
                + coverage_points
                + dimension_bonus
                + _retrieval_multi_hit_bonus(signal.row.retrieval_query_matches)
                + signal.recent_funding_score,
            )
            if signal.technology_penalty > 0:
                total_score = max(0, total_score - signal.technology_penalty)
            if technology_required_for_high and not signal.ai_match:
                total_score = min(total_score, 79)

            reasoning = build_reasoning(
                score=total_score,
                matched_dimensions=signal.matched_dimensions,
                missing_dimensions=signal.missing_dimensions,
                matched_concepts=signal.matched_concepts,
                mesh_matches=signal.mesh_matches,
                semantic_similarity=signal.semantic_similarity,
                recent_funding_score=signal.recent_funding_score,
            )

            ranked_rows.append(
                signal.row.model_copy(
                    update={
                        "relevance_score": total_score,
                        "matched_dimensions": signal.matched_dimensions,
                        "missing_dimensions": signal.missing_dimensions,
                        "dimension_match_count": signal.dimension_match_count,
                        "dimension_coverage_ratio": round(signal.dimension_coverage_ratio, 3),
                        "reasoning": reasoning,
                        "semantic_similarity": round(signal.semantic_similarity, 4),
                        "mesh_matches": signal.mesh_matches,
                        "matched_concepts": signal.matched_concepts,
                        "ai_match": signal.ai_match,
                        "disease_match": signal.disease_match,
                        "population_match": signal.population_match,
                        "relevance_badge": relevance_badge_for_score(total_score),
                        "score_breakdown": {
                            "exact_topic_match": signal.keyword_exact_score,
                            "semantic_similarity": semantic_points[index],
                            "mesh_overlap": mesh_points[index],
                            "dimension_coverage": coverage_points,
                            "dimension_bonus": dimension_bonus,
                            "retrieval_multi_hit_bonus": _retrieval_multi_hit_bonus(signal.row.retrieval_query_matches),
                            "recent_funding": signal.recent_funding_score,
                        },
                    }
                )
            )

        ranked_rows.sort(
            key=lambda row: (
                -(row.relevance_score or 0),
                -(row.dimension_coverage_ratio or 0.0),
                -(row.semantic_similarity or 0.0),
                row.pi_last_name or "",
                row.pi_name or "",
            )
        )
        return ranked_rows, _ranking_summary(ranked_rows)

    def _collect_row_signals(
        self,
        *,
        row: PIOutreachRow,
        index: int,
        context: OutreachRankingContext,
        semantic_results: dict[str, object],
        dimension_families: list[DimensionFamily],
        technology_required_for_high: bool,
    ) -> RowSignals:
        title_text = " ".join(row.sample_project_titles)
        full_text = " ".join([title_text, " ".join(row.project_abstracts), " ".join(row.project_terms)]).strip()
        keyword_result = compute_keyword_score(
            research_question=context.research_question,
            title_text=title_text,
            full_text=full_text,
            fiscal_years=row.fiscal_years,
        )
        mesh_result = compute_mesh_score(row.project_terms, context.expanded_terms)
        row_key = row.pi_profile_id or "|".join(row.project_numbers) or f"row-{index}"
        semantic_similarity = float(getattr(semantic_results.get(row_key), "similarity", 0.0))

        matched_dimensions: list[str] = []
        missing_dimensions: list[str] = []
        matched_concepts = list(keyword_result.exact_topic_matches)

        normalized_text = normalize_text(full_text)
        for family in dimension_families:
            matches = match_family_terms(normalized_text, family)
            if matches:
                matched_dimensions.append(family.label)
                matched_concepts.extend(matches)
            else:
                missing_dimensions.append(family.label)

        if mesh_result.mesh_matches:
            matched_concepts.extend(mesh_result.mesh_matches)

        dimension_match_count = len(matched_dimensions)
        dimension_coverage_ratio = (
            dimension_match_count / len(dimension_families) if dimension_families else 0.0
        )

        return RowSignals(
            row=row,
            title_text=title_text,
            full_text=full_text,
            keyword_exact_score=keyword_result.exact_topic_score,
            exact_topic_matches=keyword_result.exact_topic_matches,
            recent_funding_score=keyword_result.recent_funding_score,
            semantic_similarity=semantic_similarity,
            mesh_matches=mesh_result.mesh_matches,
            raw_mesh_overlap_count=mesh_result.raw_overlap_count,
            matched_dimensions=matched_dimensions,
            missing_dimensions=missing_dimensions,
            matched_concepts=unique_preserve_order(matched_concepts),
            ai_match="AI / Data Science" in matched_dimensions,
            disease_match=any(label.endswith("/ Condition") or "Disease" in label for label in matched_dimensions),
            population_match="Population / Equity / Access" in matched_dimensions,
            dimension_match_count=dimension_match_count,
            dimension_coverage_ratio=dimension_coverage_ratio,
            technology_penalty=12 if technology_required_for_high and "AI / Data Science" not in matched_dimensions else 0,
        )


def _dimension_bonus(matched_count: int, total_count: int) -> int:
    if total_count == 0 or matched_count == 0:
        return 0
    if matched_count >= 3:
        return 12
    if matched_count == 2:
        return 6
    return 3


def _retrieval_multi_hit_bonus(query_matches: list[str]) -> int:
    match_count = len(query_matches)
    if match_count >= 4:
        return 6
    if match_count >= 2:
        return 3
    return 0


def _normalize_component(values: list[float], *, max_points: int) -> list[int]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high <= low:
        if high <= 0:
            return [0 for _ in values]
        return [max_points for _ in values]

    normalized: list[int] = []
    for value in values:
        ratio = (value - low) / (high - low)
        normalized.append(int(round(max_points * ratio)))
    return normalized


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
        "highly_relevant_count": sum(1 for score in scores if score >= 80),
        "moderately_relevant_count": sum(1 for score in scores if 60 <= score < 80),
        "weak_match_count": sum(1 for score in scores if 40 <= score < 60),
        "low_match_count": sum(1 for score in scores if score < 40),
    }
