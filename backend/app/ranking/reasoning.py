from __future__ import annotations

from ..utils import unique_preserve_order


def relevance_badge_for_score(score: int) -> str:
    if score >= 90:
        return "Highly Relevant"
    if score >= 75:
        return "Moderately Relevant"
    if score >= 60:
        return "Weak Match"
    return "Low Match"


def build_reasoning(
    *,
    exact_topic_score: int,
    mesh_matches: list[str],
    semantic_score: int,
    population_match: bool,
    ai_match: bool,
    disease_match: bool,
    recent_funding_score: int,
) -> tuple[list[str], str]:
    dimensions: list[str] = []

    if exact_topic_score > 0:
        dimensions.append("Exact topic wording matched in title")
    if mesh_matches:
        dimensions.append("MeSH concepts overlapped expanded terms")
    if semantic_score >= 16:
        dimensions.append("High semantic similarity")
    elif semantic_score > 0:
        dimensions.append("Moderate semantic similarity")
    if population_match:
        dimensions.append("Health equity or underserved population terms matched")
    if ai_match:
        dimensions.append("AI concepts matched")
    if disease_match:
        dimensions.append("Diabetes concepts matched")
    if recent_funding_score > 0:
        dimensions.append("Recent NIH funding activity")

    ordered = unique_preserve_order(dimensions)
    if not ordered:
        return [], "No strong relevance signals matched the current research question."
    reasoning = " ; ".join(f"✓ {dimension}" for dimension in ordered)
    return ordered, reasoning
