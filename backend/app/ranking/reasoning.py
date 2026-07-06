from __future__ import annotations

def relevance_badge_for_score(score: int) -> str:
    if score >= 80:
        return "Highly Relevant"
    if score >= 60:
        return "Moderately Relevant"
    if score >= 40:
        return "Weak Match"
    return "Low Relevance"


def build_reasoning(
    *,
    score: int,
    matched_dimensions: list[str],
    missing_dimensions: list[str],
    matched_concepts: list[str],
    mesh_matches: list[str],
    semantic_similarity: float,
    recent_funding_score: int,
) -> str:
    if score >= 80:
        strength = "Strong match"
        candidate_line = "This looks like a strong outreach candidate."
    elif score >= 60:
        strength = "Moderate match"
        candidate_line = "This looks like a reasonable outreach candidate with some gaps."
    elif score >= 40:
        strength = "Weak match"
        candidate_line = "This project overlaps part of the query but is missing important dimensions."
    else:
        strength = "Low relevance"
        candidate_line = "This project does not align closely enough to prioritize for outreach."

    detail_parts: list[str] = []
    if matched_dimensions:
        detail_parts.append(f"Matched dimensions: {', '.join(matched_dimensions)}")
    if missing_dimensions:
        detail_parts.append(f"Missing dimensions: {', '.join(missing_dimensions)}")
    if matched_concepts:
        detail_parts.append(f"Matched concepts include {', '.join(matched_concepts[:6])}")
    if mesh_matches:
        detail_parts.append(f"MeSH overlap: {', '.join(mesh_matches[:5])}")
    if semantic_similarity >= 0.75:
        detail_parts.append("Semantic alignment is high across the retrieved set")
    elif semantic_similarity >= 0.45:
        detail_parts.append("Semantic alignment is moderate across the retrieved set")
    elif semantic_similarity > 0:
        detail_parts.append("Semantic alignment is limited across the retrieved set")
    if recent_funding_score >= 5:
        detail_parts.append("Recent NIH funding strengthens relevance")

    if not detail_parts:
        detail_parts.append("No strong dimension or semantic signals were detected")

    return f"{strength}: {'; '.join(detail_parts)}. {candidate_line}"
