from __future__ import annotations

from dataclasses import dataclass

import re

from ..utils import normalize_text, unique_preserve_order


@dataclass(slots=True)
class MeshScoreResult:
    raw_overlap_count: int
    mesh_matches: list[str]


def compute_mesh_score(project_terms: list[str], expanded_terms: list[str]) -> MeshScoreResult:
    if not project_terms or not expanded_terms:
        return MeshScoreResult(raw_overlap_count=0, mesh_matches=[])

    normalized_project_terms = normalize_text(" ".join(project_terms))
    matches = [
        term
        for term in expanded_terms
        if _contains_phrase(normalized_project_terms, term)
    ]
    unique_matches = unique_preserve_order(matches)

    return MeshScoreResult(raw_overlap_count=len(unique_matches), mesh_matches=unique_matches)


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return False
    pattern = re.compile(rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)")
    return bool(pattern.search(text))
