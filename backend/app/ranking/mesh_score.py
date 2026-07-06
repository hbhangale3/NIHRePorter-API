from __future__ import annotations

from dataclasses import dataclass

import re

from ..utils import normalize_text, unique_preserve_order


@dataclass(slots=True)
class MeshScoreResult:
    score: int
    mesh_matches: list[str]


def compute_mesh_score(project_terms: list[str], expanded_terms: list[str]) -> MeshScoreResult:
    if not project_terms or not expanded_terms:
        return MeshScoreResult(score=0, mesh_matches=[])

    normalized_project_terms = normalize_text(" ".join(project_terms))
    matches = [
        term
        for term in expanded_terms
        if _contains_phrase(normalized_project_terms, term)
    ]
    unique_matches = unique_preserve_order(matches)

    if len(unique_matches) >= 3:
        score = 20
    elif len(unique_matches) == 2:
        score = 15
    elif len(unique_matches) == 1:
        score = 10
    else:
        score = 0

    return MeshScoreResult(score=score, mesh_matches=unique_matches)


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return False
    pattern = re.compile(rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)")
    return bool(pattern.search(text))
