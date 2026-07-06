from __future__ import annotations

import re
from dataclasses import dataclass

from ..utils import normalize_text, unique_preserve_order


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "using",
    "via",
    "with",
}

POPULATION_TERMS = [
    "underserved",
    "minority",
    "minorities",
    "equity",
    "health equity",
    "health disparities",
    "disparities",
    "community",
    "rural",
    "medically underserved",
    "low income",
    "uninsured",
]

AI_TERMS = [
    "ai",
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "predictive model",
    "predictive modeling",
    "large language model",
    "llm",
    "computer vision",
    "natural language processing",
    "nlp",
    "clinical decision support",
]

DISEASE_TERMS = [
    "diabetes",
    "diabetic",
    "glycemic",
    "glucose",
    "insulin",
    "a1c",
    "hemoglobin a1c",
    "prediabetes",
]


@dataclass(slots=True)
class KeywordScoreResult:
    exact_topic_score: int
    exact_topic_matches: list[str]
    population_match: bool
    population_terms: list[str]
    ai_match: bool
    ai_terms: list[str]
    disease_match: bool
    disease_terms: list[str]
    recent_funding_score: int


def compute_keyword_score(
    *,
    research_question: str,
    title_text: str,
    full_text: str,
    fiscal_years: list[int],
) -> KeywordScoreResult:
    title_normalized = normalize_text(title_text)
    full_text_normalized = normalize_text(full_text)
    question_tokens = _significant_tokens(research_question)
    exact_matches = [token for token in question_tokens if _contains_phrase(title_normalized, token)]
    exact_topic_score = _exact_topic_points(
        research_question=research_question,
        title_normalized=title_normalized,
        total_tokens=len(question_tokens),
        matched_tokens=len(exact_matches),
    )

    population_terms = _find_matching_terms(full_text_normalized, POPULATION_TERMS)
    ai_terms = _find_matching_terms(full_text_normalized, AI_TERMS)
    disease_terms = _find_matching_terms(full_text_normalized, DISEASE_TERMS)

    return KeywordScoreResult(
        exact_topic_score=exact_topic_score,
        exact_topic_matches=exact_matches,
        population_match=bool(population_terms),
        population_terms=population_terms,
        ai_match=bool(ai_terms),
        ai_terms=ai_terms,
        disease_match=bool(disease_terms),
        disease_terms=disease_terms,
        recent_funding_score=_recent_funding_points(fiscal_years),
    )


def _significant_tokens(text: str) -> list[str]:
    normalized = normalize_text(text)
    tokens = re.findall(r"[a-z0-9]+", normalized)
    return unique_preserve_order(
        [token for token in tokens if len(token) >= 3 and token not in STOPWORDS]
    )


def _exact_topic_points(
    *,
    research_question: str,
    title_normalized: str,
    total_tokens: int,
    matched_tokens: int,
) -> int:
    normalized_question = normalize_text(research_question)
    if normalized_question and normalized_question in title_normalized:
        return 25
    if total_tokens == 0 or matched_tokens == 0:
        return 0

    overlap_ratio = matched_tokens / total_tokens
    if matched_tokens >= 3 or overlap_ratio >= 0.6:
        return 25
    if matched_tokens >= 2 or overlap_ratio >= 0.4:
        return 18
    return 10


def _find_matching_terms(text: str, terms: list[str]) -> list[str]:
    matches = [term for term in terms if _contains_phrase(text, term)]
    return unique_preserve_order(matches)


def _recent_funding_points(fiscal_years: list[int]) -> int:
    if not fiscal_years:
        return 0
    latest = max(fiscal_years)
    if latest >= 2025:
        return 5
    if latest >= 2024:
        return 3
    return 0


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return False
    pattern = re.compile(rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)")
    return bool(pattern.search(text))
