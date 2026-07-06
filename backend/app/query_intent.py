from __future__ import annotations

from dataclasses import dataclass, field
import re

from .utils import normalize_text, unique_preserve_order


GENERIC_TERMS = {
    "access",
    "adults",
    "algorithm",
    "algorithms",
    "analytics",
    "care",
    "clinical",
    "community",
    "condition",
    "conditions",
    "context",
    "data",
    "delivery",
    "disease",
    "diseases",
    "health",
    "healthcare",
    "improve",
    "intervention",
    "management",
    "medicine",
    "method",
    "methods",
    "model",
    "models",
    "patient",
    "patients",
    "population",
    "populations",
    "predictive",
    "question",
    "research",
    "science",
    "service",
    "services",
    "setting",
    "study",
    "support",
    "system",
    "systems",
    "technology",
    "treatment",
}

TECHNOLOGY_METHOD_TERMS = [
    "ai",
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "neural network",
    "predictive model",
    "predictive modeling",
    "predictive analytics",
    "natural language processing",
    "nlp",
    "llm",
    "large language model",
    "computer vision",
    "clinical decision support",
    "decision support systems clinical",
    "clinical decision support systems",
    "medical informatics",
    "informatics",
    "algorithm",
    "algorithms",
    "data science",
]

POPULATION_EQUITY_ACCESS_TERMS = [
    "underserved populations",
    "underserved population",
    "medically underserved",
    "medically underserved area",
    "health disparities",
    "healthcare disparities",
    "health equity",
    "equity",
    "minority",
    "minorities",
    "vulnerable population",
    "vulnerable populations",
    "social determinants",
    "rural",
    "community health",
    "community",
    "access to care",
    "underserved",
]

CARE_DELIVERY_CONTEXT_TERMS = [
    "clinical care",
    "care delivery",
    "care coordination",
    "disease management",
    "clinical management",
    "clinical workflow",
    "patient care",
    "care",
    "clinical",
]

DIGITAL_HEALTH_TERMS = [
    "telemedicine",
    "telehealth",
    "digital health",
    "mobile health",
    "mhealth",
    "remote monitoring",
    "wearable",
    "wearables",
]

DISEASE_SYNONYM_MAP: dict[str, list[str]] = {
    "diabetes": ["diabetes", "diabetes mellitus", "diabetic", "glycemic", "glucose", "insulin", "a1c"],
    "obesity": ["obesity", "obese", "body mass index", "bmi", "adiposity"],
    "hypertension": ["hypertension", "hypertensive", "blood pressure"],
    "cancer": ["cancer", "tumor", "tumour", "oncology", "neoplasm"],
    "depression": ["depression", "depressive", "mood disorder"],
}


@dataclass(slots=True)
class QueryIntent:
    technology_method: list[str] = field(default_factory=list)
    disease_condition: list[str] = field(default_factory=list)
    population_equity_access: list[str] = field(default_factory=list)
    care_delivery_context: list[str] = field(default_factory=list)
    other: list[str] = field(default_factory=list)

    def dimension_items(self) -> list[tuple[str, list[str]]]:
        return [
            ("technology_method", self.technology_method),
            ("disease_condition", self.disease_condition),
            ("population_equity_access", self.population_equity_access),
            ("care_delivery_context", self.care_delivery_context),
            ("other", self.other),
        ]


def extract_query_intent(
    *,
    research_question: str,
    selected_concepts: list[str] | None = None,
    final_keywords: list[str] | None = None,
    mesh_terms: list[str] | None = None,
    semantic_terms: list[str] | None = None,
    semantic_concepts: list[str] | None = None,
) -> QueryIntent:
    source_terms = unique_preserve_order(
        [research_question]
        + list(selected_concepts or [])
        + list(final_keywords or [])
        + list(mesh_terms or [])
        + list(semantic_terms or [])
        + list(semantic_concepts or [])
    )
    normalized_source_text = " | ".join(
        normalize_text(term) for term in source_terms if normalize_text(term)
    )

    technology_method = _collect_matches(normalized_source_text, TECHNOLOGY_METHOD_TERMS)
    population_equity_access = _collect_matches(normalized_source_text, POPULATION_EQUITY_ACCESS_TERMS)
    care_delivery_context = _collect_matches(normalized_source_text, CARE_DELIVERY_CONTEXT_TERMS)
    digital_health_matches = _collect_matches(normalized_source_text, DIGITAL_HEALTH_TERMS)

    if digital_health_matches:
        care_delivery_context = unique_preserve_order(care_delivery_context + digital_health_matches)

    disease_condition = _collect_disease_matches(source_terms, normalized_source_text)
    other = _collect_other_terms(
        source_terms,
        exclude_terms=technology_method + population_equity_access + care_delivery_context + disease_condition,
    )

    return QueryIntent(
        technology_method=technology_method,
        disease_condition=disease_condition,
        population_equity_access=population_equity_access,
        care_delivery_context=care_delivery_context,
        other=other,
    )


def _collect_matches(source_text: str, dictionary_terms: list[str]) -> list[str]:
    return unique_preserve_order([term for term in dictionary_terms if _contains_phrase(source_text, term)])


def _collect_disease_matches(source_terms: list[str], normalized_source_text: str) -> list[str]:
    matches: list[str] = []
    for synonyms in DISEASE_SYNONYM_MAP.values():
        if any(_contains_phrase(normalized_source_text, synonym) for synonym in synonyms):
            matches.extend(synonyms[:2])

    if matches:
        return unique_preserve_order(matches)

    residual_terms: list[str] = []
    for term in source_terms:
        normalized = normalize_text(term)
        if not normalized:
            continue
        tokens = [token for token in re.findall(r"[a-z0-9]+", normalized) if token not in GENERIC_TERMS]
        if not tokens:
            continue
        phrase = " ".join(tokens[:3]).strip()
        if len(phrase) >= 4:
            residual_terms.append(phrase)
        residual_terms.extend(token for token in tokens if len(token) >= 4)
    return unique_preserve_order(residual_terms[:8])


def _collect_other_terms(source_terms: list[str], *, exclude_terms: list[str]) -> list[str]:
    normalized_exclude = {normalize_text(term) for term in exclude_terms if normalize_text(term)}
    other: list[str] = []
    for term in source_terms:
        normalized = normalize_text(term)
        if not normalized or normalized in normalized_exclude:
            continue
        tokens = [token for token in re.findall(r"[a-z0-9]+", normalized) if token not in GENERIC_TERMS]
        if not tokens:
            continue
        phrase = " ".join(tokens[:3]).strip()
        if phrase and phrase not in other:
            other.append(phrase)
    return other[:8]


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return False
    pattern = re.compile(rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)")
    return bool(pattern.search(text))
