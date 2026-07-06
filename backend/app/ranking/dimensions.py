from __future__ import annotations

from dataclasses import dataclass
import re

from ..utils import normalize_text, unique_preserve_order


GENERIC_QUERY_TERMS = {
    "access",
    "algorithm",
    "analytics",
    "care",
    "clinical",
    "community",
    "data",
    "disease",
    "health",
    "healthcare",
    "medicine",
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
    "study",
    "support",
    "system",
    "technology",
    "treatment",
}

AI_FAMILY_TERMS = [
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
    "algorithm",
    "algorithms",
    "data science",
]

POPULATION_FAMILY_TERMS = [
    "health disparities",
    "healthcare disparities",
    "health equity",
    "equity",
    "underserved",
    "minority",
    "minorities",
    "vulnerable population",
    "vulnerable populations",
    "medically underserved",
    "social determinants",
    "rural",
    "access to care",
    "community health",
    "community",
    "access",
]

DIGITAL_HEALTH_FAMILY_TERMS = [
    "telemedicine",
    "telehealth",
    "mobile health",
    "mhealth",
    "remote monitoring",
    "digital health",
    "wearable",
    "wearables",
]

DISEASE_SYNONYM_MAP: dict[str, list[str]] = {
    "diabetes": ["diabetes", "diabetic", "glycemic", "glucose", "insulin", "a1c", "hemoglobin a1c"],
    "obesity": ["obesity", "obese", "body mass index", "bmi", "weight loss", "adiposity"],
    "hypertension": ["hypertension", "hypertensive", "blood pressure"],
    "cancer": ["cancer", "tumor", "tumour", "oncology", "neoplasm"],
    "depression": ["depression", "depressive", "mental health", "mood disorder"],
}


@dataclass(slots=True)
class DimensionFamily:
    key: str
    label: str
    terms: list[str]


def build_dimension_families(
    *,
    research_question: str,
    selected_concepts: list[str],
    final_keywords: list[str],
    mesh_terms: list[str],
    semantic_terms: list[str],
    semantic_concepts: list[str],
) -> list[DimensionFamily]:
    source_terms = unique_preserve_order(
        [research_question]
        + selected_concepts
        + final_keywords
        + mesh_terms
        + semantic_terms
        + semantic_concepts
    )
    normalized_sources = [normalize_text(term) for term in source_terms if normalize_text(term)]
    combined_source_text = " | ".join(normalized_sources)

    families: list[DimensionFamily] = []

    ai_matches = _source_family_matches(combined_source_text, AI_FAMILY_TERMS)
    if ai_matches:
        families.append(
            DimensionFamily(
                key="ai_data_science",
                label="AI / Data Science",
                terms=unique_preserve_order(ai_matches + AI_FAMILY_TERMS[:4]),
            )
        )

    population_matches = _source_family_matches(combined_source_text, POPULATION_FAMILY_TERMS)
    if population_matches:
        families.append(
            DimensionFamily(
                key="population_equity_access",
                label="Population / Equity / Access",
                terms=unique_preserve_order(population_matches + POPULATION_FAMILY_TERMS[:5]),
            )
        )

    digital_matches = _source_family_matches(combined_source_text, DIGITAL_HEALTH_FAMILY_TERMS)
    if digital_matches:
        families.append(
            DimensionFamily(
                key="digital_health",
                label="Digital Health / Telemedicine",
                terms=unique_preserve_order(digital_matches + DIGITAL_HEALTH_FAMILY_TERMS[:4]),
            )
        )

    disease_family = _build_disease_family(source_terms, combined_source_text)
    if disease_family is not None:
        families.append(disease_family)

    if families:
        return families

    fallback_terms = [
        term.strip()
        for term in source_terms
        if term.strip() and len(normalize_text(term)) >= 4
    ]
    if fallback_terms:
        return [
            DimensionFamily(
                key="query_concepts",
                label="Query Concepts",
                terms=unique_preserve_order(fallback_terms[:12]),
            )
        ]
    return []


def match_family_terms(text: str, family: DimensionFamily) -> list[str]:
    normalized_text = normalize_text(text)
    if not normalized_text:
        return []
    return unique_preserve_order(
        [term for term in family.terms if _contains_phrase(normalized_text, term)]
    )


def _source_family_matches(source_text: str, family_terms: list[str]) -> list[str]:
    return [term for term in family_terms if _contains_phrase(source_text, term)]


def _build_disease_family(source_terms: list[str], source_text: str) -> DimensionFamily | None:
    matched_seed: str | None = None
    disease_terms: list[str] = []

    for disease_key, synonyms in DISEASE_SYNONYM_MAP.items():
        if any(_contains_phrase(source_text, synonym) for synonym in synonyms):
            if matched_seed is None:
                matched_seed = disease_key
            disease_terms.extend(synonyms)

    residual_terms = _candidate_disease_terms(source_terms)
    for term in residual_terms:
        disease_terms.append(term)

    disease_terms = unique_preserve_order([term for term in disease_terms if term.strip()])
    if not disease_terms:
        return None

    if matched_seed == "diabetes":
        label = "Diabetes / Metabolic Disease"
    elif matched_seed:
        label = f"{matched_seed.title()} / Condition"
    else:
        label = "Disease / Condition"

    return DimensionFamily(
        key="disease_condition",
        label=label,
        terms=disease_terms[:16],
    )


def _candidate_disease_terms(source_terms: list[str]) -> list[str]:
    candidates: list[str] = []
    for term in source_terms:
        normalized = normalize_text(term)
        if not normalized:
            continue
        if normalized in AI_FAMILY_TERMS or normalized in POPULATION_FAMILY_TERMS or normalized in DIGITAL_HEALTH_FAMILY_TERMS:
            continue
        if any(_contains_phrase(normalized, family_term) for family_term in AI_FAMILY_TERMS + POPULATION_FAMILY_TERMS + DIGITAL_HEALTH_FAMILY_TERMS):
            continue

        tokens = [token for token in re.findall(r"[a-z0-9]+", normalized) if token not in GENERIC_QUERY_TERMS]
        if not tokens:
            continue
        phrase = " ".join(tokens[:4]).strip()
        if len(phrase) < 4:
            continue
        candidates.append(phrase)
        candidates.extend(token for token in tokens if len(token) >= 4)
    return unique_preserve_order(candidates)


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return False
    pattern = re.compile(rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)")
    return bool(pattern.search(text))
