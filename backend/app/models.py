from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class TopicConfig(BaseModel):
    name: str
    include_any: list[str] = Field(default_factory=list)
    include_all: list[str] = Field(default_factory=list)
    exclude_any: list[str] = Field(default_factory=list)
    co_require_groups: list[list[str]] = Field(default_factory=list)


class AIExpansionConfig(BaseModel):
    enabled: bool = False
    openai_api_key: str | None = None
    model: str = "gpt-4o-mini"
    max_expansions_per_keyword: int = 5
    context: str = "biomedical research and health disparities"


class MeshExpansionConfig(BaseModel):
    enabled: bool = False
    max_terms_per_keyword: int = 15
    include_entry_terms: bool = True
    include_tree_children: bool = True
    max_tree_depth: int = 1
    fallback_to_original: bool = True
    cache_enabled: bool = True


class SemanticExpansionConfig(BaseModel):
    enabled: bool = False
    top_k: int = 10
    max_terms: int = 30
    min_score: float | None = None
    include_synonyms: bool = True
    require_existing_index: bool = False


class MultiQueryRetrievalConfig(BaseModel):
    enabled: bool = False
    max_queries: int = 8
    pages_per_query: int = 1
    require_dimension_overlap: bool = True
    include_original_query: bool = True


class EmailEnrichmentConfig(BaseModel):
    enabled: bool = False
    max_researchers: int = 50
    sources: list[str] = Field(default_factory=lambda: ["institution_web", "pubmed", "orcid"])
    timeout_seconds: int = 10
    max_pages_per_researcher: int = 3
    require_high_confidence: bool = False


class QueryConfig(BaseModel):
    research_question: str | None = None
    fiscal_years: list[int] = Field(default_factory=list)
    broad_keywords: list[str] = Field(default_factory=list)
    text_search_field: str = "all"
    text_search_operator: Literal["and", "or"] | None = None
    mesh_expansion: MeshExpansionConfig = Field(default_factory=MeshExpansionConfig)
    semantic_expansion: SemanticExpansionConfig = Field(default_factory=SemanticExpansionConfig)
    ai_expansion: AIExpansionConfig = Field(default_factory=AIExpansionConfig)
    multi_query_retrieval: MultiQueryRetrievalConfig = Field(default_factory=MultiQueryRetrievalConfig)
    email_enrichment: EmailEnrichmentConfig = Field(default_factory=EmailEnrichmentConfig)


class AppConfig(BaseModel):
    query: QueryConfig
    topics: list[TopicConfig]


class RunRequest(BaseModel):
    config_yaml: str
    max_pages: int | None = None

    @field_validator("max_pages")
    @classmethod
    def cap_max_pages(cls, v: int | None) -> int | None:
        if v is not None and v > 200:
            return 200
        return v


class RunStatus(BaseModel):
    run_id: str
    status: Literal["queued", "running", "completed", "failed"]
    message: str | None = None
    progress: dict[str, Any] = Field(default_factory=dict)
    keyword_expansions: dict[str, list[str]] | None = None
    expansion_trace: dict[str, Any] | None = None
    retrieval_trace: dict[str, Any] | None = None


class PIOutreachRow(BaseModel):
    pi_name: str | None = None
    pi_first_name: str | None = None
    pi_last_name: str | None = None
    pi_email: str | None = None
    email_confidence: str | None = None
    email_source: str | None = None
    email_source_url: str | None = None
    email_status: str | None = None
    email_notes: str | None = None

    organization_name: str | None = None
    organization_city: str | None = None
    organization_state: str | None = None
    organization_country: str | None = None

    admin_ic: str | None = None
    fiscal_years: list[int] = Field(default_factory=list)
    project_count: int = 0

    matched_topics: list[str] = Field(default_factory=list)
    sample_project_titles: list[str] = Field(default_factory=list)
    project_numbers: list[str] = Field(default_factory=list)
    project_abstracts: list[str] = Field(default_factory=list)
    project_terms: list[str] = Field(default_factory=list)

    project_ids: list[str] = Field(default_factory=list)
    project_urls: list[str] = Field(default_factory=list)

    pi_profile_id: str | None = None
    total_funding_amount: float | None = None
    project_start_date: str | None = None
    project_end_date: str | None = None
    retrieval_query_matches: list[str] = Field(default_factory=list)
    retrieval_query_reasons: list[str] = Field(default_factory=list)
    retrieved_by: list[str] = Field(default_factory=list)

    relevance_score: int = 0
    matched_dimensions: list[str] = Field(default_factory=list)
    missing_dimensions: list[str] = Field(default_factory=list)
    dimension_match_count: int = 0
    dimension_coverage_ratio: float = 0.0
    matched_concepts: list[str] = Field(default_factory=list)
    missing_concepts: list[str] = Field(default_factory=list)
    reasoning: str | None = None
    semantic_similarity: float | None = None
    mesh_matches: list[str] = Field(default_factory=list)
    ai_match: bool = False
    disease_match: bool = False
    population_match: bool = False
    relevance_badge: str = "Low Match"
    score_breakdown: dict[str, int] = Field(default_factory=dict)
    score_breakdown_entries: list[dict[str, Any]] = Field(default_factory=list)
    score_breakdown_text: str | None = None
