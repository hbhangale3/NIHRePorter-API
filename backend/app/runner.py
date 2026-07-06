from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from .config_loader import load_config_from_yaml_str
from .models import AppConfig
from .reporter_client import ReporterClient
from .processor import build_outreach_rows
from .keyword_expander import KeywordExpander
from .mesh_expander import MeshExpander
from .ranking import OutreachRankingContext, OutreachRankingScorer
from .semantic import MeshSemanticRetriever


logger = logging.getLogger(__name__)


def _deduplicate_terms_case_insensitive(terms: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        normalized = term.strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(normalized)
    return deduped


def format_search_terms_for_nih(terms: list[str]) -> str:
    formatted_terms: list[str] = []
    for term in _deduplicate_terms_case_insensitive(terms):
        escaped = term.replace('"', '\\"')
        if any(char.isspace() for char in escaped):
            formatted_terms.append(f'"{escaped}"')
        else:
            formatted_terms.append(escaped)
    return " ".join(formatted_terms)


def _resolve_text_search_operator(config: AppConfig) -> str:
    if config.query.text_search_operator is not None:
        return config.query.text_search_operator
    if config.query.mesh_expansion.enabled or config.query.semantic_expansion.enabled:
        return "or"
    return "and"


def _build_stage1_criteria(config: AppConfig, expanded_keywords: list[str] | None = None) -> dict[str, Any]:
    criteria: dict[str, Any] = {}

    if config.query.fiscal_years:
        criteria["fiscal_years"] = config.query.fiscal_years

    keywords_to_use = expanded_keywords if expanded_keywords else config.query.broad_keywords

    if keywords_to_use:
        operator = _resolve_text_search_operator(config)
        criteria["advanced_text_search"] = {
            "search_text": format_search_terms_for_nih(keywords_to_use),
            "search_field": config.query.text_search_field,
            "operator": operator,
        }

    return criteria


def _resolve_research_question(config: AppConfig, original_keywords: list[str]) -> str:
    if config.query.research_question and config.query.research_question.strip():
        return config.query.research_question.strip()
    if config.topics:
        topic_name = config.topics[0].name.strip()
        if topic_name:
            return topic_name
    return " ".join(original_keywords).strip()


async def run_pipeline_async(
    config_yaml: str,
    *,
    max_pages: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[str]], dict[str, Any]]:
    config = load_config_from_yaml_str(config_yaml)

    original_keywords = list(config.query.broad_keywords)
    research_question = _resolve_research_question(config, original_keywords)
    current_keywords = _deduplicate_terms_case_insensitive(original_keywords)
    mesh_trace: dict[str, list[str]] = {}
    keyword_expansions: dict[str, list[str]] = {}
    semantic_trace: dict[str, Any] = {
        "enabled": config.query.semantic_expansion.enabled,
        "query": None,
        "concepts": [],
        "expanded_terms": [],
        "error": None,
    }

    if current_keywords and config.query.mesh_expansion.enabled:
        mesh_expander = MeshExpander()
        current_keywords, mesh_trace = mesh_expander.expand_keywords(
            current_keywords,
            config.query.mesh_expansion,
        )
        current_keywords = _deduplicate_terms_case_insensitive(current_keywords)
    logger.info("Keyword pipeline: original YAML keywords=%s", json.dumps(original_keywords))
    logger.info("Keyword pipeline: after MeSH expansion=%s", json.dumps(current_keywords))

    if current_keywords and config.query.semantic_expansion.enabled:
        semantic_query = " ".join(_deduplicate_terms_case_insensitive(original_keywords))
        semantic_trace["query"] = semantic_query
        try:
            retriever = MeshSemanticRetriever()
            semantic_result = retriever.expand_query(
                query=semantic_query,
                top_k=config.query.semantic_expansion.top_k,
                include_synonyms=config.query.semantic_expansion.include_synonyms,
                max_terms=config.query.semantic_expansion.max_terms,
                min_score=config.query.semantic_expansion.min_score,
            )
            semantic_trace["concepts"] = semantic_result["semantic_concepts"]
            semantic_trace["expanded_terms"] = semantic_result["expanded_terms"]
            current_keywords = _deduplicate_terms_case_insensitive(
                current_keywords + list(semantic_result["expanded_terms"])
            )
            logger.info(
                "Keyword pipeline: after semantic expansion=%s",
                json.dumps(current_keywords),
            )
        except Exception as exc:
            semantic_trace["error"] = str(exc)
            if config.query.semantic_expansion.require_existing_index:
                raise RuntimeError(
                    "Semantic MeSH expansion was enabled and require_existing_index=true, "
                    f"but retrieval failed: {exc}"
                ) from exc
            logger.warning("Semantic MeSH expansion unavailable; continuing without semantic terms: %s", exc)

    if current_keywords and config.query.ai_expansion.enabled:
        expander = KeywordExpander(
            api_key=config.query.ai_expansion.openai_api_key,
            model=config.query.ai_expansion.model,
        )
        current_keywords, keyword_expansions = expander.expand_query_keywords(
            current_keywords,
            enabled=True,
            context=config.query.ai_expansion.context,
            max_expansions=config.query.ai_expansion.max_expansions_per_keyword,
        )
        current_keywords = _deduplicate_terms_case_insensitive(current_keywords)
    logger.info("Keyword pipeline: after AI expansion=%s", json.dumps(current_keywords))

    criteria = _build_stage1_criteria(config, current_keywords)
    logger.info(
        "Keyword pipeline: final keywords passed to NIH query builder=%s",
        json.dumps(current_keywords),
    )
    logger.info("NIH criteria payload=%s", json.dumps(criteria, sort_keys=True))

    # NIH RePORTER API v2 requires PascalCase field names in include_fields
    include_fields = [
        "ApplId",
        "CoreProjectNum",      # stable year-invariant project identifier
        "ProjectTitle",
        "AbstractText",
        "FiscalYear",
        "AgencyIcAdmin",       # returns agency_ic_admin.abbreviation (the IC code)
        "PrincipalInvestigators",
        "Organization",
        "AwardAmount",
        "ProjectNum",
        "ProjectStartDate",
        "ProjectEndDate",
        "Terms",
    ]

    client = ReporterClient()
    try:
        projects = await client.fetch_all_projects(criteria, include_fields=include_fields, max_pages=max_pages)
    finally:
        await client.aclose()
    logger.info("NIH projects returned before local filtering=%d", len(projects))
    logger.info("Semantic terms available to run=%s", bool(semantic_trace["expanded_terms"]))

    rows, summary = build_outreach_rows(projects, config)
    ranking_context = OutreachRankingContext(
        research_question=research_question,
        expanded_terms=current_keywords,
    )
    ranking_scorer = OutreachRankingScorer()
    rows, ranking_summary = ranking_scorer.rank_rows(rows, ranking_context)
    summary["ranking"] = ranking_summary
    logger.info(
        "NIH projects after local filtering=%d; unique outreach rows=%d",
        summary.get("matched_project_count", 0),
        len(rows),
    )

    # convert to json-serializable dicts
    results = [r.model_dump() for r in rows]
    expansion_trace = {
        "original_keywords": original_keywords,
        "mesh": {
            "enabled": config.query.mesh_expansion.enabled,
            "terms_by_keyword": mesh_trace,
        },
        "semantic": semantic_trace,
        "ai": {
            "enabled": config.query.ai_expansion.enabled,
            "terms_by_keyword": keyword_expansions,
        },
        "final_keywords": current_keywords,
    }
    return results, summary, keyword_expansions, expansion_trace


def run_pipeline(
    config_yaml: str,
    *,
    max_pages: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[str]], dict[str, Any]]:
    return asyncio.run(run_pipeline_async(config_yaml, max_pages=max_pages))
