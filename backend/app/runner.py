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


logger = logging.getLogger(__name__)


def _build_stage1_criteria(config: AppConfig, expanded_keywords: list[str] | None = None) -> dict[str, Any]:
    criteria: dict[str, Any] = {}

    if config.query.fiscal_years:
        criteria["fiscal_years"] = config.query.fiscal_years

    keywords_to_use = expanded_keywords if expanded_keywords else config.query.broad_keywords
    
    if keywords_to_use:
        # NIH RePORTER API v2 uses "advanced_text_search" (not "text_search")
        # and "operator" (not "search_text_operator")
        criteria["advanced_text_search"] = {
            "search_text": " ".join(keywords_to_use),
            "search_field": config.query.text_search_field,
            "operator": config.query.text_search_operator,
        }

    return criteria


async def run_pipeline_async(
    config_yaml: str,
    *,
    max_pages: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[str]], dict[str, Any]]:
    config = load_config_from_yaml_str(config_yaml)

    original_keywords = list(config.query.broad_keywords)
    current_keywords = list(original_keywords)
    mesh_trace: dict[str, list[str]] = {}
    keyword_expansions: dict[str, list[str]] = {}

    if current_keywords and config.query.mesh_expansion.enabled:
        mesh_expander = MeshExpander()
        current_keywords, mesh_trace = mesh_expander.expand_keywords(
            current_keywords,
            config.query.mesh_expansion,
        )
    logger.info("Keyword pipeline: original YAML keywords=%s", json.dumps(original_keywords))
    logger.info("Keyword pipeline: after MeSH expansion=%s", json.dumps(current_keywords))

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

    rows, summary = build_outreach_rows(projects, config)
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
