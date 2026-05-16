from __future__ import annotations

import asyncio
from typing import Any

from .config_loader import load_config_from_yaml_str
from .models import AppConfig
from .reporter_client import ReporterClient
from .processor import build_outreach_rows
from .keyword_expander import KeywordExpander


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


async def run_pipeline_async(config_yaml: str, *, max_pages: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[str]]]:
    config = load_config_from_yaml_str(config_yaml)

    # AI keyword expansion
    keyword_expansions = {}
    expanded_keywords = None
    
    if config.query.broad_keywords and config.query.ai_expansion.enabled:
        expander = KeywordExpander(
            api_key=config.query.ai_expansion.openai_api_key,
            model=config.query.ai_expansion.model
        )
        expanded_keywords, keyword_expansions = expander.expand_query_keywords(
            config.query.broad_keywords,
            enabled=True,
            context=config.query.ai_expansion.context,
            max_expansions=config.query.ai_expansion.max_expansions_per_keyword
        )
    
    criteria = _build_stage1_criteria(config, expanded_keywords)

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

    rows, summary = build_outreach_rows(projects, config)

    # convert to json-serializable dicts
    results = [r.model_dump() for r in rows]
    return results, summary, keyword_expansions


def run_pipeline(config_yaml: str, *, max_pages: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[str]]]:
    return asyncio.run(run_pipeline_async(config_yaml, max_pages=max_pages))
