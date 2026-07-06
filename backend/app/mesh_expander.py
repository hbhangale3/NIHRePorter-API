from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import quote

import httpx

from .cache import cache_key, make_cache
from .models import MeshExpansionConfig
from .settings import settings


logger = logging.getLogger(__name__)


class MeshExpander:
    def __init__(self, *, timeout_seconds: float = 10.0) -> None:
        self.timeout_seconds = timeout_seconds
        self.cache = make_cache()
        self.base_url = "https://id.nlm.nih.gov/mesh"

    def expand_keywords(
        self,
        keywords: list[str],
        config: MeshExpansionConfig,
    ) -> tuple[list[str], dict[str, list[str]]]:
        final_terms: list[str] = []
        final_seen: set[str] = set()
        trace: dict[str, list[str]] = {}

        for keyword in keywords:
            self._append_unique(final_terms, final_seen, keyword)
            added_terms = self._expand_single_keyword(keyword, config) if config.enabled else []
            trace[keyword] = added_terms
            for term in added_terms:
                self._append_unique(final_terms, final_seen, term)

        return final_terms, trace

    def _expand_single_keyword(self, keyword: str, config: MeshExpansionConfig) -> list[str]:
        if not keyword.strip():
            return []

        payload = {"keyword": keyword, "config": config.model_dump()}
        lookup_cache_key = cache_key("mesh_expansion", payload)
        if config.cache_enabled:
            cached = self.cache.get(lookup_cache_key)
            if cached is not None:
                return cached

        try:
            descriptor_ids = self._lookup_descriptor_ids(keyword)
            expanded_terms: list[str] = []
            seen_terms: set[str] = set()

            logger.info("MeSH lookup: keyword=%s descriptor_ids=%s", keyword, json.dumps(descriptor_ids))
            for descriptor_id in descriptor_ids:
                details = self._lookup_descriptor_details(descriptor_id)
                self._append_terms_from_details(expanded_terms, seen_terms, details, config)
                if len(expanded_terms) >= config.max_terms_per_keyword:
                    break

            expanded_terms = expanded_terms[: config.max_terms_per_keyword]
            if not expanded_terms and config.fallback_to_original:
                expanded_terms = [keyword]
            logger.info(
                "MeSH expansion result: keyword=%s final_added_terms=%s",
                keyword,
                json.dumps(expanded_terms),
            )
            if config.cache_enabled:
                self.cache.set(lookup_cache_key, expanded_terms, expire=settings.cache_ttl_seconds)
            return expanded_terms
        except Exception as exc:
            logger.warning("MeSH expansion failed for keyword '%s': %s", keyword, exc)
            fallback_terms = [keyword] if config.fallback_to_original else []
            if config.cache_enabled:
                self.cache.set(lookup_cache_key, fallback_terms, expire=settings.cache_ttl_seconds)
            return fallback_terms

    def _lookup_descriptor_ids(self, keyword: str) -> list[str]:
        for match in ("exact", "contains"):
            payload = self._get_json(
                f"{self.base_url}/lookup/term?label={quote(keyword)}&match={match}&limit=5"
            )
            descriptor_ids = self._extract_descriptor_ids(payload)
            if descriptor_ids:
                return descriptor_ids
        return []

    def _lookup_descriptor_details(self, descriptor_id: str) -> dict[str, Any]:
        payload = self._get_json(
            f"{self.base_url}/lookup/details?descriptor={quote(descriptor_id)}"
        )
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected MeSH payload for descriptor {descriptor_id}")
        return payload

    def _get_json(self, url: str) -> Any:
        with httpx.Client(timeout=self.timeout_seconds, headers={"Accept": "application/json"}) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.json()

    def _extract_descriptor_ids(self, payload: Any) -> list[str]:
        if not isinstance(payload, list):
            return []

        descriptor_ids: list[str] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            resource = item.get("resource") or item.get("descriptor") or item.get("id")
            if isinstance(resource, str):
                descriptor_ids.append(resource.rsplit("/", 1)[-1])
        return descriptor_ids

    def _append_terms_from_details(
        self,
        expanded_terms: list[str],
        seen_terms: set[str],
        details: dict[str, Any],
        config: MeshExpansionConfig,
    ) -> None:
        preferred_term = details.get("label")
        entry_terms: list[str] = []
        child_terms: list[str] = []

        self._append_label(expanded_terms, seen_terms, preferred_term)

        if config.include_entry_terms:
            raw_terms = details.get("terms", [])
            if isinstance(raw_terms, list):
                for term in raw_terms:
                    if isinstance(term, dict):
                        label = term.get("label")
                        if isinstance(label, str):
                            entry_terms.append(label)
                        self._append_label(expanded_terms, seen_terms, label)
                    elif isinstance(term, str):
                        entry_terms.append(term)
                        self._append_label(expanded_terms, seen_terms, term)

        if config.include_tree_children and config.max_tree_depth > 0:
            child_ids = self._extract_related_descriptor_ids(details, "narrowerDescriptor")
            for child_id in child_ids:
                try:
                    child_details = self._lookup_descriptor_details(child_id)
                except Exception as exc:
                    logger.info("Skipping MeSH child descriptor '%s': %s", child_id, exc)
                    continue
                child_label = child_details.get("label")
                if isinstance(child_label, str):
                    child_terms.append(child_label)
                self._append_label(expanded_terms, seen_terms, child_label)

        logger.info(
            "MeSH detail trace: preferred_term=%s entry_terms=%s child_terms=%s",
            json.dumps(preferred_term),
            json.dumps(entry_terms),
            json.dumps(child_terms),
        )

    def _extract_related_descriptor_ids(self, details: dict[str, Any], key: str) -> list[str]:
        raw = details.get(key, [])
        if isinstance(raw, dict):
            raw = [raw]
        if not isinstance(raw, list):
            return []

        descriptor_ids: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                resource = item.get("resource") or item.get("descriptor") or item.get("id")
            else:
                resource = item
            if isinstance(resource, str):
                descriptor_ids.append(resource.rsplit("/", 1)[-1])
        return descriptor_ids

    def _append_label(self, expanded_terms: list[str], seen_terms: set[str], label: Any) -> None:
        if isinstance(label, str):
            self._append_unique(expanded_terms, seen_terms, label)

    def _append_unique(self, terms: list[str], seen_terms: set[str], term: str) -> None:
        normalized = term.strip()
        if not normalized:
            return
        lowered = normalized.lower()
        if lowered in seen_terms:
            return
        seen_terms.add(lowered)
        terms.append(normalized)
