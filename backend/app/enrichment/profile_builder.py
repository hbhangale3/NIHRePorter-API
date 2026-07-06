from __future__ import annotations

from collections import Counter
import logging

from ..models import PIOutreachRow, ProfileEnrichmentConfig
from ..utils import unique_preserve_order
from .profile_models import ProfileSourceResult
from .profile_sources import (
    SOURCE_FACTORIES,
    ProfileSource,
    build_google_scholar_query_url,
    has_verified_nih_reporter_pi_url,
)


logger = logging.getLogger(__name__)


class ProfileBuilder:
    def __init__(
        self,
        config: ProfileEnrichmentConfig,
        *,
        sources: list[ProfileSource] | None = None,
    ) -> None:
        self.config = config
        self._provided_sources = sources

    async def enrich_rows(self, rows: list[PIOutreachRow]) -> tuple[list[PIOutreachRow], dict[str, int]]:
        if not self.config.enabled:
            return rows, {"enabled": 0, "processed": 0}

        if not rows:
            return rows, {"enabled": 1, "processed": 0}

        updated_rows = list(rows)
        target_count = max(0, min(self.config.max_researchers, len(updated_rows)))
        counters = Counter()
        sources = self._provided_sources or self._build_sources()

        for index, row in enumerate(updated_rows):
            if index >= target_count:
                if not row.researcher_profile_status:
                    updated_rows[index] = row.model_copy(
                        update={
                            "researcher_profile_status": "skipped",
                            "profile_notes": "Skipped because researcher rank was outside the profile-enrichment limit.",
                        }
                    )
                counters["skipped"] += 1
                continue

            enriched_row = await self._enrich_single_row(row, sources)
            updated_rows[index] = enriched_row
            counters[enriched_row.researcher_profile_status or "unknown"] += 1

        counters["enabled"] = 1
        counters["processed"] = target_count
        return updated_rows, dict(counters)

    async def _enrich_single_row(
        self,
        row: PIOutreachRow,
        sources: list[ProfileSource],
    ) -> PIOutreachRow:
        if not (row.pi_name or row.pi_profile_id or row.project_ids):
            return row.model_copy(
                update={
                    "researcher_profile_status": "not_found",
                    "profile_notes": "Profile enrichment could not run because the researcher identity was incomplete.",
                    "outreach_recommendation": _recommendation_for_row(row, has_profile_links=False),
                }
            )

        notes: list[str] = []
        had_error = False
        url_updates: dict[str, str] = {}

        for source in sources:
            try:
                result = await source.build(row)
            except Exception as exc:
                logger.warning("Profile enrichment source %s failed for %s: %s", source.name, row.pi_name, exc)
                notes.append(f"{source.name}: {exc}")
                had_error = True
                continue

            _merge_result(url_updates, result)
            notes.extend(result.notes)

        scholar_url = build_google_scholar_query_url(row)
        if scholar_url and "google_scholar_query_url" not in url_updates:
            url_updates["google_scholar_query_url"] = scholar_url
            notes.append("Google Scholar search URL generated.")

        profile_source_urls = unique_preserve_order(
            [
                url_updates.get("faculty_profile_url"),
                url_updates.get("orcid_url"),
                url_updates.get("pubmed_author_url"),
                url_updates.get("nih_reporter_pi_url"),
                url_updates.get("google_scholar_query_url"),
            ]
        )
        profile_source_urls = [url for url in profile_source_urls if url]
        confidence = _profile_confidence(row, url_updates)
        status = _profile_status(row, url_updates, had_error)
        recommendation = _recommendation_for_row(row, has_profile_links=bool(profile_source_urls))
        summary = _profile_summary(row, has_profile_links=bool(profile_source_urls))
        if not row.pi_email:
            notes.append("Email discovery is best effort and may not find public emails.")

        return row.model_copy(
            update={
                **url_updates,
                "researcher_profile_status": status,
                "researcher_profile_summary": summary,
                "researcher_profile_confidence": confidence,
                "profile_source_urls": profile_source_urls,
                "profile_notes": " ".join(notes).strip() or None,
                "outreach_recommendation": recommendation,
            }
        )

    def _build_sources(self) -> list[ProfileSource]:
        sources: list[ProfileSource] = []
        for source_name in self.config.sources:
            factory = SOURCE_FACTORIES.get(source_name)
            if factory is None:
                logger.info("Skipping unknown profile enrichment source: %s", source_name)
                continue
            sources.append(factory())
        return sources


def _merge_result(url_updates: dict[str, str], result: ProfileSourceResult) -> None:
    for key, value in result.urls.items():
        if value and not url_updates.get(key):
            url_updates[key] = value


def _profile_status(row: PIOutreachRow, url_updates: dict[str, str], had_error: bool) -> str:
    profile_url_count = len([value for value in url_updates.values() if value])
    if has_verified_nih_reporter_pi_url(row, url_updates.get("nih_reporter_pi_url")):
        return "verified_profile_link"
    if profile_url_count >= 3:
        return "search_links_generated"
    if profile_url_count >= 1:
        return "partial"
    if had_error:
        return "error"
    return "not_found"


def _profile_confidence(row: PIOutreachRow, url_updates: dict[str, str]) -> str:
    profile_url_count = len([value for value in url_updates.values() if value])
    if has_verified_nih_reporter_pi_url(row, url_updates.get("nih_reporter_pi_url")):
        return "verified"
    if profile_url_count >= 1 and row.organization_name:
        return "search_only" if profile_url_count >= 3 else "likely"
    return "not_found"


def _recommendation_for_row(row: PIOutreachRow, *, has_profile_links: bool) -> str:
    score = int(row.relevance_score or 0)
    has_relevant_concepts = bool(row.matched_concepts or row.mesh_matches or row.matched_dimensions)
    if score >= 80 and (row.pi_email or has_profile_links):
        return "priority_contact"
    if score >= 60 and has_profile_links:
        return "good_candidate"
    if score >= 40 and has_relevant_concepts:
        return "review_manually"
    return "low_priority"


def _profile_summary(row: PIOutreachRow, *, has_profile_links: bool) -> str | None:
    if not (row.pi_name or row.organization_name or row.sample_project_titles):
        return None

    top_concepts = list(row.matched_concepts[:3]) or list(row.mesh_matches[:3]) or list(row.matched_dimensions[:2])
    concept_text = ", ".join(top_concepts) if top_concepts else "the selected research themes"
    latest_fy = max(row.fiscal_years) if row.fiscal_years else None
    funding_sentence = (
        f" Recent NIH funding through FY {latest_fy} suggests current activity."
        if latest_fy is not None
        else ""
    )
    contact_sentence = (
        " Public search/profile links are available for manual outreach follow-up."
        if has_profile_links
        else ""
    )
    return (
        f"{row.pi_name or 'This researcher'} appears to be a {row.relevance_badge.lower()} outreach candidate "
        f"because their NIH-funded work aligns with {concept_text}.{funding_sentence}{contact_sentence}"
    ).strip()
