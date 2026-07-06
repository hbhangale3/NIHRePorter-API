from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.enrichment.profile_builder import ProfileBuilder
from app.enrichment.profile_models import ProfileSourceResult
from app.enrichment.profile_sources import (
    build_faculty_profile_search_url,
    build_google_scholar_query_url,
    build_orcid_search_url,
    build_pubmed_author_url,
)
from app.models import PIOutreachRow, ProfileEnrichmentConfig


class StaticProfileSource:
    def __init__(self, name: str, results: dict[str, ProfileSourceResult]) -> None:
        self.name = name
        self.results = results

    async def build(self, row: PIOutreachRow) -> ProfileSourceResult:
        return self.results.get(row.pi_name or "", ProfileSourceResult(source=self.name))


class FailingProfileSource:
    def __init__(self, name: str) -> None:
        self.name = name

    async def build(self, row: PIOutreachRow) -> ProfileSourceResult:
        raise RuntimeError(f"{self.name} unavailable")


def _row(name: str, score: int = 85, email: str | None = None) -> PIOutreachRow:
    first, last = name.split(" ", 1)
    return PIOutreachRow(
        pi_name=name,
        pi_first_name=first,
        pi_last_name=last,
        pi_email=email,
        organization_name="New York University",
        organization_city="New York",
        organization_state="NY",
        sample_project_titles=["Artificial intelligence for diabetes care"],
        matched_concepts=["Artificial Intelligence", "Diabetes Mellitus", "Health Equity"],
        matched_dimensions=["AI / Data Science", "Disease / Condition"],
        relevance_score=score,
        relevance_badge="Highly Relevant" if score >= 80 else "Moderately Relevant" if score >= 60 else "Weak Match",
        fiscal_years=[2025, 2026],
        project_ids=[f"APP-{last}"],
        project_numbers=[f"P-{last}"],
        pi_profile_id=f"PI-{last}" if score >= 80 else None,
    )


def test_profile_urls_generated_correctly() -> None:
    row = _row("Devin Mann")

    assert "scholar.google.com" in (build_google_scholar_query_url(row) or "")
    assert "Devin+Mann+New+York+University" in (build_google_scholar_query_url(row) or "")
    assert "pubmed.ncbi.nlm.nih.gov" in (build_pubmed_author_url(row) or "")
    assert "%22Devin+Mann%22%5BAuthor%5D" in (build_pubmed_author_url(row) or "")
    assert "orcid.org/orcid-search/search" in (build_orcid_search_url(row) or "")
    assert "faculty+profile" in (build_faculty_profile_search_url(row) or "")


def test_profile_summary_is_deterministic() -> None:
    builder = ProfileBuilder(ProfileEnrichmentConfig(enabled=True, max_researchers=1))

    enriched_rows, _summary = asyncio.run(builder.enrich_rows([_row("Devin Mann")]))

    assert enriched_rows[0].researcher_profile_summary == (
        "Devin Mann appears to be a highly relevant outreach candidate because their NIH-funded work "
        "aligns with Artificial Intelligence, Diabetes Mellitus, Health Equity. Recent NIH funding through FY 2026 "
        "suggests current activity. Public profile links are available to support outreach follow-up."
    )


def test_profile_builder_respects_max_researchers_limit() -> None:
    rows = [_row("Alice Smith"), _row("Brian Jones", 72), _row("Cara Lopez", 48)]
    builder = ProfileBuilder(ProfileEnrichmentConfig(enabled=True, max_researchers=2))

    enriched_rows, summary = asyncio.run(builder.enrich_rows(rows))

    assert enriched_rows[0].researcher_profile_status in {"enriched", "partial"}
    assert enriched_rows[1].researcher_profile_status in {"enriched", "partial"}
    assert enriched_rows[2].researcher_profile_status == "skipped"
    assert summary["processed"] == 2


def test_disabled_profile_enrichment_preserves_old_behavior() -> None:
    rows = [_row("Dana Reed", 67)]
    builder = ProfileBuilder(ProfileEnrichmentConfig(enabled=False))

    enriched_rows, summary = asyncio.run(builder.enrich_rows(rows))

    assert enriched_rows == rows
    assert summary == {"enabled": 0, "processed": 0}


def test_outreach_recommendation_rules_work() -> None:
    rows = [
        _row("Evan Cole", 92, email="evan.cole@nyu.edu"),
        _row("Frank Hall", 70),
        _row("Grace Kim", 44),
        _row("Hector Lane", 25),
    ]
    builder = ProfileBuilder(ProfileEnrichmentConfig(enabled=True, max_researchers=4))

    enriched_rows, _summary = asyncio.run(builder.enrich_rows(rows))

    assert enriched_rows[0].outreach_recommendation == "priority_contact"
    assert enriched_rows[1].outreach_recommendation == "good_candidate"
    assert enriched_rows[2].outreach_recommendation == "review_manually"
    assert enriched_rows[3].outreach_recommendation == "low_priority"


def test_profile_builder_continues_after_source_failure() -> None:
    rows = [_row("Ivy Park", 88), _row("Jon Mills", 74)]
    source = StaticProfileSource(
        "nih_reporter",
        {
            "Ivy Park": ProfileSourceResult(
                source="nih_reporter",
                urls={"nih_reporter_pi_url": "https://reporter.nih.gov/pi-details/PI-Park"},
                notes=["Direct NIH RePORTER PI profile URL available."],
            ),
            "Jon Mills": ProfileSourceResult(
                source="nih_reporter",
                urls={"nih_reporter_pi_url": "https://reporter.nih.gov/project-details/APP-Mills"},
                notes=["Using NIH RePORTER project page as a profile anchor."],
            ),
        },
    )
    builder = ProfileBuilder(
        ProfileEnrichmentConfig(enabled=True, max_researchers=2, sources=["nih_reporter", "institution_web"]),
        sources=[FailingProfileSource("institution_web"), source],
    )

    enriched_rows, summary = asyncio.run(builder.enrich_rows(rows))

    assert enriched_rows[0].nih_reporter_pi_url == "https://reporter.nih.gov/pi-details/PI-Park"
    assert "institution_web unavailable" in (enriched_rows[0].profile_notes or "")
    assert enriched_rows[1].nih_reporter_pi_url == "https://reporter.nih.gov/project-details/APP-Mills"
    assert summary["processed"] == 2
