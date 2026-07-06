from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.enrichment.email_enricher import EmailEnricher
from app.enrichment.email_utils import (
    EmailCandidate,
    SourceLookupResult,
    choose_best_candidate,
    extract_emails,
    score_email_candidate,
)
from app.models import EmailEnrichmentConfig, PIOutreachRow


class StaticSource:
    def __init__(self, name: str, results: dict[str, SourceLookupResult]) -> None:
        self.name = name
        self.results = results

    async def lookup(self, row: PIOutreachRow) -> SourceLookupResult:
        return self.results.get(
            row.pi_name or "",
            SourceLookupResult(source=self.name, status="not_found", notes="No match."),
        )


class FailingSource:
    def __init__(self, name: str) -> None:
        self.name = name

    async def lookup(self, row: PIOutreachRow) -> SourceLookupResult:
        raise RuntimeError(f"{self.name} unavailable")


def _row(name: str, score: int = 80, email: str | None = None) -> PIOutreachRow:
    first, last = name.split(" ", 1)
    return PIOutreachRow(
        pi_name=name,
        pi_first_name=first,
        pi_last_name=last,
        pi_email=email,
        organization_name="New York University",
        sample_project_titles=["Artificial intelligence for diabetes care"],
        relevance_score=score,
        project_numbers=[f"P-{last}"],
        project_ids=[f"P-{last}"],
    )


def test_extract_emails_deduplicates_and_normalizes() -> None:
    emails = extract_emails("Contact DEVIN.MANN@NYU.EDU, devin.mann@nyu.edu, or info@nyu.edu.")

    assert emails == ["devin.mann@nyu.edu", "info@nyu.edu"]


def test_choose_best_candidate_prefers_non_generic_email() -> None:
    generic = EmailCandidate(
        email="info@nyu.edu",
        confidence="medium",
        source="institution_web",
    )
    direct = EmailCandidate(
        email="devin.mann@nyu.edu",
        confidence="medium",
        source="pubmed",
    )

    best = choose_best_candidate([generic, direct])

    assert best is direct


def test_score_email_candidate_assigns_confidence_levels() -> None:
    high = score_email_candidate(
        email="devin.mann@nyu.edu",
        source="institution_web",
        source_url="https://med.nyu.edu/faculty/devin-mann",
        page_text="Devin Mann faculty profile Devin Mann devin.mann@nyu.edu",
        pi_name="Devin Mann",
        pi_first_name="Devin",
        pi_last_name="Mann",
        organization_name="New York University",
    )
    medium = score_email_candidate(
        email="mann@nyu.edu",
        source="pubmed",
        source_url="https://pubmed.ncbi.nlm.nih.gov/123456/",
        page_text="Mann D. Department of Medicine, New York University.",
        pi_name="Devin Mann",
        pi_first_name="Devin",
        pi_last_name="Mann",
        organization_name="New York University",
    )
    low = score_email_candidate(
        email="info@research.org",
        source="pubmed",
        source_url="https://pubmed.ncbi.nlm.nih.gov/123456/",
        page_text="General study contact info@research.org",
        pi_name="Devin Mann",
        pi_first_name="Devin",
        pi_last_name="Mann",
        organization_name="New York University",
    )

    assert high.confidence == "high"
    assert medium.confidence == "medium"
    assert low.confidence == "low"


def test_enrichment_continues_after_failure_and_respects_max_researchers() -> None:
    rows = [
        _row("Alice Smith", 95),
        _row("Brian Jones", 88),
        _row("Cara Lopez", 77),
    ]
    pubmed_source = StaticSource(
        "pubmed",
        {
            "Alice Smith": SourceLookupResult(
                source="pubmed",
                status="candidate",
                candidate=EmailCandidate(
                    email="alice.smith@nyu.edu",
                    confidence="high",
                    source="pubmed",
                    source_url="https://pubmed.ncbi.nlm.nih.gov/1/",
                    notes="Exact author match.",
                ),
            ),
            "Brian Jones": SourceLookupResult(
                source="pubmed",
                status="not_found",
                notes="No publication email found.",
            ),
        },
    )
    enricher = EmailEnricher(
        EmailEnrichmentConfig(enabled=True, max_researchers=2, sources=["institution_web", "pubmed"]),
        sources=[FailingSource("institution_web"), pubmed_source],
    )

    enriched_rows, summary = asyncio.run(enricher.enrich_rows(rows))

    assert enriched_rows[0].pi_email == "alice.smith@nyu.edu"
    assert enriched_rows[0].email_status == "found_high_confidence"
    assert enriched_rows[1].email_status == "not_found"
    assert "institution_web unavailable" in (enriched_rows[1].email_notes or "")
    assert enriched_rows[2].email_status == "skipped"
    assert summary["processed"] == 2


def test_disabled_enrichment_preserves_existing_rows() -> None:
    rows = [_row("Dana Reed", 82)]
    enricher = EmailEnricher(EmailEnrichmentConfig(enabled=False))

    enriched_rows, summary = asyncio.run(enricher.enrich_rows(rows))

    assert enriched_rows == rows
    assert summary == {"enabled": 0, "processed": 0}


def test_existing_pi_email_is_not_overwritten() -> None:
    rows = [_row("Evan Cole", 91, email="evan.cole@nyu.edu")]
    enricher = EmailEnricher(
        EmailEnrichmentConfig(enabled=True, max_researchers=1),
        sources=[FailingSource("pubmed")],
    )

    enriched_rows, _summary = asyncio.run(enricher.enrich_rows(rows))

    assert enriched_rows[0].pi_email == "evan.cole@nyu.edu"
    assert enriched_rows[0].email_source == "nih_reporter"
    assert enriched_rows[0].email_status == "found_high_confidence"
