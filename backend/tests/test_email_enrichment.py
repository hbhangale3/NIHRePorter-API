from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.enrichment.email_enricher import EmailEnricher
from app.enrichment.email_utils import (
    EmailCandidate,
    SourceLookupResult,
    choose_best_candidate,
    choose_best_rejected_candidate,
    email_local_part_matches_pi_name,
    extract_emails,
    score_email_candidate,
)
from app.enrichment.pubmed_email import PubMedEmailLookup
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


class SequenceAsyncClient:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict | None]] = []

    async def get(self, url: str, *, params=None, timeout=None):  # noqa: ANN001
        self.calls.append((url, params))
        if not self.responses:
            raise AssertionError("No more fake responses available.")
        return self.responses.pop(0)


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
    assert low.confidence == "withheld"


def test_email_local_part_matching_is_precision_first() -> None:
    assert (
        email_local_part_matches_pi_name(
            "rsayaman@ucsf.edu",
            pi_first_name="Rosalyn",
            pi_last_name="Sayaman",
            pi_name="Rosalyn Sayaman",
        )
        == "strong"
    )
    assert (
        email_local_part_matches_pi_name(
            "rosalyn.sayaman@ucsf.edu",
            pi_first_name="Rosalyn",
            pi_last_name="Sayaman",
            pi_name="Rosalyn Sayaman",
        )
        == "strong"
    )
    assert (
        email_local_part_matches_pi_name(
            "dbergles@jhmi.edu",
            pi_first_name="Jeremias",
            pi_last_name="Sulam",
            pi_name="Jeremias Sulam",
        )
        == "none"
    )


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


def test_pubmed_429_triggers_retry_and_backoff() -> None:
    PubMedEmailLookup.reset_rate_limit_state()
    request = httpx.Request("GET", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi")
    client = SequenceAsyncClient(
        [
            httpx.Response(429, request=request),
            httpx.Response(429, request=request),
            httpx.Response(200, request=request, json={"esearchresult": {"idlist": []}}),
            httpx.Response(200, request=request, json={"esearchresult": {"idlist": []}}),
            httpx.Response(200, request=request, json={"esearchresult": {"idlist": []}}),
        ]
    )
    sleeps: list[float] = []
    lookup = PubMedEmailLookup(
        client=client,
        sleep_func=lambda seconds: sleeps.append(seconds) or asyncio.sleep(0),
        min_request_interval_seconds=0,
    )

    result = asyncio.run(lookup.lookup(_row("Dana Reed")))

    assert result.status == "not_found"
    assert sleeps[:2] == [1, 2]
    assert len(client.calls) == 5


def test_pubmed_lookup_uses_cache_for_repeated_pi() -> None:
    PubMedEmailLookup.reset_rate_limit_state()
    request = httpx.Request("GET", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi")
    client = SequenceAsyncClient(
        [
            httpx.Response(200, request=request, json={"esearchresult": {"idlist": []}}),
            httpx.Response(200, request=request, json={"esearchresult": {"idlist": []}}),
            httpx.Response(200, request=request, json={"esearchresult": {"idlist": []}}),
        ]
    )
    lookup = PubMedEmailLookup(client=client, min_request_interval_seconds=0)
    row = _row("Dana Reed")

    first = asyncio.run(lookup.lookup(row))
    second = asyncio.run(lookup.lookup(row))

    assert first.status == "not_found"
    assert second.status == "not_found"
    assert len(client.calls) == 3


def test_pubmed_query_strategy_does_not_require_title() -> None:
    PubMedEmailLookup.reset_rate_limit_state()
    request = httpx.Request("GET", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi")
    client = SequenceAsyncClient(
        [
            httpx.Response(200, request=request, json={"esearchresult": {"idlist": []}}),
            httpx.Response(200, request=request, json={"esearchresult": {"idlist": []}}),
            httpx.Response(200, request=request, json={"esearchresult": {"idlist": []}}),
        ]
    )
    row = _row("Rosalyn Sayaman")
    row.sample_project_titles = []
    lookup = PubMedEmailLookup(client=client, min_request_interval_seconds=0)

    result = asyncio.run(lookup.lookup(row))

    terms = [params["term"] for _url, params in client.calls]
    assert result.status == "not_found"
    assert terms[0] == "Sayaman R[Author]"
    assert terms[1] == "\"Rosalyn Sayaman\"[Author]"
    assert "Title" not in " ".join(terms)


def test_pubmed_failure_returns_error_without_crashing_enrichment() -> None:
    PubMedEmailLookup.reset_rate_limit_state()
    request = httpx.Request("GET", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi")
    client = SequenceAsyncClient(
        [
            httpx.Response(429, request=request),
            httpx.Response(429, request=request),
            httpx.Response(429, request=request),
            httpx.Response(429, request=request),
        ]
    )
    lookup = PubMedEmailLookup(client=client, min_request_interval_seconds=0, sleep_func=lambda _s: asyncio.sleep(0))
    enricher = EmailEnricher(
        EmailEnrichmentConfig(enabled=True, max_researchers=1, sources=["pubmed"]),
        sources=[lookup],
    )

    enriched_rows, summary = asyncio.run(enricher.enrich_rows([_row("Ivy Park")]))

    assert enriched_rows[0].email_status == "error"
    assert "status 429" in (enriched_rows[0].email_notes or "")
    assert summary["processed"] == 1


def test_choose_best_candidate_prefers_direct_name_match_over_wrong_person() -> None:
    wrong = score_email_candidate(
        email="dbergles@jhmi.edu",
        source="pubmed",
        source_url="https://pubmed.ncbi.nlm.nih.gov/1/",
        page_text="Jeremias Sulam Department of Biomedical Engineering dbergles@jhmi.edu",
        pi_name="Jeremias Sulam",
        pi_first_name="Jeremias",
        pi_last_name="Sulam",
        organization_name="Johns Hopkins University",
    )
    right = score_email_candidate(
        email="jsulam@jhu.edu",
        source="pubmed",
        source_url="https://pubmed.ncbi.nlm.nih.gov/2/",
        page_text="Jeremias Sulam jsulam@jhu.edu",
        pi_name="Jeremias Sulam",
        pi_first_name="Jeremias",
        pi_last_name="Sulam",
        organization_name="Johns Hopkins University",
    )

    best = choose_best_candidate([wrong, right])
    rejected = choose_best_rejected_candidate([wrong, right])

    assert best is not None
    assert best.email == "jsulam@jhu.edu"
    assert rejected is not None
    assert rejected.email == "dbergles@jhmi.edu"


def test_wrong_person_email_is_withheld_for_manual_review() -> None:
    rows = [_row("Jeremias Sulam", 95)]
    pubmed_source = StaticSource(
        "pubmed",
        {
            "Jeremias Sulam": SourceLookupResult(
                source="pubmed",
                status="manual_review_required",
                rejected_candidate=EmailCandidate(
                    email="dbergles@jhmi.edu",
                    confidence="withheld",
                    source="pubmed",
                    source_url="https://pubmed.ncbi.nlm.nih.gov/1/",
                    notes="PubMed found an email, but the local-part did not clearly match the target PI name.",
                    rejection_reason="Email local-part does not match PI Jeremias Sulam.",
                ),
                notes="Email local-part does not match PI Jeremias Sulam.",
            ),
        },
    )
    enricher = EmailEnricher(
        EmailEnrichmentConfig(enabled=True, max_researchers=1, sources=["pubmed"]),
        sources=[pubmed_source],
    )

    enriched_rows, summary = asyncio.run(enricher.enrich_rows(rows))

    assert enriched_rows[0].pi_email is None
    assert enriched_rows[0].email_status == "manual_review_required"
    assert enriched_rows[0].email_confidence == "withheld"
    assert enriched_rows[0].email_candidate_rejected == "dbergles@jhmi.edu"
    assert enriched_rows[0].email_rejection_reason == "Email local-part does not match PI Jeremias Sulam."
    assert summary["processed"] == 1


def test_cynthia_rudin_rejects_unrelated_pubmed_email() -> None:
    candidate = score_email_candidate(
        email="rekhtman@mskcc.org",
        source="pubmed",
        source_url="https://pubmed.ncbi.nlm.nih.gov/3/",
        page_text="Cynthia Rudin rekhtman@mskcc.org",
        pi_name="Cynthia Rudin",
        pi_first_name="Cynthia",
        pi_last_name="Rudin",
        organization_name="Duke University",
    )

    assert candidate.confidence == "withheld"
    assert candidate.rejection_reason == "Email local-part does not match PI Cynthia Rudin."


def test_accepts_matching_email_when_page_evidence_is_present() -> None:
    candidate = score_email_candidate(
        email="rsayaman@ucsf.edu",
        source="institution_web",
        source_url="https://profiles.ucsf.edu/rosalyn.sayaman",
        page_text="Rosalyn Sayaman faculty profile Rosalyn Sayaman rsayaman@ucsf.edu",
        pi_name="Rosalyn Sayaman",
        pi_first_name="Rosalyn",
        pi_last_name="Sayaman",
        organization_name="University of California San Francisco",
    )

    assert candidate.confidence == "high"


def test_generic_emails_are_rejected() -> None:
    candidate = score_email_candidate(
        email="info@jhmi.edu",
        source="institution_web",
        source_url="https://www.hopkinsmedicine.org/department/contact",
        page_text="Jeremias Sulam Department info@jhmi.edu",
        pi_name="Jeremias Sulam",
        pi_first_name="Jeremias",
        pi_last_name="Sulam",
        organization_name="Johns Hopkins University",
    )

    assert candidate.confidence == "withheld"
    assert "Generic mailbox" in (candidate.rejection_reason or "")


def test_pubmed_coauthor_email_is_withheld_if_name_does_not_match() -> None:
    request = httpx.Request("GET", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi")
    fetch_request = httpx.Request("GET", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi")
    client = SequenceAsyncClient(
        [
            httpx.Response(200, request=request, json={"esearchresult": {"idlist": ["12345"]}}),
            httpx.Response(
                200,
                request=fetch_request,
                text=(
                    "<PubmedArticleSet><PubmedArticle><MedlineCitation><Article>"
                    "<ArticleTitle>AI for diabetes</ArticleTitle>"
                    "<AuthorList><Author><LastName>Sulam</LastName><ForeName>Jeremias</ForeName></Author></AuthorList>"
                    "<AffiliationInfo><Affiliation>Contact dbergles@jhmi.edu</Affiliation></AffiliationInfo>"
                    "</Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"
                ),
            ),
        ]
    )
    lookup = PubMedEmailLookup(client=client, min_request_interval_seconds=0)

    result = asyncio.run(lookup.lookup(_row("Jeremias Sulam")))

    assert result.status == "manual_review_required"
    assert result.candidate is None
    assert result.rejected_candidate is not None
    assert result.rejected_candidate.email == "dbergles@jhmi.edu"


def test_run_continues_when_all_email_candidates_are_withheld() -> None:
    rows = [_row("Cynthia Rudin", 99)]
    pubmed_source = StaticSource(
        "pubmed",
        {
            "Cynthia Rudin": SourceLookupResult(
                source="pubmed",
                status="manual_review_required",
                rejected_candidate=EmailCandidate(
                    email="rekhtman@mskcc.org",
                    confidence="withheld",
                    source="pubmed",
                    source_url="https://pubmed.ncbi.nlm.nih.gov/9/",
                    rejection_reason="Email local-part does not match PI Cynthia Rudin.",
                ),
                notes="Email local-part does not match PI Cynthia Rudin.",
            ),
        },
    )
    enricher = EmailEnricher(
        EmailEnrichmentConfig(enabled=True, max_researchers=1, sources=["pubmed"]),
        sources=[pubmed_source],
    )

    enriched_rows, summary = asyncio.run(enricher.enrich_rows(rows))

    assert enriched_rows[0].pi_email is None
    assert enriched_rows[0].email_status == "manual_review_required"
    assert summary["processed"] == 1
