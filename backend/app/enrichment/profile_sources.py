from __future__ import annotations

from typing import Protocol
from urllib.parse import quote_plus

from ..models import PIOutreachRow
from .profile_models import ProfileSourceResult


GOOGLE_SCHOLAR_BASE_URL = "https://scholar.google.com/scholar?q="
GOOGLE_SEARCH_BASE_URL = "https://www.google.com/search?q="
ORCID_SEARCH_BASE_URL = "https://orcid.org/orcid-search/search?searchQuery="
PUBMED_SEARCH_BASE_URL = "https://pubmed.ncbi.nlm.nih.gov/?term="
NIH_REPORTER_SEARCH_BASE_URL = "https://reporter.nih.gov/search/projects?query="


class ProfileSource(Protocol):
    name: str

    async def build(self, row: PIOutreachRow) -> ProfileSourceResult:
        ...


def _join_query_parts(parts: list[str]) -> str:
    return " ".join(part.strip() for part in parts if part and part.strip())


def _encode_query(parts: list[str]) -> str | None:
    query = _join_query_parts(parts)
    if not query:
        return None
    return quote_plus(query)


def build_google_scholar_query_url(row: PIOutreachRow) -> str | None:
    encoded = _encode_query([row.pi_name or "", row.organization_name or ""])
    return f"{GOOGLE_SCHOLAR_BASE_URL}{encoded}" if encoded else None


def build_faculty_profile_search_url(row: PIOutreachRow) -> str | None:
    encoded = _encode_query([row.pi_name or "", row.organization_name or "", "faculty profile"])
    return f"{GOOGLE_SEARCH_BASE_URL}{encoded}" if encoded else None


def build_pubmed_author_url(row: PIOutreachRow) -> str | None:
    pi_name = (row.pi_name or "").strip()
    if not pi_name:
        return None
    organization = (row.organization_name or "").strip()
    author_query = f'"{pi_name}"[Author]'
    query = f"{author_query} {organization}".strip()
    return f"{PUBMED_SEARCH_BASE_URL}{quote_plus(query)}"


def build_orcid_search_url(row: PIOutreachRow) -> str | None:
    encoded = _encode_query([row.pi_name or "", row.organization_name or ""])
    return f"{ORCID_SEARCH_BASE_URL}{encoded}" if encoded else None


def build_nih_reporter_profile_url(row: PIOutreachRow) -> str | None:
    if row.pi_profile_id:
        return f"https://reporter.nih.gov/pi-details/{row.pi_profile_id}"
    if row.project_ids:
        return f"https://reporter.nih.gov/project-details/{row.project_ids[0]}"
    encoded = _encode_query([row.pi_name or "", row.organization_name or ""])
    return f"{NIH_REPORTER_SEARCH_BASE_URL}{encoded}" if encoded else None


class NihReporterProfileSource:
    name = "nih_reporter"

    async def build(self, row: PIOutreachRow) -> ProfileSourceResult:
        url = build_nih_reporter_profile_url(row)
        notes: list[str] = []
        if row.pi_profile_id:
            notes.append("Direct NIH RePORTER PI profile URL available.")
        elif row.project_ids:
            notes.append("Using NIH RePORTER project page as a profile anchor.")
        elif url:
            notes.append("Generated NIH RePORTER search URL from PI name and organization.")
        return ProfileSourceResult(
            source=self.name,
            urls={"nih_reporter_pi_url": url} if url else {},
            notes=notes,
        )


class PubMedProfileSource:
    name = "pubmed"

    async def build(self, row: PIOutreachRow) -> ProfileSourceResult:
        url = build_pubmed_author_url(row)
        return ProfileSourceResult(
            source=self.name,
            urls={"pubmed_author_url": url} if url else {},
            notes=["Generated PubMed author search URL."] if url else [],
        )


class OrcidProfileSource:
    name = "orcid"

    async def build(self, row: PIOutreachRow) -> ProfileSourceResult:
        url = build_orcid_search_url(row)
        return ProfileSourceResult(
            source=self.name,
            urls={"orcid_url": url} if url else {},
            notes=["Generated ORCID public search URL."] if url else [],
        )


class InstitutionWebProfileSource:
    name = "institution_web"

    async def build(self, row: PIOutreachRow) -> ProfileSourceResult:
        url = build_faculty_profile_search_url(row)
        return ProfileSourceResult(
            source=self.name,
            urls={"faculty_profile_url": url} if url else {},
            notes=["Generated faculty-profile search URL for manual review."] if url else [],
        )


SOURCE_FACTORIES = {
    "nih_reporter": NihReporterProfileSource,
    "pubmed": PubMedProfileSource,
    "orcid": OrcidProfileSource,
    "institution_web": InstitutionWebProfileSource,
}
