from __future__ import annotations

import xml.etree.ElementTree as ET

import httpx

from ..models import PIOutreachRow
from .email_utils import SourceLookupResult, choose_best_candidate, extract_emails, score_email_candidate


class PubMedEmailLookup:
    name = "pubmed"

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        timeout_seconds: int = 10,
        max_pages_per_researcher: int = 3,
    ) -> None:
        self.client = client
        self.timeout_seconds = timeout_seconds
        self.max_pages_per_researcher = max_pages_per_researcher

    async def lookup(self, row: PIOutreachRow) -> SourceLookupResult:
        pi_name = (row.pi_name or "").strip()
        project_title = (row.sample_project_titles[0] if row.sample_project_titles else "").strip()
        if not pi_name or not project_title:
            return SourceLookupResult(
                source=self.name,
                status="skipped",
                notes="PubMed lookup requires both PI name and at least one project title.",
            )

        term = f"\"{pi_name}\"[Author] AND \"{project_title}\"[Title]"
        search_response = await self.client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={
                "db": "pubmed",
                "retmode": "json",
                "retmax": str(self.max_pages_per_researcher),
                "sort": "pub+date",
                "term": term,
            },
            timeout=self.timeout_seconds,
        )
        search_response.raise_for_status()
        payload = search_response.json()
        ids = (
            payload.get("esearchresult", {}).get("idlist", [])
            if isinstance(payload, dict)
            else []
        )
        if not ids:
            return SourceLookupResult(
                source=self.name,
                status="not_found",
                notes="No strong PubMed author/title matches were found.",
            )

        fetch_response = await self.client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={
                "db": "pubmed",
                "retmode": "xml",
                "id": ",".join(ids[: self.max_pages_per_researcher]),
            },
            timeout=self.timeout_seconds,
        )
        fetch_response.raise_for_status()

        root = ET.fromstring(fetch_response.text)
        candidates = []
        for article in root.findall(".//PubmedArticle"):
            article_text = " ".join(text.strip() for text in article.itertext() if text and text.strip())
            emails = extract_emails(article_text)
            for email in emails:
                candidates.append(
                    score_email_candidate(
                        email=email,
                        source=self.name,
                        source_url=f"https://pubmed.ncbi.nlm.nih.gov/{ids[0]}/",
                        page_text=article_text,
                        pi_name=row.pi_name,
                        pi_first_name=row.pi_first_name,
                        pi_last_name=row.pi_last_name,
                        organization_name=row.organization_name,
                    )
                )

        best = choose_best_candidate(candidates)
        if best is not None:
            return SourceLookupResult(
                source=self.name,
                status="candidate",
                candidate=best,
                source_url=best.source_url,
                notes=best.notes,
            )

        return SourceLookupResult(
            source=self.name,
            status="not_found",
            notes="PubMed results did not contain a usable public email address.",
        )
