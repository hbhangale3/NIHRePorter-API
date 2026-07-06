from __future__ import annotations

import httpx

from ..models import PIOutreachRow
from .email_utils import SourceLookupResult


class OrcidLookup:
    name = "orcid"

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
        given_name = (row.pi_first_name or "").strip()
        family_name = (row.pi_last_name or "").strip()
        if not given_name or not family_name:
            return SourceLookupResult(
                source=self.name,
                status="skipped",
                notes="ORCID lookup requires split PI first and last names.",
            )

        query = f'given-and-family-names:"{given_name} {family_name}"'
        response = await self.client.get(
            "https://pub.orcid.org/v3.0/expanded-search/",
            params={"q": query, "rows": str(self.max_pages_per_researcher)},
            headers={"Accept": "application/json"},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("expanded-result", []) if isinstance(payload, dict) else []
        if not results:
            return SourceLookupResult(
                source=self.name,
                status="not_found",
                notes="No public ORCID profile match was found.",
            )

        first_result = results[0]
        orcid_id = first_result.get("orcid-id")
        organization = first_result.get("institution-name")
        notes = "Public ORCID profile found, but email addresses are typically private."
        if organization:
            notes = f"{notes} Matched institution: {organization}."
        return SourceLookupResult(
            source=self.name,
            status="not_found",
            source_url=f"https://orcid.org/{orcid_id}" if orcid_id else None,
            notes=notes,
        )
