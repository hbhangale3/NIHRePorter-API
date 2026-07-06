from __future__ import annotations

import httpx

from ..models import PIOutreachRow
from .email_utils import SourceLookupResult


class InstitutionWebLookup:
    name = "institution_web"

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
        return SourceLookupResult(
            source=self.name,
            status="skipped",
            notes=(
                "Institution web lookup is limited to direct public profile URLs. "
                "No institution profile URL was available for this researcher."
            ),
        )
