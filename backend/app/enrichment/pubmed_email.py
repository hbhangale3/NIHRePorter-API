from __future__ import annotations

import asyncio
import os
import xml.etree.ElementTree as ET

import httpx

from ..models import PIOutreachRow
from ..utils import normalize_text
from .email_utils import (
    SourceLookupResult,
    choose_best_candidate,
    choose_best_rejected_candidate,
    extract_emails,
    score_email_candidate,
)


class PubMedEmailLookup:
    name = "pubmed"
    _rate_limit_lock = asyncio.Lock()
    _last_request_monotonic: float | None = None

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        timeout_seconds: int = 10,
        max_pages_per_researcher: int = 3,
        ncbi_api_key: str | None = None,
        min_request_interval_seconds: float | None = None,
        sleep_func=None,
    ) -> None:
        self.client = client
        self.timeout_seconds = timeout_seconds
        self.max_pages_per_researcher = max_pages_per_researcher
        self.ncbi_api_key = ncbi_api_key or os.getenv("NCBI_API_KEY")
        self.min_request_interval_seconds = (
            min_request_interval_seconds
            if min_request_interval_seconds is not None
            else 0.12 if self.ncbi_api_key else 0.5
        )
        self._sleep = sleep_func or asyncio.sleep
        self._lookup_cache: dict[str, SourceLookupResult] = {}

    async def lookup(self, row: PIOutreachRow) -> SourceLookupResult:
        pi_name = (row.pi_name or "").strip()
        if not pi_name:
            return SourceLookupResult(
                source=self.name,
                status="skipped",
                notes="PubMed lookup requires a PI name.",
            )

        cache_key = self._cache_key(row)
        cached = self._lookup_cache.get(cache_key)
        if cached is not None:
            return cached

        queries = _build_pubmed_queries(row)
        ids: list[str] = []
        query_used: str | None = None

        try:
            for query in queries:
                payload = await self._search(query)
                ids = (
                    payload.get("esearchresult", {}).get("idlist", [])
                    if isinstance(payload, dict)
                    else []
                )
                if ids:
                    query_used = query
                    break
        except httpx.HTTPStatusError as exc:
            result = SourceLookupResult(
                source=self.name,
                status="error",
                notes=f"PubMed lookup failed after retries with status {exc.response.status_code}.",
            )
            self._lookup_cache[cache_key] = result
            return result
        except Exception as exc:
            result = SourceLookupResult(
                source=self.name,
                status="error",
                notes=f"PubMed lookup failed: {exc}",
            )
            self._lookup_cache[cache_key] = result
            return result

        if not ids:
            result = SourceLookupResult(
                source=self.name,
                status="not_found",
                notes="No strong PubMed author matches were found.",
            )
            self._lookup_cache[cache_key] = result
            return result

        try:
            fetch_response = await self._request_with_retry(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={
                    "db": "pubmed",
                    "retmode": "xml",
                    "id": ",".join(ids[: self.max_pages_per_researcher]),
                    **({"api_key": self.ncbi_api_key} if self.ncbi_api_key else {}),
                },
            )
        except httpx.HTTPStatusError as exc:
            result = SourceLookupResult(
                source=self.name,
                status="error",
                notes=f"PubMed fetch failed after retries with status {exc.response.status_code}.",
            )
            self._lookup_cache[cache_key] = result
            return result

        root = ET.fromstring(fetch_response.text)
        candidates = []
        rejected_candidates = []
        for article, pmid in zip(root.findall(".//PubmedArticle"), ids[: self.max_pages_per_researcher], strict=False):
            article_text = " ".join(text.strip() for text in article.itertext() if text and text.strip())
            emails = extract_emails(article_text)
            for email in emails:
                scored = score_email_candidate(
                    email=email,
                    source=self.name,
                    source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    page_text=article_text,
                    pi_name=row.pi_name,
                    pi_first_name=row.pi_first_name,
                    pi_last_name=row.pi_last_name,
                    organization_name=row.organization_name,
                )
                if scored.confidence in {"high", "medium"}:
                    candidates.append(scored)
                else:
                    rejected_candidates.append(scored)

        best = choose_best_candidate(candidates)
        if best is not None:
            result = SourceLookupResult(
                source=self.name,
                status="candidate",
                candidate=best,
                source_url=best.source_url,
                notes=best.notes or (f"Matched via PubMed query: {query_used}" if query_used else None),
            )
            self._lookup_cache[cache_key] = result
            return result

        rejected = choose_best_rejected_candidate(rejected_candidates)
        if rejected is not None:
            result = SourceLookupResult(
                source=self.name,
                status="manual_review_required",
                rejected_candidate=rejected,
                source_url=rejected.source_url,
                notes=(
                    rejected.rejection_reason
                    or "PubMed found an email candidate, but it did not match the target PI name strongly enough."
                ),
            )
            self._lookup_cache[cache_key] = result
            return result

        result = SourceLookupResult(
            source=self.name,
            status="not_found",
            notes=(
                f"PubMed query returned records but no usable public email address was found."
                if query_used
                else "PubMed results did not contain a usable public email address."
            ),
        )
        self._lookup_cache[cache_key] = result
        return result

    async def _search(self, query: str) -> dict:
        response = await self._request_with_retry(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={
                "db": "pubmed",
                "retmode": "json",
                "retmax": str(self.max_pages_per_researcher),
                "sort": "pub+date",
                "term": query,
                **({"api_key": self.ncbi_api_key} if self.ncbi_api_key else {}),
            },
        )
        return response.json()

    async def _request_with_retry(self, url: str, *, params: dict[str, str]) -> httpx.Response:
        backoff_seconds = [1, 2, 4]
        last_exc: httpx.HTTPStatusError | None = None
        for attempt in range(len(backoff_seconds) + 1):
            await self._throttle()
            response = await self.client.get(
                url,
                params=params,
                timeout=self.timeout_seconds,
            )
            try:
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 429 or attempt >= len(backoff_seconds):
                    raise
                last_exc = exc
                await self._sleep(backoff_seconds[attempt])
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("PubMed request retry loop exited unexpectedly.")

    async def _throttle(self) -> None:
        async with self._rate_limit_lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            if self._last_request_monotonic is not None:
                elapsed = now - self._last_request_monotonic
                remaining = self.min_request_interval_seconds - elapsed
                if remaining > 0:
                    await self._sleep(remaining)
            self.__class__._last_request_monotonic = loop.time()

    def _cache_key(self, row: PIOutreachRow) -> str:
        return "|".join(
            [
                normalize_text(row.pi_name or ""),
                normalize_text(row.organization_name or ""),
            ]
        )

    @classmethod
    def reset_rate_limit_state(cls) -> None:
        cls._last_request_monotonic = None


def _build_pubmed_queries(row: PIOutreachRow) -> list[str]:
    queries: list[str] = []
    last_name = (row.pi_last_name or "").strip()
    first_name = (row.pi_first_name or "").strip()
    full_name = (row.pi_name or "").strip()
    organization_keyword = _organization_keyword(row.organization_name)

    if last_name and first_name:
        initials = "".join(part[0].upper() for part in first_name.split() if part)
        if initials:
            queries.append(f"{last_name} {initials}[Author]")
    if full_name:
        queries.append(f"\"{full_name}\"[Author]")
    if last_name:
        query = f"{last_name}[Author]"
        if organization_keyword:
            query = f"{query} AND {organization_keyword}"
        queries.append(query)

    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(query)
    return deduped


def _organization_keyword(organization_name: str | None) -> str:
    if not organization_name:
        return ""
    words = [word for word in organization_name.replace(",", " ").split() if word]
    acronym = "".join(word[0].upper() for word in words if word[0].isalpha())
    if 2 <= len(acronym) <= 6:
        return acronym
    for word in words:
        cleaned = word.strip()
        if len(cleaned) >= 4 and cleaned.lower() not in {"university", "college", "school", "health"}:
            return cleaned
    return words[0] if words else ""
