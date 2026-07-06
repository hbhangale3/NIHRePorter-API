from __future__ import annotations

from collections import Counter
import logging
from typing import Protocol

import httpx

from ..models import EmailEnrichmentConfig, PIOutreachRow
from .email_utils import (
    EmailCandidate,
    SourceLookupResult,
    choose_best_candidate,
    choose_best_rejected_candidate,
)
from .institution_search import InstitutionWebLookup
from .orcid_lookup import OrcidLookup
from .pubmed_email import PubMedEmailLookup


logger = logging.getLogger(__name__)


class EmailLookupSource(Protocol):
    name: str

    async def lookup(self, row: PIOutreachRow) -> SourceLookupResult:
        ...


SOURCE_FACTORIES = {
    "institution_web": InstitutionWebLookup,
    "pubmed": PubMedEmailLookup,
    "orcid": OrcidLookup,
}


class EmailEnricher:
    def __init__(
        self,
        config: EmailEnrichmentConfig,
        *,
        sources: list[EmailLookupSource] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self._provided_sources = sources
        self._provided_client = http_client

    async def enrich_rows(self, rows: list[PIOutreachRow]) -> tuple[list[PIOutreachRow], dict[str, int]]:
        if not self.config.enabled:
            return rows, {"enabled": 0, "processed": 0}

        if not rows:
            return rows, {"enabled": 1, "processed": 0}

        updated_rows = list(rows)
        target_count = max(0, min(self.config.max_researchers, len(updated_rows)))
        counters = Counter()

        async with self._http_client_context() as client:
            sources = self._provided_sources or self._build_sources(client)
            for index, row in enumerate(updated_rows):
                if index >= target_count:
                    if not row.email_status:
                        updated_rows[index] = row.model_copy(
                            update={
                                "email_status": "skipped",
                                "email_notes": "Skipped because researcher rank was outside the enrichment limit.",
                            }
                        )
                    counters["skipped"] += 1
                    continue

                enriched_row = await self._enrich_single_row(row, sources)
                updated_rows[index] = enriched_row
                counters[enriched_row.email_status or "unknown"] += 1

        counters["enabled"] = 1
        counters["processed"] = target_count
        return updated_rows, dict(counters)

    async def _enrich_single_row(
        self,
        row: PIOutreachRow,
        sources: list[EmailLookupSource],
    ) -> PIOutreachRow:
        if row.pi_email:
            return row.model_copy(
                update={
                    "email_confidence": row.email_confidence or "high",
                    "email_source": row.email_source or "nih_reporter",
                    "email_status": row.email_status or "found_high_confidence",
                    "email_notes": row.email_notes or "Existing PI email preserved from source data.",
                    "email_candidate_rejected": None,
                    "email_rejection_reason": None,
                }
            )

        if not (row.pi_name or row.pi_last_name):
            return row.model_copy(
                update={
                    "email_status": "skipped",
                    "email_notes": "Skipped because the researcher name was incomplete.",
                }
            )

        candidates: list[EmailCandidate] = []
        rejected_candidates: list[EmailCandidate] = []
        notes: list[str] = []
        saw_non_skipped = False
        saw_non_error_result = False
        had_error = False

        for source in sources:
            try:
                result = await source.lookup(row)
            except Exception as exc:
                logger.warning("Email enrichment source %s failed for %s: %s", source.name, row.pi_name, exc)
                notes.append(f"{source.name}: {exc}")
                had_error = True
                continue

            if result.notes:
                notes.append(f"{source.name}: {result.notes}")
            if result.status == "error":
                had_error = True
            else:
                if result.status != "skipped":
                    saw_non_error_result = True
            if result.status != "skipped":
                saw_non_skipped = True
            if result.candidate is not None:
                candidates.append(result.candidate)
            if result.rejected_candidate is not None:
                rejected_candidates.append(result.rejected_candidate)

        best = choose_best_candidate(candidates, require_high_confidence=self.config.require_high_confidence)
        if best is not None:
            return row.model_copy(
                update={
                    "pi_email": best.email,
                    "email_confidence": best.confidence,
                    "email_source": best.source,
                    "email_source_url": best.source_url,
                    "email_status": f"found_{best.confidence}_confidence",
                    "email_notes": best.notes or " ".join(notes).strip() or None,
                    "email_candidate_rejected": None,
                    "email_rejection_reason": None,
                }
            )

        rejected = choose_best_rejected_candidate(rejected_candidates)
        if rejected is not None:
            notes.append(rejected.rejection_reason or "Email candidate was withheld pending manual review.")
            return row.model_copy(
                update={
                    "email_confidence": "withheld",
                    "email_source": rejected.source,
                    "email_source_url": rejected.source_url,
                    "email_status": "manual_review_required",
                    "email_notes": " ".join(notes).strip() or rejected.notes,
                    "email_candidate_rejected": rejected.email,
                    "email_rejection_reason": rejected.rejection_reason,
                }
            )

        status = "error" if had_error and not saw_non_error_result else "not_found" if saw_non_skipped else "skipped"
        confidence = "none" if status in {"not_found", "skipped"} else None
        if self.config.require_high_confidence and candidates:
            notes.append("Only medium confidence candidates were found, so none were kept because high confidence was required.")
        return row.model_copy(
            update={
                "email_status": status,
                "email_confidence": confidence,
                "email_notes": " ".join(notes).strip() or "No public email found.",
                "email_candidate_rejected": None,
                "email_rejection_reason": None,
            }
        )

    def _build_sources(self, client: httpx.AsyncClient) -> list[EmailLookupSource]:
        sources: list[EmailLookupSource] = []
        for source_name in self.config.sources:
            factory = SOURCE_FACTORIES.get(source_name)
            if factory is None:
                logger.info("Skipping unknown email enrichment source: %s", source_name)
                continue
            sources.append(
                factory(
                    client=client,
                    timeout_seconds=self.config.timeout_seconds,
                    max_pages_per_researcher=self.config.max_pages_per_researcher,
                )
            )
        return sources

    def _http_client_context(self):
        if self._provided_client is not None:
            return _ProvidedClientContext(self._provided_client)
        return httpx.AsyncClient(
            follow_redirects=True,
            timeout=self.config.timeout_seconds,
            headers={"User-Agent": "NIHRePORTER-API email enrichment/1.0"},
        )


class _ProvidedClientContext:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def __aenter__(self) -> httpx.AsyncClient:
        return self.client

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None
