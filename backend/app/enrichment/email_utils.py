from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import urlparse

from ..utils import normalize_text, split_name


EMAIL_REGEX = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b")
GENERIC_LOCAL_PART_PREFIXES = {
    "info",
    "admissions",
    "support",
    "webmaster",
    "privacy",
    "media",
}
COMMON_ORG_WORDS = {
    "and",
    "at",
    "center",
    "centre",
    "clinic",
    "college",
    "for",
    "health",
    "hospital",
    "institute",
    "medicine",
    "medical",
    "of",
    "school",
    "system",
    "the",
    "university",
}


@dataclass(slots=True)
class EmailCandidate:
    email: str
    confidence: str
    source: str
    source_url: str | None = None
    notes: str | None = None


@dataclass(slots=True)
class SourceLookupResult:
    source: str
    candidate: EmailCandidate | None = None
    status: str = "not_found"
    notes: str | None = None
    source_url: str | None = None


def extract_emails(text: str) -> list[str]:
    if not text:
        return []
    seen: set[str] = set()
    emails: list[str] = []
    for raw_email in EMAIL_REGEX.findall(text):
        normalized = raw_email.strip().strip(".,;:()[]{}<>").lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        emails.append(normalized)
    return emails


def is_generic_email(email: str) -> bool:
    local_part = email.split("@", 1)[0].lower()
    return any(local_part == prefix or local_part.startswith(f"{prefix}.") for prefix in GENERIC_LOCAL_PART_PREFIXES)


def _organization_domain_tokens(organization_name: str | None) -> set[str]:
    if not organization_name:
        return set()
    base_tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", organization_name.lower())
        if len(token) >= 3 and token not in COMMON_ORG_WORDS
    ]
    tokens = set(base_tokens)
    if len(base_tokens) >= 2:
        acronym = "".join(token[0] for token in base_tokens)
        if len(acronym) >= 2:
            tokens.add(acronym)
    return tokens


def _email_domain(email: str) -> str:
    return email.rsplit("@", 1)[-1].lower()


def domain_matches_organization(email: str, organization_name: str | None) -> bool:
    domain = _email_domain(email)
    tokens = _organization_domain_tokens(organization_name)
    return any(token in domain for token in tokens)


def page_host_matches_organization(source_url: str | None, organization_name: str | None) -> bool:
    if not source_url:
        return False
    host = urlparse(source_url).netloc.lower()
    tokens = _organization_domain_tokens(organization_name)
    return any(token in host for token in tokens)


def score_email_candidate(
    *,
    email: str,
    source: str,
    source_url: str | None,
    page_text: str,
    pi_name: str | None,
    pi_first_name: str | None,
    pi_last_name: str | None,
    organization_name: str | None,
) -> EmailCandidate:
    normalized_page = normalize_text(page_text)
    normalized_name = normalize_text(pi_name or "")
    first_name, last_name = split_name(pi_name)
    first_name = (pi_first_name or first_name or "").strip().lower()
    last_name = (pi_last_name or last_name or "").strip().lower()
    email_local = email.split("@", 1)[0].lower()

    exact_name_match = bool(normalized_name) and normalized_name in normalized_page
    last_name_match = bool(last_name) and (last_name in normalized_page or last_name in email_local)
    first_name_match = bool(first_name) and (first_name in normalized_page or first_name in email_local)
    org_domain_match = domain_matches_organization(email, organization_name)
    page_org_match = page_host_matches_organization(source_url, organization_name)
    generic = is_generic_email(email)

    confidence = "low"
    notes: list[str] = []

    institutional_context = source == "institution_web" or page_org_match
    if exact_name_match and not generic and institutional_context:
        confidence = "high"
        notes.append("Exact PI name matched on an institutional or profile page.")
    elif last_name_match and not generic and (org_domain_match or page_org_match):
        confidence = "medium"
        notes.append("PI last name and institution domain matched.")
    else:
        if generic:
            notes.append("Generic mailbox detected.")
        elif first_name_match or last_name_match:
            notes.append("Partial PI name match detected.")
        else:
            notes.append("Email found with weak identity evidence.")
        if org_domain_match:
            notes.append("Email domain aligns with institution.")

    return EmailCandidate(
        email=email,
        confidence=confidence,
        source=source,
        source_url=source_url,
        notes=" ".join(notes).strip() or None,
    )


def choose_best_candidate(
    candidates: list[EmailCandidate],
    *,
    require_high_confidence: bool = False,
) -> EmailCandidate | None:
    if not candidates:
        return None

    confidence_rank = {"high": 3, "medium": 2, "low": 1}

    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -confidence_rank.get(candidate.confidence, 0),
            is_generic_email(candidate.email),
            candidate.source,
            candidate.email,
        ),
    )

    best = ranked[0]
    if require_high_confidence and best.confidence != "high":
        return None
    return best
