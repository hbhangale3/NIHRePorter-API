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
    rejection_reason: str | None = None


@dataclass(slots=True)
class SourceLookupResult:
    source: str
    candidate: EmailCandidate | None = None
    rejected_candidate: EmailCandidate | None = None
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


def _alpha_only(value: str | None) -> str:
    return re.sub(r"[^a-z]", "", (value or "").lower())


def email_local_part_matches_pi_name(
    email: str,
    *,
    pi_first_name: str | None,
    pi_last_name: str | None,
    pi_name: str | None = None,
) -> str:
    first_name, last_name = split_name(pi_name)
    first = _alpha_only(pi_first_name or first_name)
    last = _alpha_only(pi_last_name or last_name)
    local_part = email.split("@", 1)[0].lower()
    local_alpha = _alpha_only(local_part)

    if not last or not local_alpha:
        return "none"

    first_initial = first[:1]
    strong_patterns = {
        f"{first_initial}{last}" if first_initial else "",
        f"{last}{first_initial}" if first_initial else "",
        f"{first}{last}" if first else "",
        f"{last}{first}" if first else "",
    }
    if local_alpha in {pattern for pattern in strong_patterns if pattern}:
        return "strong"
    if local_alpha == last:
        return "medium"

    tokens = [token for token in re.split(r"[._\-+]+", local_part) if token]
    token_alpha = [_alpha_only(token) for token in tokens if _alpha_only(token)]
    if last in token_alpha and (first in token_alpha or first_initial in token_alpha):
        return "strong"
    if last in token_alpha:
        return "medium"

    if last in local_alpha:
        if first and first in local_alpha:
            return "strong"
        if first_initial and local_alpha.startswith(f"{first_initial}{last}"):
            return "strong"
        return "medium"

    return "none"


def is_assignable_confidence(confidence: str) -> bool:
    return confidence in {"high", "medium"}


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
    name_match_strength = email_local_part_matches_pi_name(
        email,
        pi_first_name=first_name,
        pi_last_name=last_name,
        pi_name=pi_name,
    )

    confidence = "withheld"
    notes: list[str] = []
    rejection_reason: str | None = None

    institutional_context = source == "institution_web" or page_org_match
    if generic:
        rejection_reason = f"Generic mailbox withheld for PI {pi_name or 'unknown researcher'}."
        notes.append("Generic mailbox detected.")
    elif source == "pubmed":
        if name_match_strength == "strong":
            confidence = "high" if org_domain_match else "medium"
            notes.append("PubMed email local-part strongly matches the target PI name.")
            if org_domain_match:
                notes.append("Email domain aligns with institution.")
        elif name_match_strength == "medium" and org_domain_match:
            confidence = "medium"
            notes.append("PubMed email local-part partially matches the target PI name.")
            notes.append("Email domain aligns with institution.")
        else:
            rejection_reason = f"Email local-part does not match PI {pi_name or 'unknown researcher'}."
            notes.append("PubMed found an email, but the local-part did not clearly match the target PI name.")
    elif institutional_context:
        if exact_name_match and name_match_strength == "strong":
            confidence = "high"
            notes.append("Exact PI name matched on an institutional or profile page.")
            notes.append("Email local-part strongly matches the PI name.")
        elif exact_name_match and name_match_strength == "medium" and org_domain_match:
            confidence = "medium"
            notes.append("Exact PI name matched on page and email local-part partially matches the PI name.")
            notes.append("Email domain aligns with institution.")
        else:
            rejection_reason = (
                f"Email local-part does not match PI {pi_name or 'unknown researcher'} strongly enough for assignment."
            )
            if not exact_name_match:
                notes.append("Institution page did not show an exact PI name match near the email.")
            if name_match_strength == "none":
                notes.append("Email local-part does not match the PI name.")
            else:
                notes.append("Email local-part matched only partially, so it was withheld.")
    elif exact_name_match and name_match_strength == "strong" and org_domain_match:
        confidence = "medium"
        notes.append("Exact PI name matched and email local-part strongly matches the PI name.")
        notes.append("Email domain aligns with institution.")
    else:
        rejection_reason = f"Email local-part does not match PI {pi_name or 'unknown researcher'} strongly enough."
        if first_name_match or last_name_match:
            notes.append("Partial PI name match detected, but evidence was too weak to assign the email.")
        else:
            notes.append("Email found with weak identity evidence.")
        if org_domain_match:
            notes.append("Email domain aligns with institution.")

    if confidence == "withheld" and rejection_reason is None:
        rejection_reason = f"Email withheld because PI identity could not be confirmed for {pi_name or 'unknown researcher'}."

    return EmailCandidate(
        email=email,
        confidence=confidence,
        source=source,
        source_url=source_url,
        notes=" ".join(notes).strip() or None,
        rejection_reason=rejection_reason,
    )


def choose_best_candidate(
    candidates: list[EmailCandidate],
    *,
    require_high_confidence: bool = False,
) -> EmailCandidate | None:
    assignable = [candidate for candidate in candidates if is_assignable_confidence(candidate.confidence)]
    if not assignable:
        return None

    confidence_rank = {"high": 3, "medium": 2, "low": 1}

    ranked = sorted(
        assignable,
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


def choose_best_rejected_candidate(candidates: list[EmailCandidate]) -> EmailCandidate | None:
    rejected = [candidate for candidate in candidates if not is_assignable_confidence(candidate.confidence)]
    if not rejected:
        return None

    confidence_rank = {"withheld": 2, "none": 1}
    ranked = sorted(
        rejected,
        key=lambda candidate: (
            -confidence_rank.get(candidate.confidence, 0),
            is_generic_email(candidate.email),
            candidate.source,
            candidate.email,
        ),
    )
    return ranked[0]
