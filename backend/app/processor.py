from __future__ import annotations

from collections import defaultdict
import logging
from typing import Any

from .models import AppConfig, PIOutreachRow, TopicConfig
from .settings import settings
from .topic_matcher import match_topic
from .utils import normalize_text, split_name, unique_preserve_order


logger = logging.getLogger(__name__)


def _project_text(project: dict[str, Any]) -> str:
    title = project.get("project_title") or ""
    abstract = project.get("abstract_text") or project.get("project_abstract") or ""
    terms = project.get("terms") or project.get("phr_text") or ""
    if isinstance(terms, list):
        terms = " ".join([str(t) for t in terms])
    return f"{title}\n{abstract}\n{terms}".strip()


def _pick_admin_ic(project: dict[str, Any]) -> str | None:
    # direct field
    v = project.get("admin_ic") or project.get("administering_ic") or project.get("ic_name")
    if v:
        return v
    # nested agency_ic_admin object (actual NIH API field)
    agency = project.get("agency_ic_admin")
    if isinstance(agency, dict):
        return agency.get("abbreviation") or agency.get("code") or agency.get("name")
    return None


def _pick_fiscal_year(project: dict[str, Any]) -> int | None:
    fy = project.get("fiscal_year") or project.get("fy")
    try:
        return int(fy) if fy is not None else None
    except Exception:
        return None


def _pick_project_id(project: dict[str, Any]) -> str | None:
    for k in ["appl_id", "application_id", "project_id", "project_num", "project_number"]:
        v = project.get(k)
        if v is not None:
            return str(v)
    return None


def _pick_project_num(project: dict[str, Any]) -> str | None:
    v = project.get("project_num") or project.get("project_number")
    return str(v) if v is not None else None


def _pick_terms(project: dict[str, Any]) -> str | None:
    terms = project.get("terms") or project.get("phr_text")
    if terms is None:
        return None
    if isinstance(terms, list):
        return " ".join(str(t) for t in terms)
    return str(terms)


def _project_url(project_id: str) -> str:
    # NIH RePORTER UI canonical path varies; project-details/{appl_id} works for appl_id
    return f"https://reporter.nih.gov/project-details/{project_id}"


def _iter_pis(project: dict[str, Any]) -> list[dict[str, Any]]:
    for k in ["principal_investigators", "principal_investigator", "pis", "pi"]:
        v = project.get(k)
        if v is None:
            continue
        if isinstance(v, list):
            return [pi for pi in v if isinstance(pi, dict)]
        if isinstance(v, dict):
            return [v]
    return []


def _pick_contact_pi(project: dict[str, Any]) -> dict[str, Any]:
    """Return the contact PI dict (is_contact_pi=True), falling back to first PI."""
    pis = _iter_pis(project)
    for pi in pis:
        if pi.get("is_contact_pi"):
            return pi
    return pis[0] if pis else {}


def _pick_core_project_num(project: dict[str, Any]) -> str | None:
    """Return the core (year-invariant) project number, e.g. 'R01CA123456'.
    Falls back to project_num if core_project_num is absent."""
    v = project.get("core_project_num") or project.get("project_serial_num")
    if v:
        return str(v)
    # derive from project_num by stripping leading activity type + trailing support year
    pn = project.get("project_num")
    if pn:
        return str(pn)
    return None


def _pick_org(project: dict[str, Any]) -> dict[str, Any]:
    org = project.get("organization")
    if isinstance(org, dict):
        return org
    return {}


def _match_topics(text: str, topics: list[TopicConfig]) -> tuple[list[str], dict[str, list[str]]]:
    matched: list[str] = []
    reasons: dict[str, list[str]] = {}
    for topic in topics:
        r = match_topic(text, topic)
        if r.matched:
            matched.append(topic.name)
            reasons[topic.name] = r.matched_terms
    return matched, reasons


def build_outreach_rows(projects: list[dict[str, Any]], config: AppConfig) -> tuple[list[PIOutreachRow], dict[str, Any]]:
    # ── Phase 1: topic-match and group all records by core project number ──────
    # Each core project may appear once per fiscal year; we collect all years,
    # then derive PI info from the most-recent year only.
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)   # core_num → [project, ...]
    matched_topics_by_core: dict[str, list[str]] = {}

    per_topic_counts: dict[str, int] = defaultdict(int)
    per_year_counts: dict[int, int] = defaultdict(int)
    per_admin_ic_counts: dict[str, int] = defaultdict(int)

    logger.info("Local filtering: received %d raw NIH projects", len(projects))

    matched_project_records = 0
    for project in projects:
        text = _project_text(project)
        matched_topics, _ = _match_topics(text, config.topics)
        if not matched_topics:
            continue
        matched_project_records += 1

        core_num = _pick_core_project_num(project)
        if not core_num:
            continue

        grouped[core_num].append(project)

        # accumulate topics across all fiscal years for this core project
        existing = matched_topics_by_core.setdefault(core_num, [])
        for t in matched_topics:
            if t not in existing:
                existing.append(t)

        fy = _pick_fiscal_year(project)
        admin_ic = _pick_admin_ic(project)
        if fy is not None:
            per_year_counts[fy] += 1
        if admin_ic:
            per_admin_ic_counts[str(admin_ic)] += 1
        for t in matched_topics:
            per_topic_counts[t] += 1

    # ── Phase 2: build one output row per core project ─────────────────────────
    out_rows: list[PIOutreachRow] = []

    for core_num, year_records in grouped.items():
        # Sort fiscal years descending — most recent first
        year_records.sort(key=lambda p: _pick_fiscal_year(p) or 0, reverse=True)
        latest = year_records[0]   # most-recent fiscal year record

        # Contact PI comes from the most-recent year only
        pi = _pick_contact_pi(latest)
        pi_name = pi.get("full_name") or pi.get("pi_name") or pi.get("name")
        pi_profile_id = pi.get("profile_id") or pi.get("pi_profile_id") or pi.get("person_id")
        first, last = split_name(str(pi_name) if pi_name is not None else None)

        org = _pick_org(latest)
        org_name = (
            org.get("org_name")
            or org.get("organization_name")
            or latest.get("org_name")
            or latest.get("organization_name")
        )
        admin_ic = _pick_admin_ic(latest)

        row = PIOutreachRow(
            pi_name=str(pi_name) if pi_name is not None else None,
            pi_first_name=pi.get("first_name") or first,
            pi_last_name=pi.get("last_name") or last,
            pi_email=pi.get("email") or None,
            organization_name=str(org_name) if org_name is not None else None,
            organization_city=org.get("city") or org.get("org_city") or latest.get("org_city"),
            organization_state=org.get("state") or org.get("org_state") or latest.get("org_state"),
            organization_country=org.get("country") or org.get("org_country") or latest.get("org_country"),
            admin_ic=str(admin_ic) if admin_ic is not None else None,
            pi_profile_id=str(pi_profile_id) if pi_profile_id is not None else None,
            matched_topics=sorted(set(matched_topics_by_core.get(core_num, []))),
            project_count=len(year_records),
        )

        # Aggregate across ALL fiscal years (oldest → newest for natural ordering)
        seen_titles: list[str] = []
        for rec in reversed(year_records):
            fy = _pick_fiscal_year(rec)
            if fy is not None:
                row.fiscal_years.append(fy)

            project_id = _pick_project_id(rec)
            if project_id:
                row.project_ids.append(project_id)
                row.project_urls.append(_project_url(project_id))

            pn = _pick_project_num(rec)
            if pn:
                row.project_numbers.append(pn)

            title = rec.get("project_title")
            if title and str(title) not in seen_titles:
                seen_titles.append(str(title))

            abstract = rec.get("abstract_text") or rec.get("project_abstract")
            if abstract:
                row.project_abstracts.append(str(abstract))

            terms_text = _pick_terms(rec)
            if terms_text:
                row.project_terms.append(terms_text)

            if isinstance(rec.get("retrieval_query_matches"), list):
                row.retrieval_query_matches.extend(str(item) for item in rec["retrieval_query_matches"])
            if isinstance(rec.get("retrieval_query_reasons"), list):
                row.retrieval_query_reasons.extend(str(item) for item in rec["retrieval_query_reasons"])

            # date range: earliest start, latest end
            sd = rec.get("project_start_date")
            if sd:
                s = str(sd)
                if row.project_start_date is None or s < row.project_start_date:
                    row.project_start_date = s
            ed = rec.get("project_end_date")
            if ed:
                e = str(ed)
                if row.project_end_date is None or e > row.project_end_date:
                    row.project_end_date = e

            # funding aggregation (best-effort)
            amount = rec.get("award_amount") or rec.get("total_cost") or rec.get("fy_total_cost")
            try:
                if amount is not None:
                    row.total_funding_amount = (row.total_funding_amount or 0.0) + float(amount)
            except Exception:
                pass

        row.fiscal_years = sorted(set(row.fiscal_years))
        row.project_ids = unique_preserve_order(row.project_ids)
        row.project_urls = unique_preserve_order(row.project_urls)
        row.project_numbers = unique_preserve_order(row.project_numbers)
        row.sample_project_titles = unique_preserve_order(seen_titles)[:3]
        row.retrieval_query_matches = unique_preserve_order(row.retrieval_query_matches)
        row.retrieval_query_reasons = unique_preserve_order(row.retrieval_query_reasons)

        out_rows.append(row)

    summary = {
        "matched_project_count": sum(per_year_counts.values()),
        "counts_by_topic": dict(sorted(per_topic_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "counts_by_year": dict(sorted(per_year_counts.items(), key=lambda kv: kv[0])),
        "counts_by_admin_ic": dict(sorted(per_admin_ic_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
    }

    logger.info(
        "Local filtering: matched project records=%d; grouped core projects=%d; outreach rows=%d",
        matched_project_records,
        len(grouped),
        len(out_rows),
    )

    return out_rows, summary
