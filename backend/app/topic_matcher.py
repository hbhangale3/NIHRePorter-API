from __future__ import annotations

from dataclasses import dataclass

from .models import TopicConfig


@dataclass(frozen=True)
class TopicMatchResult:
    matched: bool
    matched_terms: list[str]


def _contains_term(haystack: str, term: str) -> bool:
    t = term.strip().lower()
    if not t:
        return False
    return t in haystack


def match_topic(text: str, topic: TopicConfig) -> TopicMatchResult:
    text_lc = (text or "").lower()

    matched_terms: list[str] = []

    include_any = [t for t in topic.include_any if t.strip()]
    if include_any:
        if not any(_contains_term(text_lc, t) for t in include_any):
            return TopicMatchResult(matched=False, matched_terms=[])
        matched_terms.extend([t for t in include_any if _contains_term(text_lc, t)])

    include_all = [t for t in topic.include_all if t.strip()]
    if include_all:
        if not all(_contains_term(text_lc, t) for t in include_all):
            return TopicMatchResult(matched=False, matched_terms=[])
        matched_terms.extend([t for t in include_all if _contains_term(text_lc, t)])

    exclude_any = [t for t in topic.exclude_any if t.strip()]
    if exclude_any and any(_contains_term(text_lc, t) for t in exclude_any):
        return TopicMatchResult(matched=False, matched_terms=[])

    if topic.co_require_groups:
        for group in topic.co_require_groups:
            group_terms = [t for t in group if t.strip()]
            if not group_terms:
                continue
            if not any(_contains_term(text_lc, t) for t in group_terms):
                return TopicMatchResult(matched=False, matched_terms=[])
            matched_terms.extend([t for t in group_terms if _contains_term(text_lc, t)])

    # de-dupe terms while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for t in matched_terms:
        tl = t.lower()
        if tl in seen:
            continue
        seen.add(tl)
        deduped.append(t)

    return TopicMatchResult(matched=True, matched_terms=deduped)
