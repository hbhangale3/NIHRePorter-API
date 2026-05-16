import re
import string
from typing import Iterable


_UNIV_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\buniv\.?\b", re.IGNORECASE), "university"),
    (re.compile(r"\buniv\b", re.IGNORECASE), "university"),
    (re.compile(r"\bdept\.?\b", re.IGNORECASE), "department"),
    (re.compile(r"\binst\.?\b", re.IGNORECASE), "institute"),
]


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    s = value.strip().lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    for pattern, repl in _UNIV_PATTERNS:
        s = pattern.sub(repl, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def split_name(full_name: str | None) -> tuple[str | None, str | None]:
    if not full_name:
        return None, None
    parts = [p for p in re.split(r"\s+", full_name.strip()) if p]
    if not parts:
        return None, None
    if len(parts) == 1:
        return parts[0], None
    return parts[0], parts[-1]


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
