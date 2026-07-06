from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ProfileSourceResult:
    source: str
    urls: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
