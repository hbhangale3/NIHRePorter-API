from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class MeshQualifier:
    qualifier_ui: str
    name: str
    abbreviation: str | None = None
    history_note: str | None = None
    tree_numbers: list[str] = field(default_factory=list)
    scope_note: str | None = None
    entry_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MeshQualifier":
        return cls(**data)


@dataclass(slots=True)
class MeshDescriptor:
    descriptor_ui: str
    preferred_name: str
    entry_terms: list[str] = field(default_factory=list)
    tree_numbers: list[str] = field(default_factory=list)
    scope_note: str | None = None
    allowable_qualifiers: list[dict[str, str | None]] = field(default_factory=list)
    pharmacological_actions: list[dict[str, str]] = field(default_factory=list)
    previous_indexing: list[str] = field(default_factory=list)
    see_related: list[str] = field(default_factory=list)
    history_note: str | None = None
    parents: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    ancestors: list[str] = field(default_factory=list)
    descendants: list[str] = field(default_factory=list)
    source: str = "descriptor"

    def all_terms(self) -> list[str]:
        terms = [self.preferred_name]
        terms.extend(self.entry_terms)
        deduped: list[str] = []
        seen: set[str] = set()
        for term in terms:
            normalized = term.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(term.strip())
        return deduped

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MeshDescriptor":
        return cls(**data)


@dataclass(slots=True)
class MeshSupplementaryRecord:
    supplemental_ui: str
    preferred_name: str
    entry_terms: list[str] = field(default_factory=list)
    mapped_descriptors: list[dict[str, str]] = field(default_factory=list)
    pharmacological_actions: list[dict[str, str]] = field(default_factory=list)
    previous_indexing: list[str] = field(default_factory=list)
    note: str | None = None
    source: str = "supplementary"

    def all_terms(self) -> list[str]:
        terms = [self.preferred_name]
        terms.extend(self.entry_terms)
        deduped: list[str] = []
        seen: set[str] = set()
        for term in terms:
            normalized = term.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(term.strip())
        return deduped

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MeshSupplementaryRecord":
        return cls(**data)


@dataclass(slots=True)
class MeshSearchResult:
    mesh_id: str
    preferred_name: str
    score: float
    matched_term: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
