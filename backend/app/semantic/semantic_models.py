from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class MeshVectorMetadata:
    mesh_id: str
    preferred_name: str
    synonyms: list[str]
    tree_numbers: list[str]
    scope_note: str | None
    source_text_preview: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MeshVectorMetadata":
        return cls(**data)


@dataclass(slots=True)
class SemanticMeshResult:
    mesh_id: str
    preferred_name: str
    score: float
    synonyms: list[str]
    tree_numbers: list[str]
    scope_note: str | None
    source_text_preview: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

