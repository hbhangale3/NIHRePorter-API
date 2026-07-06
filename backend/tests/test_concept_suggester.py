from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.concept_suggester import ConceptSuggester
from app.mesh.mesh_models import MeshDescriptor, MeshSearchResult
from app.semantic.semantic_models import SemanticMeshResult


class FakeSemanticRetriever:
    def __init__(self, results: list[SemanticMeshResult]) -> None:
        self.results = results

    def retrieve(self, question: str, top_k: int = 10) -> list[SemanticMeshResult]:
        return self.results[:top_k]


class FakeMeshKnowledgeBase:
    def __init__(
        self,
        *,
        lookup_matches: dict[str, list[MeshDescriptor]] | None = None,
        search_matches: dict[str, list[MeshSearchResult]] | None = None,
    ) -> None:
        self.lookup_matches = lookup_matches or {}
        self.search_matches = search_matches or {}

    def lookup_by_term(self, term: str) -> list[MeshDescriptor]:
        return self.lookup_matches.get(term, [])

    def search(self, term: str, *, limit: int = 10, score_cutoff: float = 60.0) -> list[MeshSearchResult]:
        return self.search_matches.get(term, [])[:limit]


def test_concept_suggester_returns_semantic_mesh_concepts_when_available() -> None:
    suggester = ConceptSuggester(
        semantic_retriever_factory=lambda: FakeSemanticRetriever(
            [
                SemanticMeshResult(
                    mesh_id="D1",
                    preferred_name="Artificial Intelligence",
                    score=0.82,
                    synonyms=["AI"],
                    tree_numbers=["L01"],
                    scope_note=None,
                    source_text_preview="preview",
                ),
                SemanticMeshResult(
                    mesh_id="D2",
                    preferred_name="Diabetes Mellitus",
                    score=0.79,
                    synonyms=["Diabetes"],
                    tree_numbers=["C18.452"],
                    scope_note=None,
                    source_text_preview="preview",
                ),
            ]
        ),
        mesh_kb_factory=lambda: FakeMeshKnowledgeBase(),
    )

    result = suggester.suggest("AI for diabetes care in underserved populations", top_k=8)

    assert result["fallback_used"] is False
    assert result["error"] is None
    assert [item["source"] for item in result["concepts"]] == ["semantic_mesh", "semantic_mesh"]
    assert [item["label"] for item in result["concepts"]] == ["Artificial Intelligence", "Diabetes Mellitus"]


def test_concept_suggester_falls_back_safely_when_semantic_is_unavailable() -> None:
    mesh_lookup = FakeMeshKnowledgeBase(
        search_matches={
            "diabetes care": [
                MeshSearchResult(
                    mesh_id="D2",
                    preferred_name="Diabetes Mellitus",
                    score=91.0,
                    matched_term="diabetes care",
                    source="descriptor",
                )
            ]
        }
    )
    suggester = ConceptSuggester(
        semantic_retriever_factory=lambda: (_ for _ in ()).throw(FileNotFoundError("missing semantic index")),
        mesh_kb_factory=lambda: mesh_lookup,
    )

    result = suggester.suggest("AI for diabetes care in underserved populations", top_k=8)

    assert result["fallback_used"] is True
    assert result["error"] is None
    assert result["concepts"][0]["label"] == "Diabetes Mellitus"
    assert result["concepts"][0]["source"] == "mesh_lookup"


def test_concept_suggester_dedupes_duplicate_concepts() -> None:
    suggester = ConceptSuggester(
        semantic_retriever_factory=lambda: FakeSemanticRetriever(
            [
                SemanticMeshResult(
                    mesh_id="D1",
                    preferred_name="Artificial Intelligence",
                    score=0.82,
                    synonyms=["AI"],
                    tree_numbers=[],
                    scope_note=None,
                    source_text_preview="preview",
                ),
                SemanticMeshResult(
                    mesh_id="D2",
                    preferred_name="artificial intelligence",
                    score=0.75,
                    synonyms=[],
                    tree_numbers=[],
                    scope_note=None,
                    source_text_preview="preview",
                ),
            ]
        ),
        mesh_kb_factory=lambda: FakeMeshKnowledgeBase(),
    )

    result = suggester.suggest("AI for diabetes care", top_k=8)

    assert [item["label"] for item in result["concepts"]] == ["Artificial Intelligence"]


def test_concept_suggester_returns_safe_empty_response_for_empty_question() -> None:
    suggester = ConceptSuggester(
        semantic_retriever_factory=lambda: FakeSemanticRetriever([]),
        mesh_kb_factory=lambda: FakeMeshKnowledgeBase(),
    )

    result = suggester.suggest("   ", top_k=8)

    assert result == {
        "question": "",
        "concepts": [],
        "fallback_used": False,
        "error": None,
    }
