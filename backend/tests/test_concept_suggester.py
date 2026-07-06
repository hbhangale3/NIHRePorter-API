from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.concept_suggester import ConceptSuggester
from app.mesh.mesh_models import MeshDescriptor, MeshSearchResult
from app.semantic.semantic_models import SemanticMeshResult


class FakeSemanticRetriever:
    def __init__(self, results_by_query: dict[str, list[SemanticMeshResult]]) -> None:
        self.results_by_query = results_by_query
        self.queries: list[str] = []

    def retrieve(self, question: str, top_k: int = 10) -> list[SemanticMeshResult]:
        self.queries.append(question)
        return list(self.results_by_query.get(question, []))[:top_k]


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


def _semantic_result(
    mesh_id: str,
    preferred_name: str,
    score: float,
    *,
    tree_numbers: list[str] | None = None,
) -> SemanticMeshResult:
    return SemanticMeshResult(
        mesh_id=mesh_id,
        preferred_name=preferred_name,
        score=score,
        synonyms=[],
        tree_numbers=tree_numbers or [],
        scope_note=None,
        source_text_preview="preview",
    )


def test_concept_suggester_returns_balanced_semantic_concepts_across_dimensions() -> None:
    retriever = FakeSemanticRetriever(
        {
            "ai": [
                _semantic_result("D1", "Artificial Intelligence", 0.95, tree_numbers=["L01"]),
                _semantic_result("D2", "Machine Learning", 0.91, tree_numbers=["L01.224"]),
                _semantic_result("D3", "Medical Informatics", 0.86, tree_numbers=["L01.700"]),
            ],
            "diabetes": [
                _semantic_result("D4", "Diabetic Coma", 0.96, tree_numbers=["C19.246.267"]),
                _semantic_result("D5", "Diabetes Mellitus", 0.90, tree_numbers=["C18.452"]),
                _semantic_result("D6", "Diabetic Foot", 0.89, tree_numbers=["C17.800.174"]),
            ],
            "underserved populations": [
                _semantic_result("D7", "Health Equity", 0.88, tree_numbers=["N03.706"]),
                _semantic_result("D8", "Healthcare Disparities", 0.87, tree_numbers=["N03.706.437"]),
                _semantic_result("D9", "Medically Underserved Area", 0.86, tree_numbers=["Z01.542"]),
            ],
            "care": [
                _semantic_result("D10", "Clinical Decision Support Systems", 0.89, tree_numbers=["L01.224.500"]),
                _semantic_result("D11", "Telemedicine", 0.84, tree_numbers=["N04.761"]),
            ],
        }
    )
    suggester = ConceptSuggester(
        semantic_retriever_factory=lambda: retriever,
        mesh_kb_factory=lambda: FakeMeshKnowledgeBase(),
    )

    result = suggester.suggest("AI for diabetes care in underserved populations", top_k=8)
    labels = [item["label"] for item in result["concepts"]]

    assert result["fallback_used"] is False
    assert "Artificial Intelligence" in labels
    assert "Machine Learning" in labels
    assert "Diabetes Mellitus" in labels
    assert "Health Equity" in labels or "Healthcare Disparities" in labels or "Medically Underserved Area" in labels
    assert "Clinical Decision Support Systems" in labels or "Telemedicine" in labels
    assert "Diabetic Coma" not in labels
    assert "Diabetic Foot" not in labels


def test_concept_suggester_penalizes_orgs_and_overly_specific_complications() -> None:
    retriever = FakeSemanticRetriever(
        {
            "diabetes": [
                _semantic_result("D1", "NIDDK", 0.99),
                _semantic_result("D2", "National Institute of Diabetes and Digestive and Kidney Diseases", 0.98),
                _semantic_result("D3", "Diabetic Coma", 0.97, tree_numbers=["C19.246.267"]),
                _semantic_result("D4", "Diabetes Mellitus", 0.85, tree_numbers=["C18.452"]),
            ],
            "ai": [_semantic_result("D5", "Artificial Intelligence", 0.9, tree_numbers=["L01"])],
            "underserved populations": [_semantic_result("D6", "Health Equity", 0.82, tree_numbers=["N03.706"])],
            "care": [_semantic_result("D7", "Patient Care", 0.8)],
        }
    )
    suggester = ConceptSuggester(
        semantic_retriever_factory=lambda: retriever,
        mesh_kb_factory=lambda: FakeMeshKnowledgeBase(),
    )

    result = suggester.suggest("AI for diabetes care in underserved populations", top_k=6)
    labels = [item["label"] for item in result["concepts"]]

    assert "Diabetes Mellitus" in labels
    assert "NIDDK" not in labels
    assert "National Institute of Diabetes and Digestive and Kidney Diseases" not in labels
    assert "Diabetic Coma" not in labels


def test_concept_suggester_falls_back_safely_when_semantic_is_unavailable() -> None:
    mesh_lookup = FakeMeshKnowledgeBase(
        search_matches={
            "diabetes": [
                MeshSearchResult(
                    mesh_id="D2",
                    preferred_name="Diabetes Mellitus",
                    score=91.0,
                    matched_term="diabetes",
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


def test_concept_suggester_returns_safe_empty_response_for_empty_question() -> None:
    suggester = ConceptSuggester(
        semantic_retriever_factory=lambda: FakeSemanticRetriever({}),
        mesh_kb_factory=lambda: FakeMeshKnowledgeBase(),
    )

    result = suggester.suggest("   ", top_k=8)

    assert result == {
        "question": "",
        "concepts": [],
        "fallback_used": False,
        "error": None,
    }
