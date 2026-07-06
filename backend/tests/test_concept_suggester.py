from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.concept_suggester import ConceptSuggester
from app.mesh.mesh_models import MeshDescriptor, MeshSearchResult
from app.semantic.semantic_models import SemanticMeshResult


class FakeSemanticRetriever:
    def __init__(self, results_by_query: dict[str, list[SemanticMeshResult]]) -> None:
        self.results_by_query = {key.lower(): value for key, value in results_by_query.items()}
        self.queries: list[str] = []

    def retrieve(self, question: str, top_k: int = 10) -> list[SemanticMeshResult]:
        self.queries.append(question)
        return list(self.results_by_query.get(question.lower(), []))[:top_k]


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


def test_imaging_oncology_query_returns_balanced_concepts() -> None:
    retriever = FakeSemanticRetriever(
        {
            "machine learning": [
                _semantic_result("D1", "Machine Learning", 0.96, tree_numbers=["L01.224"]),
                _semantic_result("D2", "Artificial Intelligence", 0.92, tree_numbers=["L01"]),
            ],
            "precision oncology": [
                _semantic_result("D3", "Medical Oncology", 0.89, tree_numbers=["C04"]),
                _semantic_result("D4", "Precision Medicine", 0.87, tree_numbers=["E05"]),
            ],
            "cancer": [
                _semantic_result("D5", "Neoplasms", 0.93, tree_numbers=["C04"]),
                _semantic_result("D6", "Rare Gene Pathway", 0.94, tree_numbers=["C04.588.894.797"]),
            ],
            "early cancer diagnosis": [
                _semantic_result("D7", "Early Detection of Cancer", 0.95, tree_numbers=["C04.588"]),
            ],
            "medical imaging": [
                _semantic_result("D8", "Diagnostic Imaging", 0.94, tree_numbers=["E01"]),
                _semantic_result("D9", "Radiology", 0.88, tree_numbers=["E01.370"]),
                _semantic_result("D10", "Image Processing, Computer-Assisted", 0.86, tree_numbers=["L01.224.100"]),
            ],
            "imaging": [
                _semantic_result("D11", "Medical Imaging", 0.91, tree_numbers=["E01"]),
            ],
            "oncology": [
                _semantic_result("D12", "Neoplasms", 0.91, tree_numbers=["C04"]),
            ],
        }
    )
    suggester = ConceptSuggester(
        semantic_retriever_factory=lambda: retriever,
        mesh_kb_factory=lambda: FakeMeshKnowledgeBase(),
    )

    result = suggester.suggest(
        "How can machine learning improve precision oncology and early cancer diagnosis using medical imaging?",
        top_k=8,
    )
    labels = [item["label"] for item in result["concepts"]]

    assert result["fallback_used"] is False
    assert "Machine Learning" in labels
    assert "Neoplasms" in labels or "Medical Oncology" in labels
    assert any(label in labels for label in ["Diagnostic Imaging", "Medical Imaging", "Radiology", "Image Processing, Computer-Assisted"])


def test_diabetes_underserved_query_returns_balanced_concepts() -> None:
    retriever = FakeSemanticRetriever(
        {
            "ai": [
                _semantic_result("D1", "Artificial Intelligence", 0.95, tree_numbers=["L01"]),
            ],
            "machine learning": [
                _semantic_result("D2", "Machine Learning", 0.93, tree_numbers=["L01.224"]),
            ],
            "diabetes": [
                _semantic_result("D3", "Diabetes Mellitus", 0.91, tree_numbers=["C18.452"]),
                _semantic_result("D4", "Diabetic Foot", 0.90, tree_numbers=["C17.800.174"]),
            ],
            "underserved populations": [
                _semantic_result("D5", "Medically Underserved Area", 0.89, tree_numbers=["Z01.542"]),
                _semantic_result("D6", "Health Equity", 0.88, tree_numbers=["N03.706"]),
            ],
            "disease management": [
                _semantic_result("D7", "Disease Management", 0.86, tree_numbers=["N04"]),
            ],
        }
    )
    suggester = ConceptSuggester(
        semantic_retriever_factory=lambda: retriever,
        mesh_kb_factory=lambda: FakeMeshKnowledgeBase(),
    )

    result = suggester.suggest("AI for diabetes care in underserved populations", top_k=8)
    labels = [item["label"] for item in result["concepts"]]

    assert "Artificial Intelligence" in labels or "Machine Learning" in labels
    assert "Diabetes Mellitus" in labels
    assert any(label in labels for label in ["Medically Underserved Area", "Health Equity", "Healthcare Disparities"])


def test_one_phrase_family_does_not_dominate_top_concepts() -> None:
    retriever = FakeSemanticRetriever(
        {
            "machine learning": [
                _semantic_result("D1", "Machine Learning", 0.98),
                _semantic_result("D2", "Artificial Intelligence", 0.96),
                _semantic_result("D3", "Medical Informatics", 0.94),
            ],
            "cancer": [
                _semantic_result("D4", "Neoplasms", 0.92, tree_numbers=["C04"]),
            ],
            "medical imaging": [
                _semantic_result("D5", "Diagnostic Imaging", 0.91, tree_numbers=["E01"]),
            ],
        }
    )
    suggester = ConceptSuggester(
        semantic_retriever_factory=lambda: retriever,
        mesh_kb_factory=lambda: FakeMeshKnowledgeBase(),
    )

    result = suggester.suggest(
        "How can machine learning improve cancer diagnosis using medical imaging?",
        top_k=5,
    )
    dimensions = [item["dimension"] for item in result["concepts"]]

    assert dimensions.count("technology_method") <= 2
    assert "disease_condition" in dimensions
    assert "diagnostic_imaging" in dimensions


def test_narrow_descriptors_are_penalized_when_broad_concepts_are_available() -> None:
    retriever = FakeSemanticRetriever(
        {
            "diabetes": [
                _semantic_result("D1", "Diabetic Coma", 0.97, tree_numbers=["C19.246.267"]),
                _semantic_result("D2", "Diabetic Foot", 0.96, tree_numbers=["C17.800.174"]),
                _semantic_result("D3", "Diabetes Mellitus", 0.88, tree_numbers=["C18.452"]),
                _semantic_result("D4", "NIDDK", 0.99),
            ],
            "underserved populations": [
                _semantic_result("D5", "Health Equity", 0.86, tree_numbers=["N03.706"]),
            ],
            "machine learning": [
                _semantic_result("D6", "Machine Learning", 0.90, tree_numbers=["L01.224"]),
            ],
        }
    )
    suggester = ConceptSuggester(
        semantic_retriever_factory=lambda: retriever,
        mesh_kb_factory=lambda: FakeMeshKnowledgeBase(),
    )

    result = suggester.suggest("Machine learning for diabetes in underserved populations", top_k=6)
    labels = [item["label"] for item in result["concepts"]]

    assert "Diabetes Mellitus" in labels
    assert "Diabetic Coma" not in labels
    assert "Diabetic Foot" not in labels
    assert "NIDDK" not in labels


def test_concept_metadata_includes_matched_phrase_and_dimension() -> None:
    retriever = FakeSemanticRetriever(
        {
            "medical imaging": [
                _semantic_result("D1", "Diagnostic Imaging", 0.92, tree_numbers=["E01"]),
            ],
        }
    )
    suggester = ConceptSuggester(
        semantic_retriever_factory=lambda: retriever,
        mesh_kb_factory=lambda: FakeMeshKnowledgeBase(),
    )

    result = suggester.suggest("medical imaging for cancer diagnosis", top_k=4)

    assert result["concepts"]
    concept = result["concepts"][0]
    assert concept["matched_phrase"] in {"medical imaging", "cancer diagnosis", "imaging"}
    assert concept["dimension"] in {"diagnostic_imaging", "disease_condition"}
    assert concept["source"] in {"exact_phrase", "semantic_mesh", "mesh_lookup", "fallback"}


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
    assert result["concepts"][0]["source"] in {"mesh_lookup", "exact_phrase"}


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
