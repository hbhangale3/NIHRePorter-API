from __future__ import annotations

import json
import pickle
from pathlib import Path
from difflib import SequenceMatcher

try:
    from rapidfuzz import fuzz as rapidfuzz_fuzz
    from rapidfuzz import process as rapidfuzz_process
except ImportError:
    rapidfuzz_fuzz = None
    rapidfuzz_process = None

from .mesh_index_builder import MeshIndexBuilder
from .mesh_models import MeshDescriptor, MeshQualifier, MeshSearchResult, MeshSupplementaryRecord


def _normalize_term(term: str) -> str:
    return " ".join(term.strip().lower().split())


def _fallback_extract(
    query: str,
    choices: list[str],
    *,
    limit: int,
    score_cutoff: float,
) -> list[tuple[str, float, int]]:
    scored: list[tuple[str, float, int]] = []
    for index, choice in enumerate(choices):
        score = SequenceMatcher(None, query, choice).ratio() * 100.0
        if score >= score_cutoff:
            scored.append((choice, score, index))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:limit]


class MeshKnowledgeBase:
    def __init__(
        self,
        *,
        mesh_dir: str | Path | None = None,
        processed_dir: str | Path | None = None,
        auto_build: bool = True,
        builder: MeshIndexBuilder | None = None,
    ) -> None:
        backend_dir = Path(__file__).resolve().parents[2]
        self.mesh_dir = Path(mesh_dir) if mesh_dir is not None else backend_dir / "knowledge" / "mesh"
        self.processed_dir = (
            Path(processed_dir) if processed_dir is not None else backend_dir / "knowledge" / "processed"
        )
        self.builder = builder or MeshIndexBuilder()
        self.descriptors: dict[str, MeshDescriptor] = {}
        self.qualifiers: dict[str, MeshQualifier] = {}
        self.supplementary_records: dict[str, MeshSupplementaryRecord] = {}
        self.graph: dict[str, dict[str, list[str]]] = {}
        self.lookup: dict[str, object] = {}

        if auto_build:
            self.load_or_build()

    def load_or_build(self) -> None:
        descriptor_path = self.processed_dir / "mesh_descriptors.json"
        graph_path = self.processed_dir / "mesh_graph.json"
        lookup_path = self.processed_dir / "mesh_lookup.pkl"

        if descriptor_path.exists() and graph_path.exists() and lookup_path.exists():
            self._load_processed_files(descriptor_path, graph_path, lookup_path)
            return

        artifacts = self.builder.build(self.mesh_dir, self.processed_dir)
        self.descriptors = artifacts["descriptors"]  # type: ignore[assignment]
        self.qualifiers = artifacts["qualifiers"]  # type: ignore[assignment]
        self.supplementary_records = artifacts["supplementary_records"]  # type: ignore[assignment]
        self.graph = artifacts["graph"]  # type: ignore[assignment]
        self.lookup = artifacts["lookup"]  # type: ignore[assignment]

    def lookup_by_term(self, term: str) -> list[MeshDescriptor]:
        normalized = _normalize_term(term)
        ids = self.lookup.get("term_to_descriptor_ids", {}).get(normalized, [])
        return [self.descriptors[mesh_id] for mesh_id in ids if mesh_id in self.descriptors]

    def lookup_by_mesh_id(self, mesh_id: str) -> MeshDescriptor | None:
        return self.descriptors.get(mesh_id)

    def get_children(self, mesh_id: str) -> list[MeshDescriptor]:
        return self._resolve_graph(mesh_id, "children")

    def get_parents(self, mesh_id: str) -> list[MeshDescriptor]:
        return self._resolve_graph(mesh_id, "parents")

    def get_ancestors(self, mesh_id: str) -> list[MeshDescriptor]:
        return self._resolve_graph(mesh_id, "ancestors")

    def get_descendants(self, mesh_id: str) -> list[MeshDescriptor]:
        return self._resolve_graph(mesh_id, "descendants")

    def get_synonyms(self, mesh_id: str) -> list[str]:
        descriptor = self.lookup_by_mesh_id(mesh_id)
        if descriptor is None:
            return []
        return list(descriptor.entry_terms)

    def search(self, query: str, *, limit: int = 10, score_cutoff: float = 60.0) -> list[MeshSearchResult]:
        normalized_query = _normalize_term(query)
        if not normalized_query:
            return []

        term_display: dict[str, str] = self.lookup.get("term_display", {})  # type: ignore[assignment]
        term_source: dict[str, str] = self.lookup.get("term_source", {})  # type: ignore[assignment]
        search_terms = list(self.lookup.get("term_to_descriptor_ids", {}).keys())

        results: list[MeshSearchResult] = []
        seen_ids: set[str] = set()

        exact_ids = self.lookup.get("term_to_descriptor_ids", {}).get(normalized_query, [])
        for mesh_id in exact_ids:
            descriptor = self.descriptors.get(mesh_id)
            matched_term = term_display.get(normalized_query)
            if descriptor is None or matched_term is None:
                continue
            seen_ids.add(mesh_id)
            results.append(
                MeshSearchResult(
                    mesh_id=mesh_id,
                    preferred_name=descriptor.preferred_name,
                    score=100.0,
                    matched_term=matched_term,
                    source=term_source.get(normalized_query, descriptor.source),
                )
            )
        if results:
            return results[:limit]

        fuzzy_matches = self.process_extract(
            normalized_query,
            search_terms,
            limit=max(limit * 3, limit),
            score_cutoff=score_cutoff,
        )
        for matched_term, score, _ in fuzzy_matches:
            display_term = term_display.get(matched_term)
            if display_term is None:
                continue
            for mesh_id in self.lookup.get("term_to_descriptor_ids", {}).get(matched_term, []):
                if mesh_id in seen_ids or mesh_id not in self.descriptors:
                    continue
                seen_ids.add(mesh_id)
                results.append(
                    MeshSearchResult(
                        mesh_id=mesh_id,
                        preferred_name=self.descriptors[mesh_id].preferred_name,
                        score=float(score),
                        matched_term=display_term,
                        source=term_source.get(matched_term, self.descriptors[mesh_id].source),
                    )
                )
                if len(results) >= limit:
                    return results

        return results[:limit]

    def process_extract(
        self,
        query: str,
        choices: list[str],
        *,
        limit: int,
        score_cutoff: float,
    ) -> list[tuple[str, float, int]]:
        if rapidfuzz_process is not None and rapidfuzz_fuzz is not None:
            return rapidfuzz_process.extract(
                query,
                choices,
                scorer=rapidfuzz_fuzz.WRatio,
                limit=limit,
                score_cutoff=score_cutoff,
            )
        return _fallback_extract(query, choices, limit=limit, score_cutoff=score_cutoff)

    def _load_processed_files(self, descriptor_path: Path, graph_path: Path, lookup_path: Path) -> None:
        descriptor_payload = json.loads(descriptor_path.read_text(encoding="utf-8"))
        self.descriptors = {
            ui: MeshDescriptor.from_dict(data)
            for ui, data in descriptor_payload.get("descriptors", {}).items()
        }
        self.qualifiers = {
            ui: MeshQualifier.from_dict(data)
            for ui, data in descriptor_payload.get("qualifiers", {}).items()
        }
        self.supplementary_records = {
            ui: MeshSupplementaryRecord.from_dict(data)
            for ui, data in descriptor_payload.get("supplementary_records", {}).items()
        }
        self.graph = json.loads(graph_path.read_text(encoding="utf-8"))
        try:
            with lookup_path.open("rb") as handle:
                self.lookup = pickle.load(handle)
        except Exception:
            self.lookup = self.builder._build_lookup(self.descriptors, self.supplementary_records)
            with lookup_path.open("wb") as handle:
                pickle.dump(self.lookup, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _resolve_graph(self, mesh_id: str, relation: str) -> list[MeshDescriptor]:
        node = self.graph.get(mesh_id, {})
        related_ids = node.get(relation, [])
        return [self.descriptors[related_id] for related_id in related_ids if related_id in self.descriptors]
