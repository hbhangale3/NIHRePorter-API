from __future__ import annotations

import json
import pickle
from pathlib import Path

from .mesh_models import MeshDescriptor, MeshQualifier, MeshSupplementaryRecord
from .mesh_parser import MeshParser


def _normalize_term(term: str) -> str:
    return " ".join(term.strip().lower().split())


class MeshIndexBuilder:
    def __init__(self, parser: MeshParser | None = None) -> None:
        self.parser = parser or MeshParser()

    def build(
        self,
        mesh_dir: str | Path,
        processed_dir: str | Path,
    ) -> dict[str, object]:
        mesh_dir = Path(mesh_dir)
        processed_dir = Path(processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)

        descriptor_path = mesh_dir / "desc2026.xml"
        qualifier_path = mesh_dir / "qual2026.xml"
        supplementary_path = mesh_dir / "supp2026.xml"

        if not descriptor_path.exists():
            raise FileNotFoundError(f"Required MeSH descriptor file not found: {descriptor_path}")

        descriptors = self.parser.parse_descriptors(descriptor_path)
        qualifiers = self.parser.parse_qualifiers(qualifier_path) if qualifier_path.exists() else {}
        supplementary_records = (
            self.parser.parse_supplementary_records(supplementary_path)
            if supplementary_path.exists()
            else {}
        )

        graph = self._build_graph(descriptors)
        lookup = self._build_lookup(descriptors, supplementary_records)

        self._write_json(
            processed_dir / "mesh_descriptors.json",
            {
                "descriptors": {ui: record.to_dict() for ui, record in descriptors.items()},
                "qualifiers": {ui: record.to_dict() for ui, record in qualifiers.items()},
                "supplementary_records": {ui: record.to_dict() for ui, record in supplementary_records.items()},
            },
        )
        self._write_json(processed_dir / "mesh_graph.json", graph)
        with (processed_dir / "mesh_lookup.pkl").open("wb") as handle:
            pickle.dump(lookup, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return {
            "descriptors": descriptors,
            "qualifiers": qualifiers,
            "supplementary_records": supplementary_records,
            "graph": graph,
            "lookup": lookup,
        }

    def _build_graph(self, descriptors: dict[str, MeshDescriptor]) -> dict[str, dict[str, list[str]]]:
        tree_to_descriptor: dict[str, str] = {}
        for descriptor in descriptors.values():
            for tree_number in descriptor.tree_numbers:
                tree_to_descriptor[tree_number] = descriptor.descriptor_ui

        children_by_ui: dict[str, set[str]] = {ui: set() for ui in descriptors}
        parents_by_ui: dict[str, set[str]] = {ui: set() for ui in descriptors}

        for descriptor in descriptors.values():
            for tree_number in descriptor.tree_numbers:
                parent_ui = self._immediate_parent_for_tree_number(tree_number, tree_to_descriptor)
                if parent_ui and parent_ui != descriptor.descriptor_ui:
                    parents_by_ui[descriptor.descriptor_ui].add(parent_ui)
                    children_by_ui[parent_ui].add(descriptor.descriptor_ui)

        graph: dict[str, dict[str, list[str]]] = {}
        for ui, descriptor in descriptors.items():
            ancestors = sorted(self._walk(parents_by_ui, ui))
            descendants = sorted(self._walk(children_by_ui, ui))
            descriptor.parents = sorted(parents_by_ui[ui])
            descriptor.children = sorted(children_by_ui[ui])
            descriptor.ancestors = ancestors
            descriptor.descendants = descendants
            graph[ui] = {
                "parents": descriptor.parents,
                "children": descriptor.children,
                "ancestors": ancestors,
                "descendants": descendants,
                "tree_numbers": list(descriptor.tree_numbers),
            }
        return graph

    def _build_lookup(
        self,
        descriptors: dict[str, MeshDescriptor],
        supplementary_records: dict[str, MeshSupplementaryRecord],
    ) -> dict[str, object]:
        term_to_descriptor_ids: dict[str, list[str]] = {}
        term_display: dict[str, str] = {}
        term_source: dict[str, str] = {}

        def add_term(term: str, mesh_id: str, preferred_name: str, source: str) -> None:
            normalized = _normalize_term(term)
            if not normalized:
                return
            ids = term_to_descriptor_ids.setdefault(normalized, [])
            if mesh_id not in ids:
                ids.append(mesh_id)
            term_display.setdefault(normalized, term.strip())
            term_source.setdefault(normalized, source)

        for descriptor in descriptors.values():
            for term in descriptor.all_terms():
                add_term(term, descriptor.descriptor_ui, descriptor.preferred_name, descriptor.source)

        for record in supplementary_records.values():
            if record.mapped_descriptors:
                for mapped in record.mapped_descriptors:
                    mesh_id = mapped["ui"]
                    preferred_name = mapped["name"]
                    for term in record.all_terms():
                        add_term(term, mesh_id, preferred_name, record.source)
            else:
                for term in record.all_terms():
                    add_term(term, record.supplemental_ui, record.preferred_name, record.source)

        return {
            "term_to_descriptor_ids": term_to_descriptor_ids,
            "term_display": term_display,
            "term_source": term_source,
        }

    def _immediate_parent_for_tree_number(
        self,
        tree_number: str,
        tree_to_descriptor: dict[str, str],
    ) -> str | None:
        current = tree_number
        while "." in current:
            current = current.rsplit(".", 1)[0]
            parent_ui = tree_to_descriptor.get(current)
            if parent_ui:
                return parent_ui
        return None

    def _walk(self, adjacency: dict[str, set[str]], start_ui: str) -> set[str]:
        visited: set[str] = set()
        stack = list(adjacency.get(start_ui, set()))
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(adjacency.get(current, set()) - visited)
        return visited

    def _write_json(self, path: Path, payload: dict[str, object]) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
