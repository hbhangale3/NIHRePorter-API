from __future__ import annotations

import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.mesh.mesh_index_builder import MeshIndexBuilder


INPUT_FILENAMES = {
    "descriptors": "desc2026.xml",
    "qualifiers": "qual2026.xml",
    "supplementary_records": "supp2026.xml",
    "pharmacological_actions": "pa2026.xml",
}


def _count_graph_edges(graph: dict[str, dict[str, list[str]]]) -> int:
    return sum(len(node.get("children", [])) for node in graph.values())


def main() -> int:
    mesh_dir = BACKEND_DIR / "knowledge" / "mesh"
    processed_dir = BACKEND_DIR / "knowledge" / "processed"

    print("Building MeSH knowledge base")
    print(f"Input directory: {mesh_dir}")
    print(f"Output directory: {processed_dir}")

    found_inputs: dict[str, Path] = {}
    missing_inputs: dict[str, Path] = {}
    for label, filename in INPUT_FILENAMES.items():
        path = mesh_dir / filename
        if path.exists():
            found_inputs[label] = path
        else:
            missing_inputs[label] = path

    print("\nInput files:")
    for label, filename in INPUT_FILENAMES.items():
        path = mesh_dir / filename
        status = "found" if label in found_inputs else "missing"
        print(f"- {filename}: {status}")

    descriptor_path = mesh_dir / INPUT_FILENAMES["descriptors"]
    if not descriptor_path.exists():
        print(f"\nError: required MeSH descriptor file is missing: {descriptor_path}", file=sys.stderr)
        return 1

    if "qualifiers" in missing_inputs:
        print("Warning: qual2026.xml is missing; continuing with descriptors only.", file=sys.stderr)
    if "supplementary_records" in missing_inputs:
        print("Warning: supp2026.xml is missing; continuing without supplementary records.", file=sys.stderr)
    if "pharmacological_actions" in found_inputs:
        print("Note: pa2026.xml is present but not yet processed by the current MeSH builder.")
    else:
        print("Note: pa2026.xml not found; skipping because pharmacological action XML is not yet supported.")

    artifacts = MeshIndexBuilder().build(mesh_dir, processed_dir)
    graph = artifacts["graph"]

    descriptor_count = len(artifacts["descriptors"])
    qualifier_count = len(artifacts["qualifiers"])
    supplementary_count = len(artifacts["supplementary_records"])
    node_count = len(graph)
    edge_count = _count_graph_edges(graph)

    print("\nBuild summary:")
    print(f"- descriptor count: {descriptor_count}")
    print(f"- qualifier count: {qualifier_count}")
    print(f"- supplementary record count: {supplementary_count}")
    print(f"- graph node count: {node_count}")
    print(f"- graph edge count: {edge_count}")

    print("\nGenerated files:")
    print(f"- {processed_dir / 'mesh_descriptors.json'}")
    print(f"- {processed_dir / 'mesh_graph.json'}")
    print(f"- {processed_dir / 'mesh_lookup.pkl'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
