from __future__ import annotations

import argparse
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.semantic import MeshSemanticRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the local semantic MeSH index.")
    parser.add_argument("query", help="Free-text query to map to semantic MeSH concepts.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of nearest concepts to return.")
    parser.add_argument("--min-score", type=float, default=None, help="Optional minimum cosine score.")
    parser.add_argument(
        "--max-terms",
        type=int,
        default=30,
        help="Maximum number of expanded terms to print.",
    )
    parser.add_argument(
        "--no-synonyms",
        action="store_true",
        help="Do not include synonyms in the expanded term list.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        retriever = MeshSemanticRetriever()
        results = retriever.retrieve(args.query, top_k=args.top_k, min_score=args.min_score)
        expansion = retriever.expand_query(
            args.query,
            top_k=args.top_k,
            include_synonyms=not args.no_synonyms,
            max_terms=args.max_terms,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not results:
        print("No semantic MeSH concepts found.")
        return 0

    for index, result in enumerate(results, start=1):
        print(f"{index}. {result.preferred_name} | {result.mesh_id} | score={result.score:.3f}")
        print(f"   synonyms: {', '.join(result.synonyms) if result.synonyms else '(none)'}")
        print(f"   tree_numbers: {', '.join(result.tree_numbers) if result.tree_numbers else '(none)'}")
        print(f"   scope_note: {result.scope_note or '(none)'}")

    print("\nexpanded_terms:")
    for term in expansion["expanded_terms"]:
        print(f"- {term}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
