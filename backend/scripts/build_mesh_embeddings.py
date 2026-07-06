from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.semantic import EmbeddingModel, MeshEmbeddingBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build semantic embeddings for MeSH descriptors.")
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model name to use for descriptor embeddings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional descriptor limit for quick local builds.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Save checkpoint files every N batches.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing checkpoint if available.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Discard existing final artifacts and checkpoints before rebuilding.",
    )
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Keep checkpoint files after a successful final build.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = time.perf_counter()

    try:
        embedding_model = EmbeddingModel(model_name=args.model_name)
        builder = MeshEmbeddingBuilder(embedding_model=embedding_model)
        artifacts = builder.build(
            batch_size=args.batch_size,
            limit=args.limit,
            checkpoint_every=args.checkpoint_every,
            resume=args.resume,
            force=args.force,
            keep_checkpoints=args.keep_checkpoints,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - started_at
    print("Built MeSH semantic embeddings")
    print(f"- model name: {artifacts['model_name']}")
    print(f"- descriptor count: {artifacts['descriptor_count']}")
    print(f"- embedding dimension: {artifacts['embedding_dimension']}")
    print(f"- checkpoint dir: {artifacts['checkpoint_dir']}")
    print(f"- vector index: {artifacts['vector_path']}")
    print(f"- vector metadata: {artifacts['metadata_path']}")
    print(f"- embedding config: {artifacts['config_path']}")
    print(f"- elapsed time: {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
