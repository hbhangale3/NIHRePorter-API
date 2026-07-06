from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import PIOutreachRow
from ..semantic.embedding_model import EmbeddingModel
from ..semantic.semantic_cache import get_embedding_model


@dataclass(slots=True)
class SemanticScoreResult:
    similarity: float


class SemanticSimilarityScorer:
    def __init__(self, embedding_model: EmbeddingModel | None = None) -> None:
        self.embedding_model = embedding_model or get_embedding_model()

    def score_rows(self, query: str, rows: list[PIOutreachRow]) -> dict[str, SemanticScoreResult]:
        if not query.strip() or not rows:
            return {}

        texts = [_row_text(row) for row in rows]
        try:
            query_vector = self.embedding_model.embed_text(query)
            document_vectors = self.embedding_model.embed_texts(texts, batch_size=32)
        except Exception:
            return {
                _row_key(row, index): SemanticScoreResult(similarity=0.0)
                for index, row in enumerate(rows)
            }

        if query_vector.size == 0 or document_vectors.size == 0:
            return {
                _row_key(row, index): SemanticScoreResult(similarity=0.0)
                for index, row in enumerate(rows)
            }

        similarities = np.dot(document_vectors, query_vector).astype(float)
        return {
            _row_key(row, index): SemanticScoreResult(
                similarity=float(similarities[index]),
            )
            for index, row in enumerate(rows)
        }


def _row_text(row: PIOutreachRow) -> str:
    pieces = [
        " ".join(row.sample_project_titles),
        " ".join(row.project_abstracts),
        " ".join(row.project_terms),
    ]
    return "\n".join(piece for piece in pieces if piece.strip())


def _row_key(row: PIOutreachRow, index: int) -> str:
    return row.pi_profile_id or "|".join(row.project_numbers) or f"row-{index}"
