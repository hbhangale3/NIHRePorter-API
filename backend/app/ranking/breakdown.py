from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BreakdownEntry:
    key: str
    label: str
    points: int
    max_points: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "key": self.key,
            "label": self.label,
            "points": self.points,
            "max_points": self.max_points,
        }


BREAKDOWN_SPECS: list[tuple[str, str, int]] = [
    ("exact_topic_match", "Research Question Match", 25),
    ("semantic_similarity", "Semantic Similarity", 28),
    ("mesh_overlap", "MeSH Match", 18),
    ("dimension_coverage", "Dimension Coverage", 24),
    ("dimension_bonus", "Balanced Intent Match", 12),
    ("retrieval_multi_hit_bonus", "Targeted Query Support", 6),
    ("recent_funding", "Recent Funding", 5),
]


def build_breakdown_entries(score_breakdown: dict[str, int]) -> list[BreakdownEntry]:
    entries: list[BreakdownEntry] = []
    for key, label, max_points in BREAKDOWN_SPECS:
        entries.append(
            BreakdownEntry(
                key=key,
                label=label,
                points=int(score_breakdown.get(key, 0)),
                max_points=max_points,
            )
        )
    if int(score_breakdown.get("technology_penalty", 0)) > 0:
        entries.append(
            BreakdownEntry(
                key="technology_penalty",
                label="Missing Technology Penalty",
                points=-int(score_breakdown["technology_penalty"]),
                max_points=0,
            )
        )
    return entries


def breakdown_total(entries: list[BreakdownEntry]) -> int:
    return sum(entry.points for entry in entries)


def breakdown_text(entries: list[BreakdownEntry], total_score: int) -> str:
    parts = [
        f"{entry.label}: {entry.points} / {entry.max_points}" if entry.max_points > 0 else f"{entry.label}: {entry.points}"
        for entry in entries
    ]
    parts.append(f"Total: {total_score} / 100")
    return " | ".join(parts)
