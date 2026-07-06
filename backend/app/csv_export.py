from __future__ import annotations

import io

import pandas as pd

from .models import PIOutreachRow


CSV_COLUMNS = [
    "pi_name",
    "pi_first_name",
    "pi_last_name",
    "pi_email",
    "organization_name",
    "organization_city",
    "organization_state",
    "organization_country",
    "admin_ic",
    "fiscal_years",
    "project_count",
    "matched_topics",
    "sample_project_titles",
    "project_numbers",
    "project_abstracts",
    "project_terms",
    "project_ids",
    "project_urls",
    "pi_profile_id",
    "total_funding_amount",
    "project_start_date",
    "project_end_date",
    "relevance_score",
    "relevance_badge",
    "matched_concepts",
    "matched_dimensions",
    "reasoning",
    "semantic_similarity",
    "mesh_matches",
    "ai_match",
    "disease_match",
    "population_match",
]


def rows_to_csv_bytes(rows: list[PIOutreachRow]) -> bytes:
    records: list[dict[str, object]] = []
    for r in rows:
        d = r.model_dump()
        # stringify list fields for CSV
        for k in [
            "fiscal_years",
            "matched_topics",
            "sample_project_titles",
            "project_numbers",
            "project_abstracts",
            "project_terms",
            "project_ids",
            "project_urls",
            "matched_concepts",
            "matched_dimensions",
            "mesh_matches",
        ]:
            if isinstance(d.get(k), list):
                d[k] = "; ".join([str(x) for x in d[k]])
        records.append({k: d.get(k) for k in CSV_COLUMNS})

    df = pd.DataFrame.from_records(records, columns=CSV_COLUMNS)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
