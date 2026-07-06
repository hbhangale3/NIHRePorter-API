from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models import PIOutreachRow
from app.retrieval import build_query_plans
from app.runner import run_pipeline


class QueryPlannerTests(unittest.TestCase):
    def test_query_planner_generates_dimension_crossing_queries(self) -> None:
        intent, plans = build_query_plans(
            research_question="AI for diabetes care in underserved populations",
            selected_concepts=["Artificial Intelligence", "Diabetes Mellitus", "Health Equity"],
            final_keywords=["Artificial Intelligence", "Machine Learning", "Diabetes Mellitus", "Health Equity"],
            mesh_terms=["Healthcare Disparities", "Medically Underserved Area"],
            semantic_terms=["Clinical Decision Support Systems"],
            semantic_concepts=["Medical Informatics"],
            retrieval_config=type(
                "Cfg",
                (),
                {
                    "enabled": True,
                    "max_queries": 6,
                    "pages_per_query": 1,
                    "require_dimension_overlap": True,
                    "include_original_query": True,
                },
            )(),
        )

        self.assertTrue(intent.technology_method)
        self.assertGreaterEqual(len(plans), 2)
        self.assertEqual(plans[0].query_id, "mq-original")
        self.assertTrue(any(len(plan.covered_dimensions) >= 2 for plan in plans[1:]))
        flattened = [" ".join(plan.search_terms) for plan in plans]
        self.assertTrue(any("Artificial Intelligence" in item or "Machine Learning" in item for item in flattened))
        self.assertTrue(any("Diabetes Mellitus" in item for item in flattened))


class MultiQueryRunnerTests(unittest.TestCase):
    @patch("app.runner.ReporterClient")
    @patch("app.runner.build_outreach_rows")
    def test_multi_query_disabled_preserves_single_query_behavior(
        self,
        mock_build_rows: object,
        mock_reporter_client_cls: object,
    ) -> None:
        config_yaml = """
query:
  research_question: AI for diabetes care in underserved populations
  fiscal_years: [2024]
  broad_keywords:
    - artificial intelligence
    - diabetes
  multi_query_retrieval:
    enabled: false
  ai_expansion:
    enabled: false
topics:
  - name: Topic
    include_any: [diabetes]
"""
        mock_client = mock_reporter_client_cls.return_value
        mock_client.fetch_all_projects = AsyncMock(return_value=[])
        mock_client.aclose = AsyncMock(return_value=None)
        mock_build_rows.return_value = ([], {"matched_project_count": 0, "counts_by_topic": {}})

        _results, summary, _keyword_expansions, expansion_trace = run_pipeline(config_yaml, max_pages=1)

        self.assertEqual(mock_client.fetch_all_projects.await_count, 1)
        self.assertFalse(expansion_trace["retrieval"]["multi_query_enabled"])
        self.assertEqual(summary["retrieval"]["query_count"], 0)

    @patch("app.runner.ReporterClient")
    @patch("app.runner.build_outreach_rows")
    def test_multi_query_merges_duplicates_and_records_trace(
        self,
        mock_build_rows: object,
        mock_reporter_client_cls: object,
    ) -> None:
        config_yaml = """
query:
  research_question: AI for diabetes care in underserved populations
  fiscal_years: [2024]
  broad_keywords:
    - artificial intelligence
    - diabetes
    - underserved populations
  multi_query_retrieval:
    enabled: true
    max_queries: 3
    pages_per_query: 1
    require_dimension_overlap: true
    include_original_query: false
  ai_expansion:
    enabled: false
topics:
  - name: Topic
    include_any: [diabetes]
"""
        mock_client = mock_reporter_client_cls.return_value
        mock_client.fetch_all_projects = AsyncMock(
            side_effect=[
                [{"appl_id": 1, "core_project_num": "R01-1", "project_title": "AI diabetes", "terms": []}],
                [{"appl_id": 1, "core_project_num": "R01-1", "project_title": "AI diabetes", "terms": []}],
                [{"appl_id": 2, "core_project_num": "R01-2", "project_title": "ML disparities", "terms": []}],
            ]
        )
        mock_client.aclose = AsyncMock(return_value=None)

        captured_projects: list[dict] = []

        def _build_rows(projects, _config):
            captured_projects.extend(projects)
            return (
                [
                    PIOutreachRow(
                        pi_name="Jane Doe",
                        matched_topics=["Topic"],
                        project_numbers=["R01-1"],
                        retrieval_query_matches=list(projects[0].get("retrieval_query_matches") or []),
                        retrieval_query_reasons=list(projects[0].get("retrieval_query_reasons") or []),
                    )
                ],
                {"matched_project_count": len(projects), "counts_by_topic": {"Topic": 1}},
            )

        mock_build_rows.side_effect = _build_rows

        results, summary, _keyword_expansions, expansion_trace = run_pipeline(config_yaml, max_pages=1)

        self.assertEqual(mock_client.fetch_all_projects.await_count, 3)
        self.assertTrue(expansion_trace["retrieval"]["multi_query_enabled"])
        self.assertEqual(expansion_trace["retrieval"]["merged_project_count"], 3)
        self.assertEqual(expansion_trace["retrieval"]["deduped_project_count"], 2)
        self.assertEqual(len(captured_projects), 2)
        self.assertGreaterEqual(len(captured_projects[0]["retrieval_query_matches"]), 1)
        self.assertEqual(summary["retrieval"]["deduped_project_count"], 2)
        self.assertIn("retrieval_query_matches", results[0])

