from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.mesh_expander import MeshExpander
from app.models import MeshExpansionConfig, PIOutreachRow
from app.runner import format_search_terms_for_nih, run_pipeline


class MeshExpanderTests(unittest.TestCase):
    def test_mesh_expansion_disabled_keeps_keywords_unchanged(self) -> None:
        keywords = ["telemedicine", "health disparities"]
        expander = MeshExpander()

        final_keywords, trace = expander.expand_keywords(
            keywords,
            MeshExpansionConfig(enabled=False),
        )

        self.assertEqual(final_keywords, keywords)
        self.assertEqual(trace, {kw: [] for kw in keywords})

    @patch.object(MeshExpander, "_lookup_descriptor_ids", side_effect=RuntimeError("boom"))
    def test_mesh_expansion_failure_falls_back_to_original(self, _mock_lookup: object) -> None:
        expander = MeshExpander()
        config = MeshExpansionConfig(enabled=True, fallback_to_original=True, cache_enabled=False)

        final_keywords, trace = expander.expand_keywords(["telemedicine"], config)

        self.assertEqual(final_keywords, ["telemedicine"])
        self.assertEqual(trace, {"telemedicine": ["telemedicine"]})

    @patch.object(MeshExpander, "_lookup_descriptor_ids", return_value=["D0001"])
    @patch.object(
        MeshExpander,
        "_lookup_descriptor_details",
        return_value={
            "label": "Telemedicine",
            "terms": [{"label": "Remote Consultation"}, {"label": "Mobile Health"}],
            "narrowerDescriptor": [],
        },
    )
    def test_mesh_expansion_enabled_adds_mesh_terms(
        self,
        _mock_details: object,
        _mock_lookup: object,
    ) -> None:
        expander = MeshExpander()
        config = MeshExpansionConfig(enabled=True, cache_enabled=False)

        final_keywords, trace = expander.expand_keywords(["telemedicine"], config)

        self.assertEqual(
            final_keywords,
            ["telemedicine", "Remote Consultation", "Mobile Health"],
        )
        self.assertEqual(
            trace,
            {"telemedicine": ["Telemedicine", "Remote Consultation", "Mobile Health"]},
        )


class RunnerMeshIntegrationTests(unittest.TestCase):
    @patch("app.runner.MeshExpander")
    @patch("app.runner.ReporterClient")
    @patch("app.runner.build_outreach_rows")
    def test_runner_includes_expansion_trace(
        self,
        mock_build_rows: object,
        mock_reporter_client_cls: object,
        mock_mesh_expander_cls: object,
    ) -> None:
        config_yaml = """
query:
  fiscal_years: [2024]
  broad_keywords:
    - telemedicine
  text_search_field: all
  text_search_operator: or
  mesh_expansion:
    enabled: true
  ai_expansion:
    enabled: false
topics:
  - name: Telehealth Equity
    include_any: [telemedicine]
"""
        mock_mesh_expander = mock_mesh_expander_cls.return_value
        mock_mesh_expander.expand_keywords.return_value = (
            ["telemedicine", "Remote Consultation"],
            {"telemedicine": ["Telemedicine", "Remote Consultation"]},
        )

        mock_client = mock_reporter_client_cls.return_value
        mock_client.fetch_all_projects = AsyncMock(return_value=[])
        mock_client.aclose = AsyncMock(return_value=None)

        mock_build_rows.return_value = (
            [
                PIOutreachRow(
                    pi_name="Jane Doe",
                    matched_topics=["Telehealth Equity"],
                    project_numbers=["P123"],
                )
            ],
            {"matched_project_count": 1, "counts_by_topic": {"Telehealth Equity": 1}},
        )

        results, summary, keyword_expansions, expansion_trace = run_pipeline(config_yaml, max_pages=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(summary["matched_project_count"], 1)
        self.assertEqual(keyword_expansions, {})
        self.assertEqual(expansion_trace["original_keywords"], ["telemedicine"])
        self.assertTrue(expansion_trace["mesh"]["enabled"])
        self.assertFalse(expansion_trace["semantic"]["enabled"])
        self.assertEqual(expansion_trace["semantic"]["expanded_terms"], [])
        self.assertIsNone(expansion_trace["semantic"]["query"])
        self.assertIsNone(expansion_trace["semantic"]["error"])
        self.assertEqual(
            expansion_trace["mesh"]["terms_by_keyword"]["telemedicine"],
            ["Telemedicine", "Remote Consultation"],
        )
        self.assertEqual(
            expansion_trace["final_keywords"],
            ["telemedicine", "Remote Consultation"],
        )

    @patch("app.runner.MeshSemanticRetriever")
    @patch("app.runner.ReporterClient")
    @patch("app.runner.build_outreach_rows")
    def test_runner_includes_semantic_terms_when_enabled(
        self,
        mock_build_rows: object,
        mock_reporter_client_cls: object,
        mock_retriever_cls: object,
    ) -> None:
        config_yaml = """
query:
  fiscal_years: [2024]
  broad_keywords:
    - AI
    - diabetes
    - underserved populations
  text_search_field: all
  semantic_expansion:
    enabled: true
    top_k: 5
    max_terms: 6
    include_synonyms: true
  ai_expansion:
    enabled: false
topics:
  - name: Equity
    include_any: [diabetes]
"""
        mock_retriever = mock_retriever_cls.return_value
        mock_retriever.expand_query.return_value = {
            "query": "AI diabetes underserved populations",
            "semantic_concepts": [
                {
                    "mesh_id": "D1",
                    "preferred_name": "Diabetes Mellitus",
                    "score": 0.91,
                    "synonyms": ["Diabetes"],
                    "tree_numbers": ["C18.452"],
                    "scope_note": "Test scope note",
                }
            ],
            "expanded_terms": ["Diabetes Mellitus", "Healthcare Disparities"],
        }

        mock_client = mock_reporter_client_cls.return_value
        mock_client.fetch_all_projects = AsyncMock(return_value=[])
        mock_client.aclose = AsyncMock(return_value=None)

        mock_build_rows.return_value = ([], {"matched_project_count": 0, "counts_by_topic": {}})

        _results, _summary, _keyword_expansions, expansion_trace = run_pipeline(config_yaml, max_pages=1)

        mock_retriever.expand_query.assert_called_once()
        criteria = mock_client.fetch_all_projects.await_args.args[0]
        self.assertEqual(criteria["advanced_text_search"]["operator"], "or")
        self.assertEqual(
            criteria["advanced_text_search"]["search_text"],
            'AI diabetes "underserved populations" "Diabetes Mellitus" "Healthcare Disparities"',
        )
        self.assertTrue(expansion_trace["semantic"]["enabled"])
        self.assertEqual(expansion_trace["semantic"]["query"], "AI diabetes underserved populations")
        self.assertEqual(
            expansion_trace["semantic"]["expanded_terms"],
            ["Diabetes Mellitus", "Healthcare Disparities"],
        )
        self.assertEqual(
            expansion_trace["final_keywords"],
            ["AI", "diabetes", "underserved populations", "Diabetes Mellitus", "Healthcare Disparities"],
        )

    @patch("app.runner.MeshSemanticRetriever")
    @patch("app.runner.ReporterClient")
    @patch("app.runner.build_outreach_rows")
    def test_runner_continues_when_semantic_expansion_is_optional(
        self,
        mock_build_rows: object,
        mock_reporter_client_cls: object,
        mock_retriever_cls: object,
    ) -> None:
        config_yaml = """
query:
  fiscal_years: [2024]
  broad_keywords:
    - telemedicine
  text_search_field: all
  semantic_expansion:
    enabled: true
    require_existing_index: false
  ai_expansion:
    enabled: false
topics:
  - name: Telehealth
    include_any: [telemedicine]
"""
        mock_retriever = mock_retriever_cls.return_value
        mock_retriever.expand_query.side_effect = FileNotFoundError("missing semantic index")

        mock_client = mock_reporter_client_cls.return_value
        mock_client.fetch_all_projects = AsyncMock(return_value=[])
        mock_client.aclose = AsyncMock(return_value=None)

        mock_build_rows.return_value = ([], {"matched_project_count": 0, "counts_by_topic": {}})

        _results, _summary, _keyword_expansions, expansion_trace = run_pipeline(config_yaml, max_pages=1)

        self.assertTrue(expansion_trace["semantic"]["enabled"])
        self.assertEqual(expansion_trace["semantic"]["expanded_terms"], [])
        self.assertEqual(expansion_trace["semantic"]["concepts"], [])
        self.assertEqual(expansion_trace["semantic"]["error"], "missing semantic index")
        self.assertEqual(expansion_trace["final_keywords"], ["telemedicine"])

    @patch("app.runner.MeshSemanticRetriever")
    def test_runner_fails_when_semantic_expansion_is_required(self, mock_retriever_cls: object) -> None:
        config_yaml = """
query:
  fiscal_years: [2024]
  broad_keywords:
    - telemedicine
  text_search_field: all
  semantic_expansion:
    enabled: true
    require_existing_index: true
  ai_expansion:
    enabled: false
topics:
  - name: Telehealth
    include_any: [telemedicine]
"""
        mock_retriever = mock_retriever_cls.return_value
        mock_retriever.expand_query.side_effect = FileNotFoundError("missing semantic index")

        with self.assertRaisesRegex(RuntimeError, "require_existing_index=true"):
            run_pipeline(config_yaml, max_pages=1)


class RunnerSearchFormattingTests(unittest.TestCase):
    def test_format_search_terms_for_nih_quotes_multi_word_terms(self) -> None:
        formatted = format_search_terms_for_nih(
            [
                "telemedicine",
                "Remote Consultation",
                "Health Status Disparities",
                "telemedicine",
                'He said "AI"',
            ]
        )

        self.assertEqual(
            formatted,
            'telemedicine "Remote Consultation" "Health Status Disparities" "He said \\"AI\\""',
        )
