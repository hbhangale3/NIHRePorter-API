from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.mesh_expander import MeshExpander
from app.models import MeshExpansionConfig, PIOutreachRow
from app.runner import run_pipeline


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
    @patch("app.runner.build_outreach_rows")
    @patch("app.runner.ReporterClient")
    @patch("app.runner.MeshExpander")
    def test_runner_includes_expansion_trace(
        self,
        mock_mesh_expander_cls: object,
        mock_reporter_client_cls: object,
        mock_build_rows: object,
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
        self.assertEqual(
            expansion_trace["mesh"]["terms_by_keyword"]["telemedicine"],
            ["Telemedicine", "Remote Consultation"],
        )
        self.assertEqual(
            expansion_trace["final_keywords"],
            ["telemedicine", "Remote Consultation"],
        )
