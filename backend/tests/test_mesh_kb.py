from __future__ import annotations

import json
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.mesh.mesh_index_builder import MeshIndexBuilder
from app.mesh.mesh_loader import MeshKnowledgeBase
from app.mesh.mesh_parser import MeshParser


DESCRIPTOR_XML = """<?xml version="1.0"?>
<DescriptorRecordSet LanguageCode="eng">
  <DescriptorRecord DescriptorClass="1">
    <DescriptorUI>D000001</DescriptorUI>
    <DescriptorName><String>Cardiovascular Diseases</String></DescriptorName>
    <TreeNumberList><TreeNumber>C14</TreeNumber></TreeNumberList>
    <ConceptList>
      <Concept PreferredConceptYN="Y">
        <ConceptName><String>Cardiovascular Diseases</String></ConceptName>
        <ScopeNote>Heart and vessel diseases.</ScopeNote>
        <TermList>
          <Term ConceptPreferredTermYN="Y" RecordPreferredTermYN="Y"><String>Cardiovascular Diseases</String></Term>
        </TermList>
      </Concept>
    </ConceptList>
  </DescriptorRecord>
  <DescriptorRecord DescriptorClass="1">
    <DescriptorUI>D000002</DescriptorUI>
    <DescriptorName><String>Myocardial Infarction</String></DescriptorName>
    <TreeNumberList><TreeNumber>C14.280</TreeNumber></TreeNumberList>
    <HistoryNote>Formerly indexed under heart diseases.</HistoryNote>
    <PreviousIndexingList><PreviousIndexing>HEART DISEASES (1965-1990)</PreviousIndexing></PreviousIndexingList>
    <SeeRelatedList><String>Acute Coronary Syndrome</String></SeeRelatedList>
    <AllowableQualifiersList>
      <AllowableQualifier>
        <QualifierReferredTo>
          <QualifierUI>Q000001</QualifierUI>
          <QualifierName><String>therapy</String></QualifierName>
        </QualifierReferredTo>
        <Abbreviation>TH</Abbreviation>
      </AllowableQualifier>
    </AllowableQualifiersList>
    <PharmacologicalActionList>
      <PharmacologicalAction>
        <DescriptorReferredTo>
          <DescriptorUI>D999001</DescriptorUI>
          <DescriptorName><String>Cardioprotective Agents</String></DescriptorName>
        </DescriptorReferredTo>
      </PharmacologicalAction>
    </PharmacologicalActionList>
    <ConceptList>
      <Concept PreferredConceptYN="Y">
        <ConceptName><String>Myocardial Infarction</String></ConceptName>
        <ScopeNote>Necrosis of heart muscle.</ScopeNote>
        <TermList>
          <Term ConceptPreferredTermYN="Y" RecordPreferredTermYN="Y"><String>Myocardial Infarction</String></Term>
          <Term><String>Heart Attack</String></Term>
        </TermList>
      </Concept>
    </ConceptList>
  </DescriptorRecord>
  <DescriptorRecord DescriptorClass="1">
    <DescriptorUI>D000003</DescriptorUI>
    <DescriptorName><String>Telemedicine</String></DescriptorName>
    <TreeNumberList><TreeNumber>N04.590</TreeNumber></TreeNumberList>
    <ConceptList>
      <Concept PreferredConceptYN="Y">
        <ConceptName><String>Telemedicine</String></ConceptName>
        <ScopeNote>Delivery of health services at a distance.</ScopeNote>
        <TermList>
          <Term ConceptPreferredTermYN="Y" RecordPreferredTermYN="Y"><String>Telemedicine</String></Term>
          <Term><String>Remote Consultation</String></Term>
        </TermList>
      </Concept>
    </ConceptList>
  </DescriptorRecord>
</DescriptorRecordSet>
"""


QUALIFIER_XML = """<?xml version="1.0"?>
<QualifierRecordSet LanguageCode="eng">
  <QualifierRecord>
    <QualifierUI>Q000001</QualifierUI>
    <QualifierName><String>therapy</String></QualifierName>
    <HistoryNote>Therapy qualifier history.</HistoryNote>
    <TreeNumberList><TreeNumber>Y01</TreeNumber></TreeNumberList>
    <ConceptList>
      <Concept PreferredConceptYN="Y">
        <ConceptName><String>therapy</String></ConceptName>
        <ScopeNote>Treatment aspects.</ScopeNote>
        <TermList>
          <Term ConceptPreferredTermYN="Y" RecordPreferredTermYN="Y"><String>therapy</String></Term>
          <Term><String>treatment</String></Term>
        </TermList>
      </Concept>
    </ConceptList>
  </QualifierRecord>
</QualifierRecordSet>
"""


SUPP_XML = """<?xml version="1.0"?>
<SupplementalRecordSet LanguageCode="eng">
  <SupplementalRecord SCRClass="1">
    <SupplementalRecordUI>C000001</SupplementalRecordUI>
    <SupplementalRecordName><String>telehealth</String></SupplementalRecordName>
    <Note>Consumer-facing telehealth term.</Note>
    <HeadingMappedToList>
      <HeadingMappedTo>
        <DescriptorReferredTo>
          <DescriptorUI>D000003</DescriptorUI>
          <DescriptorName><String>Telemedicine</String></DescriptorName>
        </DescriptorReferredTo>
      </HeadingMappedTo>
    </HeadingMappedToList>
    <ConceptList>
      <Concept PreferredConceptYN="Y">
        <ConceptName><String>telehealth</String></ConceptName>
        <TermList>
          <Term ConceptPreferredTermYN="Y" RecordPreferredTermYN="Y"><String>telehealth</String></Term>
        </TermList>
      </Concept>
    </ConceptList>
  </SupplementalRecord>
</SupplementalRecordSet>
"""


class MeshKnowledgeBaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.mesh_dir = self.root / "mesh"
        self.processed_dir = self.root / "processed"
        self.mesh_dir.mkdir(parents=True, exist_ok=True)
        (self.mesh_dir / "desc2026.xml").write_text(DESCRIPTOR_XML, encoding="utf-8")
        (self.mesh_dir / "qual2026.xml").write_text(QUALIFIER_XML, encoding="utf-8")
        (self.mesh_dir / "supp2026.xml").write_text(SUPP_XML, encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_parser_extracts_descriptor_metadata(self) -> None:
        parser = MeshParser()
        descriptors = parser.parse_descriptors(self.mesh_dir / "desc2026.xml")
        record = descriptors["D000002"]

        self.assertEqual(record.preferred_name, "Myocardial Infarction")
        self.assertIn("Heart Attack", record.entry_terms)
        self.assertEqual(record.tree_numbers, ["C14.280"])
        self.assertEqual(record.scope_note, "Necrosis of heart muscle.")
        self.assertEqual(record.allowable_qualifiers[0]["name"], "therapy")
        self.assertEqual(record.pharmacological_actions[0]["name"], "Cardioprotective Agents")
        self.assertEqual(record.previous_indexing, ["HEART DISEASES (1965-1990)"])
        self.assertEqual(record.see_related, ["Acute Coronary Syndrome"])
        self.assertEqual(record.history_note, "Formerly indexed under heart diseases.")

    def test_builder_creates_graph_and_processed_files(self) -> None:
        builder = MeshIndexBuilder()
        artifacts = builder.build(self.mesh_dir, self.processed_dir)

        self.assertTrue((self.processed_dir / "mesh_descriptors.json").exists())
        self.assertTrue((self.processed_dir / "mesh_graph.json").exists())
        self.assertTrue((self.processed_dir / "mesh_lookup.pkl").exists())

        graph = artifacts["graph"]
        self.assertEqual(graph["D000002"]["parents"], ["D000001"])
        self.assertEqual(graph["D000001"]["children"], ["D000002"])
        self.assertEqual(graph["D000002"]["ancestors"], ["D000001"])
        self.assertEqual(graph["D000001"]["descendants"], ["D000002"])

        payload = json.loads((self.processed_dir / "mesh_descriptors.json").read_text(encoding="utf-8"))
        self.assertIn("D000003", payload["descriptors"])
        self.assertIn("Q000001", payload["qualifiers"])
        self.assertIn("C000001", payload["supplementary_records"])

    def test_builder_supports_missing_optional_mesh_files(self) -> None:
        (self.mesh_dir / "qual2026.xml").unlink()
        (self.mesh_dir / "supp2026.xml").unlink()

        artifacts = MeshIndexBuilder().build(self.mesh_dir, self.processed_dir)

        self.assertEqual(len(artifacts["descriptors"]), 3)
        self.assertEqual(artifacts["qualifiers"], {})
        self.assertEqual(artifacts["supplementary_records"], {})

        payload = json.loads((self.processed_dir / "mesh_descriptors.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["qualifiers"], {})
        self.assertEqual(payload["supplementary_records"], {})

    def test_loader_supports_exact_and_graph_lookup(self) -> None:
        kb = MeshKnowledgeBase(mesh_dir=self.mesh_dir, processed_dir=self.processed_dir)

        exact = kb.lookup_by_term("Heart Attack")
        self.assertEqual([item.descriptor_ui for item in exact], ["D000002"])

        telemedicine = kb.lookup_by_mesh_id("D000003")
        self.assertIsNotNone(telemedicine)
        self.assertEqual(telemedicine.preferred_name, "Telemedicine")

        parents = kb.get_parents("D000002")
        self.assertEqual([item.descriptor_ui for item in parents], ["D000001"])
        ancestors = kb.get_ancestors("D000002")
        self.assertEqual([item.descriptor_ui for item in ancestors], ["D000001"])
        children = kb.get_children("D000001")
        self.assertEqual([item.descriptor_ui for item in children], ["D000002"])
        descendants = kb.get_descendants("D000001")
        self.assertEqual([item.descriptor_ui for item in descendants], ["D000002"])
        self.assertEqual(kb.get_synonyms("D000003"), ["Remote Consultation"])

    def test_fuzzy_search_returns_expected_descriptor(self) -> None:
        kb = MeshKnowledgeBase(mesh_dir=self.mesh_dir, processed_dir=self.processed_dir)

        heart_results = kb.search("heart attack")
        self.assertGreaterEqual(len(heart_results), 1)
        self.assertEqual(heart_results[0].preferred_name, "Myocardial Infarction")

        telehealth_results = kb.search("telehealth")
        self.assertGreaterEqual(len(telehealth_results), 1)
        self.assertEqual(telehealth_results[0].preferred_name, "Telemedicine")

    def test_loader_recovers_from_corrupt_lookup_pickle(self) -> None:
        builder = MeshIndexBuilder()
        builder.build(self.mesh_dir, self.processed_dir)
        (self.processed_dir / "mesh_lookup.pkl").write_bytes(b"broken")

        kb = MeshKnowledgeBase(mesh_dir=self.mesh_dir, processed_dir=self.processed_dir)

        results = kb.search("telehealth")
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].preferred_name, "Telemedicine")

    def test_build_script_reports_partial_build_and_writes_outputs(self) -> None:
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_mesh_kb.py"
        spec = importlib.util.spec_from_file_location("build_mesh_kb", script_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        backend_dir = self.root / "backend"
        mesh_dir = backend_dir / "knowledge" / "mesh"
        processed_dir = backend_dir / "knowledge" / "processed"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        (mesh_dir / "desc2026.xml").write_text(DESCRIPTOR_XML, encoding="utf-8")
        (mesh_dir / "pa2026.xml").write_text("<PharmacologicalActionSet />", encoding="utf-8")

        module.BACKEND_DIR = backend_dir

        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = module.main()

        self.assertEqual(exit_code, 0)
        self.assertIn("desc2026.xml: found", stdout.getvalue())
        self.assertIn("qual2026.xml: missing", stdout.getvalue())
        self.assertIn("supp2026.xml: missing", stdout.getvalue())
        self.assertIn("pa2026.xml: found", stdout.getvalue())
        self.assertIn("descriptor count: 3", stdout.getvalue())
        self.assertIn("qualifier count: 0", stdout.getvalue())
        self.assertIn("supplementary record count: 0", stdout.getvalue())
        self.assertIn("pa2026.xml is present but not yet processed", stdout.getvalue())
        self.assertIn("continuing with descriptors only", stderr.getvalue())
        self.assertIn("continuing without supplementary records", stderr.getvalue())
        self.assertTrue((processed_dir / "mesh_descriptors.json").exists())
        self.assertTrue((processed_dir / "mesh_graph.json").exists())
        self.assertTrue((processed_dir / "mesh_lookup.pkl").exists())
