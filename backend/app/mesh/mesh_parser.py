from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from .mesh_models import MeshDescriptor, MeshQualifier, MeshSupplementaryRecord


def _text(element: ET.Element | None, path: str, default: str = "") -> str:
    if element is None:
        return default
    found = element.find(path)
    if found is None or found.text is None:
        return default
    return found.text.strip()


def _texts(element: ET.Element | None, path: str) -> list[str]:
    if element is None:
        return []
    return [
        item.text.strip()
        for item in element.findall(path)
        if item.text and item.text.strip()
    ]


def _unique_terms(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        lowered = normalized.lower()
        if not normalized or lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(normalized)
    return deduped


class MeshParser:
    def parse_descriptors(self, xml_path: str | Path) -> dict[str, MeshDescriptor]:
        descriptors: dict[str, MeshDescriptor] = {}
        for _, element in ET.iterparse(xml_path, events=("end",)):
            if element.tag != "DescriptorRecord":
                continue

            descriptor_ui = _text(element, "DescriptorUI")
            preferred_name = _text(element, "DescriptorName/String")
            descriptor = MeshDescriptor(
                descriptor_ui=descriptor_ui,
                preferred_name=preferred_name,
                entry_terms=self._descriptor_entry_terms(element, preferred_name),
                tree_numbers=_texts(element, "TreeNumberList/TreeNumber"),
                scope_note=self._descriptor_scope_note(element),
                allowable_qualifiers=self._allowable_qualifiers(element),
                pharmacological_actions=self._descriptor_refs(
                    element,
                    "PharmacologicalActionList/PharmacologicalAction/DescriptorReferredTo",
                    id_tag="DescriptorUI",
                    name_tag="DescriptorName/String",
                ),
                previous_indexing=_texts(element, "PreviousIndexingList/PreviousIndexing"),
                see_related=self._see_related_terms(element),
                history_note=_text(element, "HistoryNote") or None,
            )
            descriptors[descriptor_ui] = descriptor
            element.clear()
        return descriptors

    def parse_qualifiers(self, xml_path: str | Path) -> dict[str, MeshQualifier]:
        qualifiers: dict[str, MeshQualifier] = {}
        for _, element in ET.iterparse(xml_path, events=("end",)):
            if element.tag != "QualifierRecord":
                continue

            qualifier_ui = _text(element, "QualifierUI")
            preferred_name = _text(element, "QualifierName/String")
            qualifier = MeshQualifier(
                qualifier_ui=qualifier_ui,
                name=preferred_name,
                history_note=_text(element, "HistoryNote") or None,
                tree_numbers=_texts(element, "TreeNumberList/TreeNumber"),
                scope_note=self._preferred_concept_scope_note(element),
                entry_terms=self._concept_terms(element, preferred_name),
            )
            qualifiers[qualifier_ui] = qualifier
            element.clear()
        return qualifiers

    def parse_supplementary_records(self, xml_path: str | Path) -> dict[str, MeshSupplementaryRecord]:
        records: dict[str, MeshSupplementaryRecord] = {}
        for _, element in ET.iterparse(xml_path, events=("end",)):
            if element.tag != "SupplementalRecord":
                continue

            supplemental_ui = _text(element, "SupplementalRecordUI")
            preferred_name = _text(element, "SupplementalRecordName/String")
            record = MeshSupplementaryRecord(
                supplemental_ui=supplemental_ui,
                preferred_name=preferred_name,
                entry_terms=self._concept_terms(element, preferred_name),
                mapped_descriptors=self._descriptor_refs(
                    element,
                    "HeadingMappedToList/HeadingMappedTo/DescriptorReferredTo",
                    id_tag="DescriptorUI",
                    name_tag="DescriptorName/String",
                ),
                pharmacological_actions=self._descriptor_refs(
                    element,
                    "PharmacologicalActionList/PharmacologicalAction/DescriptorReferredTo",
                    id_tag="DescriptorUI",
                    name_tag="DescriptorName/String",
                ),
                previous_indexing=_texts(element, "PreviousIndexingList/PreviousIndexing"),
                note=_text(element, "Note") or None,
            )
            records[supplemental_ui] = record
            element.clear()
        return records

    def _descriptor_entry_terms(self, element: ET.Element, preferred_name: str) -> list[str]:
        terms = self._concept_terms(element, preferred_name)
        concept_names = _texts(element, "ConceptList/Concept/ConceptName/String")
        combined = _unique_terms(terms + concept_names)
        return [term for term in combined if term.lower() != preferred_name.lower()]

    def _concept_terms(self, element: ET.Element, preferred_name: str) -> list[str]:
        terms = _texts(element, "ConceptList/Concept/TermList/Term/String")
        return [term for term in _unique_terms(terms) if term.lower() != preferred_name.lower()]

    def _preferred_concept_scope_note(self, element: ET.Element) -> str | None:
        for concept in element.findall("ConceptList/Concept"):
            if concept.attrib.get("PreferredConceptYN") == "Y":
                scope_note = _text(concept, "ScopeNote")
                return scope_note or None
        return None

    def _descriptor_scope_note(self, element: ET.Element) -> str | None:
        return self._preferred_concept_scope_note(element) or (_text(element, "Annotation") or None)

    def _allowable_qualifiers(self, element: ET.Element) -> list[dict[str, str | None]]:
        qualifiers: list[dict[str, str | None]] = []
        for item in element.findall("AllowableQualifiersList/AllowableQualifier"):
            qualifiers.append(
                {
                    "qualifier_ui": _text(item, "QualifierReferredTo/QualifierUI") or None,
                    "name": _text(item, "QualifierReferredTo/QualifierName/String") or None,
                    "abbreviation": _text(item, "Abbreviation") or None,
                }
            )
        return [item for item in qualifiers if item["name"]]

    def _descriptor_refs(
        self,
        element: ET.Element,
        path: str,
        *,
        id_tag: str,
        name_tag: str,
    ) -> list[dict[str, str]]:
        refs: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for item in element.findall(path):
            ref_id = _text(item, id_tag).lstrip("*")
            ref_name = _text(item, name_tag)
            key = (ref_id, ref_name)
            if not ref_id or not ref_name or key in seen:
                continue
            seen.add(key)
            refs.append({"ui": ref_id, "name": ref_name})
        return refs

    def _see_related_terms(self, element: ET.Element) -> list[str]:
        values: list[str] = []
        values.extend(_texts(element, "SeeRelatedList/String"))
        values.extend(_texts(element, "SeeRelatedList/DescriptorReferredTo/DescriptorName/String"))
        values.extend(_texts(element, "SeeRelatedList/SeeRelatedDescriptor/DescriptorReferredTo/DescriptorName/String"))
        return _unique_terms(values)
