"""Facet nodes keep human display names; matching uses normalized_name only."""

from __future__ import annotations

from knowledge_graph.ontology import facet_name_fields, normalize_name


def test_facet_name_fields_keeps_display_separates_key():
    fields = facet_name_fields("C&A")
    assert fields["name"] == "C&A"
    assert fields["normalized_name"] == "c a"
    assert fields["normalized_name"] == normalize_name("C&A")


def test_facet_name_fields_preserves_casing_and_symbols():
    assert facet_name_fields("H&M") == {"name": "H&M", "normalized_name": "h m"}
    assert facet_name_fields("UK Showroom Sample") == {
        "name": "UK Showroom Sample",
        "normalized_name": "uk showroom sample",
    }
    assert facet_name_fields("Cotton 100%") == {
        "name": "Cotton 100%",
        "normalized_name": "cotton 100",
    }


def test_facet_name_fields_empty_falls_back():
    assert facet_name_fields("") == {"name": "", "normalized_name": ""}
    assert facet_name_fields(None) == {"name": "", "normalized_name": ""}
