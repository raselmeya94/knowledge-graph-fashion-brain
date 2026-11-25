"""Soft Neo4j value needles for every facet column."""

from __future__ import annotations

from knowledge_graph.retrieval.ecom import _filter_where, _soft_match_needles


def test_soft_needles_universal_plural_and_case():
    for asked, must_have in [
        ("Sweaters", {"sweaters", "sweater"}),
        ("COLORS", {"colors", "color"}),
        ("Women", {"women"}),
        ("navy blues", {"navy blues", "navy blue"}),
    ]:
        needles = set(_soft_match_needles(asked))
        assert must_have <= needles, (asked, needles)


def test_filter_where_soft_needles_all_common_fields():
    cases = [
        ("plmex_products", {"garment_type": "Jumpers"}, "f_garment_type_needles", "jumper"),
        ("plmex_products", {"gender": "Women"}, "f_gender_needles", "women"),
        ("plmex_products", {"color": "Whites"}, "color_needles", "white"),
        ("ecom_products", {"color": "Navy"}, "color_needles", "navy"),
        ("ecom_products", {"section": "Kids"}, "section_needles", "kids"),
        ("ecom_products", {"material": "Cottons"}, "material_needles", "cotton"),
        ("ecom_products", {"category": "Jackets"}, "category_needles", "jacket"),
        ("ecom_products", {"family": "Dresses"}, "f_family_needles", "dress"),
        ("plmex_products", {"department": "Knitwear"}, "f_department_needles", "knitwear"),
    ]
    for graph, filters, param_key, stem in cases:
        params: dict = {"g": graph}
        where = _filter_where(filters, params)
        assert where is not None, filters
        assert param_key in params, (filters, sorted(params))
        assert stem in params[param_key], (filters, params[param_key])
