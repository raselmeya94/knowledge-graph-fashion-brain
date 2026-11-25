"""Graph-aware semantic field aliases (ecom ↔ plmex)."""

from __future__ import annotations

from fashion_brain.planning.semantic_field_map import (
    apply_semantic_field_map,
    category_like_fallbacks,
    material_like_fallbacks,
    remap_group_by,
    resolve_group_by_candidates,
)


def test_ecom_gender_to_section():
    out, gb = apply_semantic_field_map({"gender": "women", "brand": "primark"}, "ecom_products", group_by="gender")
    assert out == {"section": "women", "brand": "primark"}
    assert gb == "section"


def test_plmex_section_to_gender():
    out, gb = apply_semantic_field_map({"section": "Women"}, "plmex_products", group_by="section")
    assert out == {"gender": "Women"}
    assert gb == "gender"


def test_ecom_garment_type_to_subfamily():
    out, _ = apply_semantic_field_map({"garment_type": "Jumpers"}, "ecom_products")
    assert out == {"subfamily": "Jumpers"}


def test_plmex_subfamily_to_garment_type():
    out, _ = apply_semantic_field_map({"subfamily": "jumper"}, "plmex_products")
    assert out == {"garment_type": "jumper"}


def test_matalan_brand_to_buyer_on_plmex():
    out, _ = apply_semantic_field_map({"brand": "matalan", "category": "sweater"}, "plmex_products")
    assert out.get("buyer") == "matalan"
    assert "brand" not in out
    assert out.get("category") == "sweater"  # never remapped


def test_primark_brand_stays_brand_on_plmex():
    out, _ = apply_semantic_field_map({"brand": "primark"}, "plmex_products")
    assert out.get("brand") == "primark"
    assert "buyer" not in out


def test_ecom_buyer_to_brand():
    out, _ = apply_semantic_field_map({"buyer": "matalan"}, "ecom_products")
    assert out == {"brand": "matalan"}


def test_category_never_remapped():
    out, _ = apply_semantic_field_map({"category": "Clothes"}, "plmex_products")
    assert out == {"category": "Clothes"}
    out2, _ = apply_semantic_field_map({"category": "sweater"}, "ecom_products")
    assert out2 == {"category": "sweater"}


def test_composition_to_material():
    out, _ = apply_semantic_field_map({"composition": "cotton"}, "ecom_products")
    assert out == {"material": "cotton"}


def test_colour_to_color():
    out, _ = apply_semantic_field_map({"colour": "navy"}, "plmex_products")
    assert out == {"color": "navy"}


def test_unknown_graph_unchanged():
    raw = {"gender": "women", "garment_type": "jumper"}
    out, gb = apply_semantic_field_map(raw, None, group_by="gender")
    assert out["gender"] == "women"
    assert out["garment_type"] == "jumper"
    assert gb == "gender"


def test_remap_group_by_helper():
    assert remap_group_by("gender", "ecom_products") == "section"
    assert remap_group_by("section", "plmex_products") == "gender"


def test_ecom_category_group_by_candidates_include_role_siblings():
    """LLM says by category; ecom may store mix in family/section/subfamily."""

    cands = resolve_group_by_candidates("category", "ecom_products")
    assert cands[0] == "category"
    assert "family" in cands
    assert "section" in cands
    assert "subfamily" in cands
    assert category_like_fallbacks("ecom_products")[0] == "category"


def test_plmex_category_group_by_candidates_include_plm_hierarchy():
    cands = resolve_group_by_candidates("category", "plmex_products")
    assert cands[0] == "category"
    assert "plm_main_category" in cands
    assert "garment_type" in cands


def test_ecom_garment_type_group_by_remaps_then_falls_back():
    cands = resolve_group_by_candidates("garment_type", "ecom_products")
    assert cands[0] == "subfamily"
    assert "category" in cands
    assert "family" in cands


def test_material_group_by_falls_back_to_composition_siblings():
    cands = resolve_group_by_candidates("material", "ecom_products")
    assert cands[0] == "material"
    assert "composition" in cands
    assert "construction" in cands
    assert "category" not in cands


def test_construction_group_by_falls_back_to_composition_on_plmex():
    """LLM says by construction; PLM stores blends on composition/material."""

    cands = resolve_group_by_candidates("construction", "plmex_products")
    assert cands[0] == "construction"
    assert cands[1] == "composition"
    assert "material" in cands
    assert material_like_fallbacks("plmex_products")[1] == "composition"


def test_materials_alias_normalizes_to_material():
    assert resolve_group_by_candidates("materials", "plmex_products")[0] == "material"


def test_idempotent():
    once, _ = apply_semantic_field_map({"gender": "women"}, "ecom_products")
    twice, _ = apply_semantic_field_map(once, "ecom_products")
    assert once == twice == {"section": "women"}
