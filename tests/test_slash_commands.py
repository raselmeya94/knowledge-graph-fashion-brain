"""Slash triggers: free-text bodies; design never catalog-rewrites."""

from __future__ import annotations

from fashion_brain.agents.slash_commands import (
    parse_slash_command,
    slash_needs_clarify,
    slash_rewrite_question,
)
from fashion_brain.ask import garment_type_from_question


def test_parse_slash_commands_and_typos():
    assert parse_slash_command("/trending").name == "trending"
    assert parse_slash_command("/specification").name == "specification"
    assert parse_slash_command("/spec anything").name == "specification"
    assert parse_slash_command("/garment_design").name == "garment_design"
    assert parse_slash_command("/garment_desing").name == "garment_design"
    assert parse_slash_command("/advanced_garment_design").name == "advanced_garment_design"
    assert parse_slash_command("/advance_garment_design").name == "advanced_garment_design"
    assert parse_slash_command("/advanced_garment_desing").name == "advanced_garment_design"
    assert parse_slash_command("make a product specification") is None


def test_empty_command_clarifies_nonempty_runs():
    empty = parse_slash_command("/trending")
    assert slash_needs_clarify(empty, has_specification=False)
    assert slash_rewrite_question(empty) is None

    filled = parse_slash_command("/trending whatever scope the user typed")
    assert slash_needs_clarify(filled, has_specification=False) is None
    rewritten = slash_rewrite_question(filled)
    assert rewritten is not None
    assert "whatever scope the user typed" in rewritten


def test_design_trigger_never_rewrites_to_catalog():
    bare = parse_slash_command("/advanced_garment_design")
    assert slash_rewrite_question(bare) is None

    with_note = parse_slash_command("/advanced_garment_design make sleeves longer")
    assert slash_rewrite_question(with_note) is None

    sheet = parse_slash_command(
        "/advanced_garment_desing\n"
        "Buyer: primark\n"
        "Season: winter\n"
        "garment type: Jumper\n"
        "category: Sweater\n"
        "color: Brown\n"
    )
    assert sheet.name == "advanced_garment_design"
    assert slash_rewrite_question(sheet) is None
    assert "Buyer: primark" in sheet.body


def test_specification_passthrough():
    cmd = parse_slash_command("/specification need a ladies knit for our winter drop")
    assert slash_needs_clarify(cmd, has_specification=False) is None
    rewritten = slash_rewrite_question(cmd)
    assert "product specification" in rewritten.casefold()


def test_garment_design_clarify_without_spec():
    cmd = parse_slash_command("/advanced_garment_design")
    note = slash_needs_clarify(cmd, has_specification=False)
    assert note
    assert slash_needs_clarify(cmd, has_specification=True) is None


def test_ranking_top_is_not_a_garment():
    assert garment_type_from_question("top counts by category") is None
    assert garment_type_from_question("what is trending top brands") is None
    assert garment_type_from_question("crop top for summer") == "crop-top"
