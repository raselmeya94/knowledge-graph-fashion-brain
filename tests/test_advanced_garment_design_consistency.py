"""Multi-view board: shared seed + locked identity brief across shots."""

from __future__ import annotations

from fashion_brain.agents.product_spec import (
    ParsedSpec,
    SpecValue,
    build_advanced_shots,
)


def _spec() -> ParsedSpec:
    return ParsedSpec(
        values={
            "buyer": [SpecValue("primark", 10, 10)],
            "season": [SpecValue("winter", 10, 10)],
            "garment_type": [SpecValue("Jumper", 9, 10)],
            "category": [SpecValue("Sweater", 9, 10)],
            "color": [SpecValue("Brown", 6, 10)],
            "composition": [SpecValue("100% cotton", 7, 10)],
            "ai_construction_tags": [SpecValue("crew neck", 8, 10), SpecValue("rib hem", 5, 10)],
        }
    )


def test_advanced_shots_share_seed_and_locked_identity_block():
    shots = build_advanced_shots(_spec())
    assert len(shots) >= 4
    seeds = {shot.seed for shot in shots}
    assert len(seeds) == 1

    # IDENTITY / COLOUR / MANDATORY must be identical; only VIEW / EMPHASISE may differ.
    cores = []
    for shot in shots:
        assert "GARMENT IDENTITY LOCK" in shot.prompt
        assert "VIEW (" in shot.prompt
        # Split only on the real VIEW block marker (not "EMPHASISE … VIEW").
        before_view = shot.prompt.split("\nVIEW (", 1)[0]
        core = before_view.split("EMPHASISE FOR THIS ANGLE", 1)[0].rstrip()
        cores.append(core)
        assert shot.aspect_ratio == "4:5"
    assert len(set(cores)) == 1


def test_advanced_shots_do_not_embed_focus_inside_brief():
    shots = build_advanced_shots(_spec())
    for shot in shots:
        brief = shot.prompt.split("GARMENT IDENTITY LOCK", 1)[0]
        assert "EMPHASISE FOR THIS ANGLE" not in brief
