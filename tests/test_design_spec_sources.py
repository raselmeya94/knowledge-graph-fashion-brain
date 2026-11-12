"""Design commands resolve specs via the LLM Spec Resolve agent (mocked in tests)."""

from __future__ import annotations

from fashion_brain.agents import product_spec as spec_mod
from fashion_brain.agents.product_spec import (
    DesignCommand,
    ParsedSpec,
    SpecValue,
    plan_design,
)


def _sheet() -> ParsedSpec:
    return ParsedSpec(
        values={
            "buyer": [SpecValue("primark")],
            "season": [SpecValue("winter")],
            "garment_type": [SpecValue("Jumper")],
            "category": [SpecValue("Sweater")],
            "color": [SpecValue("Brown")],
            "composition": [SpecValue("100% cotton")],
        }
    )


def test_plan_design_uses_explicit_spec():
    plan = plan_design(
        DesignCommand(mode="advanced", note=""),
        history=[],
        spec=_sheet(),
    )
    assert len(plan.shots) >= 4
    assert plan.spec.text("buyer").lower() == "primark"


def test_plan_design_uses_llm_resolve_on_note(monkeypatch):
    monkeypatch.setattr(spec_mod, "resolve_specification", lambda text, platform=None: _sheet())
    plan = plan_design(
        DesignCommand(mode="advanced", note="any messy user paste"),
        history=[],
        platform=object(),
    )
    assert plan.spec.text("category").lower() == "sweater"
    assert "USER ADDITION" not in plan.shots[0].prompt


def test_plan_design_uses_history_via_llm(monkeypatch):
    monkeypatch.setattr(spec_mod, "resolve_specification", lambda text, platform=None: _sheet())
    history = [{"role": "assistant", "content": "whatever format the user had"}]
    plan = plan_design(DesignCommand(mode="single", note=""), history, platform=object())
    assert plan.spec.text("color").lower() == "brown"


def test_plan_design_keeps_creative_note_when_note_is_not_a_sheet(monkeypatch):
    monkeypatch.setattr(spec_mod, "resolve_specification", lambda text, platform=None: None)
    monkeypatch.setattr(spec_mod, "extract_last_specification", lambda history, platform=None: _sheet())
    plan = plan_design(
        DesignCommand(mode="single", note="keep neckline clean"),
        history=[{"role": "assistant", "content": "prior"}],
        platform=object(),
    )
    assert "keep neckline clean" in plan.shots[0].prompt.casefold()


def test_fields_to_spec_skips_empty_values():
    spec = spec_mod._fields_to_spec(
        {
            "buyer": "primark",
            "garment_type": "Jumper",
            "category": "Sweater",
            "gsm": "(no usable value in matching products)",
            "color": "",
        }
    )
    assert "gsm" not in spec.values
    assert "color" not in spec.values
    assert spec.text("buyer").lower() == "primark"
