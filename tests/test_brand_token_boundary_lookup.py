"""Soft brand lookup must not match grammar tokens inside longer brand names."""

from __future__ import annotations

from fashion_brain.ask import _is_grammar_party_name, _is_party_token, resolve_catalog_plan
from fashion_brain.planning import kg_catalog as kc


def test_yes_confirm_is_not_a_brand():
    assert _is_party_token("yes") is False
    assert _is_grammar_party_name("yes") is True
    assert resolve_catalog_plan("yes", None, carry=False).get("filters") == {}


def test_tell_is_probe_stopword():
    assert "tell" in kc._TOKEN_STOP
    assert "yes" in kc._TOKEN_STOP
    assert "color" in kc._TOKEN_STOP


def test_lookup_value_rejects_tell_inside_sophie_steller(monkeypatch):
    monkeypatch.setattr(
        kc,
        "_values_for",
        lambda *_a, **_k: ["sophie steller", "primark", "united colors of benetton"],
    )
    assert kc._lookup_value(None, "ecom_products", "brand", "tell") is None
    assert kc._lookup_value(None, "ecom_products", "brand", "color") is None
    assert kc._lookup_value(None, "ecom_products", "brand", "primark") == "primark"


def test_lookup_value_allows_whole_token_prefix(monkeypatch):
    monkeypatch.setattr(
        kc,
        "_values_for",
        lambda *_a, **_k: ["calvin klein", "primark"],
    )
    assert kc._lookup_value(None, "ecom_products", "brand", "calvin") == "calvin klein"
