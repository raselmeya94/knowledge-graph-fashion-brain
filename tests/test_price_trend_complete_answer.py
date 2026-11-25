"""Price-trend answers must ship the full retrieve package (table + chart + examples)."""

from __future__ import annotations

from fashion_brain.agents.answer_card import (
    AnswerCard,
    SourceResult,
    _findings_look_like_complete_price_trend,
    _price_trend_fallback,
    validate,
    write_answer,
)

_COMPLETE = """
Average same-currency price change trend for C&A:
- Average change: -17.63% (decreasing)
- Change events: 54
- Products with changes: 52
- Direction mix: 3 up, 51 down, 0 unchanged
- Range: -33.34% to 30.03%
Event % change trail (same-currency) — min -33.34% · avg -17.63% · max 30.03%:
| Period | Change % | Events |
| --- | ---: | ---: |
| 11-02T15:33 | -23.09 | 1 |
| 11-05T14:36 | 30.03 | 1 |
| 11-05T14:38 | -33.34 | 1 |
| 11-07T11:01 | -20.00 | 1 |
| … | (16 points plotted) | … |
```chart
{"type": "line", "smooth": true, "title": "Price change trends · C&A", "labels": ["a", "b"], "values": [-23.0, -20.0], "series": [{"name": "Overall", "values": [-23.0, -20.0]}], "yLabel": "Change %"}
```
Probable drivers (from recorded events only):
- Markdown pressure: 51 down vs 3 up moves (94% of directional events are decreases).
Example moves:
- Multipack of 2 - V-neck T-shirt: 29.99 → 19.99 (-33.3%)
- Jacket - faux leather: 39.99 → 29.99 (-25.0%)
""".strip()


def _card() -> AnswerCard:
    return AnswerCard(
        question="What is the price trend for C&A?",
        intent="price_trend",
        filters={"brand": "C&A"},
        findings=_COMPLETE,
        results=[SourceResult(source_id="ecom_products", total=54)],
        caveats=["Trend line uses same-currency delta_pct; currency switches excluded."],
        depth="full",
    )


def test_complete_findings_detected():
    assert _findings_look_like_complete_price_trend(_COMPLETE) is True


def test_truncated_draft_fails_validation():
    card = _card()
    truncated = """
### Price Trend Analysis: C&A
Average change -17.63%.

Period | Change % | Events
| 11-02T15:33 |
""".strip()
    issues = validate(truncated, card)
    assert any(
        key in i
        for i in issues
        for key in ("missing_chart", "incomplete_period_table", "truncated_mid_table", "missing_examples")
    )


def test_write_answer_ships_full_findings():
    card = _card()

    def _boom(_prompt: str) -> str:
        raise AssertionError("LLM should not be called when findings are complete")

    out = write_answer(card, _boom)
    assert "```chart" in out
    assert "Example moves" in out
    assert "11-05T14:38" in out
    assert "-17.63%" in out
    assert "Multipack of 2" in out


def test_price_trend_fallback_includes_chart():
    out = _price_trend_fallback(_card())
    assert "```chart" in out
    assert "Example moves" in out
