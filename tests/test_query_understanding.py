"""Query Understanding Agent — progressive clarification before retrieval."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fashion_brain.agents.contracts import ClarificationState, ClarifyResult, ReadyResult
from fashion_brain.agents.query_understanding import (
    QueryUnderstandingAgent,
    apply_validated_intent,
    clarification_awaiting,
    store_understanding,
)
from fashion_brain.agents.contracts import Intention, IntentSummary, Retrieved, ValidatedIntent


def _platform(*, answer_mode: str = "template"):
    """Default template mode skips QU LLM so heuristic unit tests stay offline."""

    return SimpleNamespace(
        settings=SimpleNamespace(answer_mode=answer_mode, request_timeout_seconds=5, generated=None),
        kg_registry=None,
    )


def _dims(*attrs):
    return list(attrs) or ["category", "color", "gender", "material", "season", "price"]


class TestQueryUnderstandingAgent:
    def test_named_analyze_live_brand_auto_runs(self):
        """Exact live-brand 'analyze Primark' runs immediately (no final_confirm quiz)."""

        agent = QueryUnderstandingAgent()
        with (
            patch.object(agent, "_available_dimensions", return_value=_dims()),
            patch(
                "fashion_brain.agents.qu_fast.resolve_entity",
                return_value=("primark", []),
            ),
            patch(
                "fashion_brain.agents.query_understanding.resolve_entity",
                return_value=("primark", []),
                create=True,
            ),
        ):
            ready = agent.evaluate(
                _platform(), "Analyze PRIMARK", history=None, conversation_state={}
            )
        assert isinstance(ready, ReadyResult)
        assert ready.intent.analysis_depth == "full"
        assert str(ready.intent.filters.get("brand") or "").casefold() == "primark"
        assert ready.state and ready.state.confirmed.get("auto_ran") is True

    def test_final_confirm_yes_runs_full_analysis(self):
        """Pending final_confirm + yes → full analysis without a second LLM call."""

        agent = QueryUnderstandingAgent()
        clar = ClarificationState(
            pending_slot="final_confirm",
            confirmed={
                "entity": "primark",
                "filters": {"brand": "primark"},
                "analysis_depth": "full",
            },
            original_question="Analyze PRIMARK",
            turn_count=1,
        )
        with patch.object(agent, "_available_dimensions", return_value=_dims()):
            ready = agent.evaluate(
                _platform(),
                "yes",
                conversation_state={"clarification": clar.to_dict()},
            )
        assert isinstance(ready, ReadyResult)
        assert ready.intent.analysis_depth == "full"
        assert ready.intent.filters.get("brand") == "primark"

    def test_final_confirm_no_asks_what_to_change(self):
        agent = QueryUnderstandingAgent()
        clar = ClarificationState(
            pending_slot="final_confirm",
            confirmed={
                "entity": "primark",
                "filters": {"brand": "primark"},
                "analysis_depth": "full",
            },
            original_question="Analyze PRIMARK",
            turn_count=1,
        )
        result = agent.evaluate(
            _platform(),
            "no",
            conversation_state={"clarification": clar.to_dict()},
        )
        assert isinstance(result, ClarifyResult)
        assert result.slot == "scope_or_direction"
        assert "change" in result.question.casefold()

    def test_price_trend_for_zara_is_ready_not_final_confirm(self):
        """Named-brand price trend must not become full-assortment final_confirm."""

        agent = QueryUnderstandingAgent()
        with patch.object(agent, "_available_dimensions", return_value=_dims()):
            result = agent.evaluate(
                _platform(),
                "make price trend analysis for Zara brand",
                history=None,
                conversation_state={},
            )
        assert isinstance(result, ReadyResult)
        assert "price" in [str(m).casefold() for m in (result.intent.metrics or [])]
        assert str(result.intent.filters.get("brand") or "").casefold() == "zara"
        assert result.intent.analysis_depth == "focused"
        assert "price" in (result.intent.effective_question or "").casefold()

    def test_final_confirm_yes_honors_price_trend_original(self):
        """If QU already asked final_confirm for a price-trend ask, yes → price metrics."""

        agent = QueryUnderstandingAgent()
        clar = ClarificationState(
            pending_slot="final_confirm",
            confirmed={
                "entity": "zara",
                "filters": {"brand": "zara"},
                "analysis_depth": "full",
            },
            original_question="make price trend analysis for Zara brand",
            turn_count=1,
        )
        with patch.object(agent, "_available_dimensions", return_value=_dims()):
            ready = agent.evaluate(
                _platform(),
                "yes",
                conversation_state={"clarification": clar.to_dict()},
            )
        assert isinstance(ready, ReadyResult)
        assert ready.intent.analysis_depth == "focused"
        assert "price" in [str(m).casefold() for m in (ready.intent.metrics or [])]
        assert str(ready.intent.filters.get("brand") or "").casefold() == "zara"

    def test_analyze_without_brand_asks_entity(self):
        agent = QueryUnderstandingAgent()
        with patch.object(agent, "_available_dimensions", return_value=_dims()):
            result = agent.evaluate(_platform(), "analyze", history=None, conversation_state={})
        assert isinstance(result, ClarifyResult)
        assert result.slot == "entity"
        assert "brand" in result.question.casefold() or "buyer" in result.question.casefold()

    def test_precise_count_skips_clarification(self):
        agent = QueryUnderstandingAgent()
        with patch.object(agent, "_available_dimensions", return_value=_dims()) as dims:
            result = agent.evaluate(
                _platform(),
                "How many red PRIMARK sweaters?",
                history=None,
                conversation_state={},
            )
        assert isinstance(result, ReadyResult)
        assert result.intent.analysis_depth == "unset"
        dims.assert_not_called()

    def test_precise_primark_count_grounds_live_brand_filter(self):
        """Regression: empty QU filters made kg_stats answer the global catalog total."""

        agent = QueryUnderstandingAgent()
        with patch(
            "fashion_brain.ask.live_brand_names",
            return_value=["Primark", "Zara", "Matalan"],
        ):
            result = agent.evaluate(
                _platform(),
                "How many products does Primark have?",
                history=None,
                conversation_state={},
            )
        assert isinstance(result, ReadyResult)
        assert "primark" in str(result.intent.filters.get("brand") or "").casefold()
        assert "count" in (result.intent.metrics or [])

    def test_explicit_analysis_party_does_not_inherit_stale_color(self):
        from fashion_brain.agents.qu_fast import clear_cache

        clear_cache()
        agent = QueryUnderstandingAgent()
        with (
            patch(
                "fashion_brain.agents.qu_fast.resolve_entity",
                return_value=("Primark", []),
            ),
            patch.object(agent, "_available_dimensions", return_value=_dims()),
        ):
            result = agent.evaluate(
                _platform(),
                "analysis primark",
                conversation_state={
                    "active_filters": {"brand": "primark", "color": "red"},
                },
            )
        # Exact live brand → auto-run full analysis; stale color must not carry.
        assert isinstance(result, ReadyResult)
        assert "primark" in str(result.intent.filters.get("brand") or "").casefold()
        assert "color" not in result.intent.filters

    def test_of_them_sweater_carries_session_brand(self):
        agent = QueryUnderstandingAgent()
        with patch(
            "fashion_brain.ask.live_brand_names",
            return_value=["Primark", "Zara"],
        ):
            result = agent.evaluate(
                _platform(),
                "And how many of them are sweaters?",
                conversation_state={"active_filters": {"brand": "primark", "buyer": "primark"}},
            )
        assert isinstance(result, ReadyResult)
        assert "primark" in str(result.intent.filters.get("brand") or "").casefold()
        assert "sweater" in str(result.intent.filters.get("garment") or "").casefold()

    def test_plm_data_is_graph_scope_not_brand(self):
        """'How many products are in the PLM data?' must not invent brand='are in plm'."""

        agent = QueryUnderstandingAgent()
        result = agent.evaluate(
            _platform(),
            "How many products are in the PLM data?",
            history=None,
            conversation_state={},
        )
        assert isinstance(result, ReadyResult)
        assert "brand" not in (result.intent.filters or {})
        assert "buyer" not in (result.intent.filters or {})
        assert result.intent.entity is None

    def test_progressive_sweater_then_full_analysis(self):
        """Incomplete 'analyze' still supports progressive filter trail."""

        agent = QueryUnderstandingAgent()
        dims = _dims()
        cats = ["Sweater", "Polo", "T-Shirt", "Trouser", "Jacket"]
        with (
            patch.object(agent, "_available_dimensions", return_value=dims),
            patch.object(agent, "_category_values", return_value=cats),
            patch(
                "fashion_brain.agents.query_understanding._basic_entity_brief",
                return_value="Basic from catalog metadata: primark currently has 1,397 products.",
            ),
            patch(
                "fashion_brain.agents.qu_fast.resolve_entity",
                side_effect=lambda _p, text, _g=None: (
                    ("primark", [])
                    if "primark" in (text or "").casefold()
                    else (None, [])
                ),
            ),
        ):
            first = agent.evaluate(_platform(), "analyze", conversation_state={})
            assert isinstance(first, ClarifyResult)
            assert first.slot == "entity"
            state = {"clarification": first.state.to_dict()}

            # With SKIP_ANALYSIS_FINAL_CONFIRM, naming the brand runs analysis immediately.
            named = agent.evaluate(_platform(), "primark", conversation_state=state)
            assert isinstance(named, ReadyResult)
            assert named.intent.analysis_depth == "full"
            assert str(named.intent.filters.get("brand") or "").casefold() == "primark"
            assert named.state is None or named.state.pending_slot is None
            ready = named
        assert isinstance(ready, ReadyResult)
        assert ready.intent.analysis_depth == "full"
        assert str(ready.intent.filters.get("brand") or "").casefold() == "primark"
        assert ready.state is None or ready.state.pending_slot is None

    def test_focused_color_path(self):
        agent = QueryUnderstandingAgent()
        dims = _dims()
        with (
            patch.object(agent, "_available_dimensions", return_value=dims),
            patch.object(agent, "_category_values", return_value=["Sweater"]),
            patch(
                "fashion_brain.agents.qu_fast.resolve_entity",
                side_effect=lambda _p, text, _g=None: (
                    ("primark", [])
                    if "primark" in (text or "").casefold()
                    else (None, [])
                ),
            ),
        ):
            first = agent.evaluate(_platform(), "analyze", conversation_state={})
            state = {"clarification": first.state.to_dict()}
            # Brand resolve auto-runs when final-confirm is skipped.
            named = agent.evaluate(_platform(), "primark", conversation_state=state)
            assert isinstance(named, ReadyResult)
            assert str(named.intent.filters.get("brand") or "").casefold() == "primark"
            ready = named
        assert isinstance(ready, ReadyResult)
        assert ready.intent.analysis_depth == "full"

    def test_options_never_invent_unavailable_fields(self):
        agent = QueryUnderstandingAgent()
        live = ["category", "color", "gender"]
        with (
            patch.object(agent, "_available_dimensions", return_value=live),
            patch.object(agent, "_category_values", return_value=["Sweater"]),
            patch(
                "fashion_brain.agents.qu_fast.resolve_entity",
                side_effect=lambda *_a, **_k: ("Zara", []),
            ),
        ):
            opener = agent.evaluate(_platform(), "analyze", conversation_state={})
            state = {"clarification": opener.state.to_dict()}
            result = agent.evaluate(_platform(), "zara", conversation_state=state)
        # Final-confirm skipped → naming the brand runs analysis immediately.
        assert isinstance(result, ReadyResult)
        assert "zara" in str(result.intent.filters.get("brand") or "").casefold()
        dims = list(result.intent.available_dimensions or [])
        assert "supplier" not in dims
        assert "sales" not in dims
        for opt in dims:
            assert opt in live or opt in _dims()

    def test_pending_slot_not_reasked_for_entity(self):
        agent = QueryUnderstandingAgent()
        clar = ClarificationState(
            pending_slot="scope_or_direction",
            confirmed={"entity": "primark", "filters": {"brand": "primark"}, "analysis_depth": "unset"},
            offered_options=["category", "color"],
            turn_count=1,
            original_question="Analyze PRIMARK",
        )
        with (
            patch.object(agent, "_available_dimensions", return_value=_dims()),
            patch.object(agent, "_category_values", return_value=["Sweater"]),
        ):
            result = agent.evaluate(
                _platform(),
                "Sweater",
                conversation_state={"clarification": clar.to_dict()},
            )
        assert isinstance(result, ClarifyResult)
        assert result.slot == "analysis_depth"
        assert "brand or buyer" not in result.question.casefold()

    def test_history_reuse_skips_reask_when_precise(self):
        agent = QueryUnderstandingAgent()
        from fashion_brain.agents.contracts import ContextDecision

        ctx = ContextDecision(
            decision="REFINE",
            is_contextual=True,
            active_filters={"brand": "primark", "category": "sweater"},
        )
        with patch.object(agent, "_available_dimensions", return_value=_dims()):
            result = agent.evaluate(
                _platform(),
                "How many are red?",
                history=[{"role": "user", "content": "PRIMARK sweaters"}],
                context=ctx,
                conversation_state={"active_filters": {"brand": "primark"}},
            )
        assert isinstance(result, ReadyResult)

    def test_gender_wise_breakdown_runs_without_entity_clarify(self):
        """Catalog-wide facet breakdown must not quiz for brand / alt dimensions."""

        agent = QueryUnderstandingAgent()
        with patch(
            "fashion_brain.planning.metadata_resolver.check_understanding",
            side_effect=AssertionError("must not clarify when group_by is known"),
        ):
            result = agent.evaluate(
                _platform(),
                "gender wise breakdown e-commerce data",
                conversation_state={},
            )
        assert isinstance(result, ReadyResult)
        assert result.intent.group_by == "gender"
        assert not result.intent.filters.get("brand")
        assert not result.intent.filters.get("buyer")

    def test_by_gender_alone_runs_without_entity_clarify(self):
        """group_by alone is catalog signal — do not fall through to entity quiz."""

        agent = QueryUnderstandingAgent()
        with patch(
            "fashion_brain.planning.metadata_resolver.check_understanding",
            side_effect=AssertionError("must not clarify when group_by is known"),
        ):
            result = agent.evaluate(
                _platform(),
                "products by gender",
                conversation_state={},
            )
        assert isinstance(result, ReadyResult)
        assert result.intent.group_by == "gender"

    def test_pending_entity_abandons_for_catalog_wide_gender_ask(self):
        """Stuck entity clarify + ecom gender breakdown → all products, not 'brand not found'."""

        agent = QueryUnderstandingAgent()
        clar = ClarificationState(
            pending_slot="entity",
            confirmed={},
            offered_options=["category", "composition", "availability"],
            turn_count=1,
            original_question="gender wise breakdown e-commerce data",
        )
        with (
            patch(
                "fashion_brain.agents.qu_fast.resolve_entity",
                return_value=(None, []),
            ),
            patch(
                "fashion_brain.planning.metadata_resolver.check_understanding",
                side_effect=AssertionError("must not re-clarify when group_by is known"),
            ),
        ):
            result = agent.evaluate(
                _platform(),
                "gender wise breakdown e-commerce data",
                conversation_state={"clarification": clar.to_dict()},
            )
        assert isinstance(result, ReadyResult)
        assert result.intent.group_by == "gender"
        assert result.intent.source == "ecom_products"
        assert not result.intent.filters.get("brand")
        assert not result.intent.filters.get("buyer")
        assert result.state is None or result.state.pending_slot is None

    def test_pending_entity_price_trend_is_not_full_analysis(self):
        """Stuck entity clarify + Primark price-trend → metrics=price, not assortment."""

        agent = QueryUnderstandingAgent()
        clar = ClarificationState(
            pending_slot="entity",
            confirmed={},
            offered_options=[],
            turn_count=1,
            original_question="analyze something",
        )
        ask = (
            "price-trend analysis for Primark on e-commerce: how recorded "
            "price-change events are moving — not another attribute-wise product-mix profile."
        )
        with patch(
            "fashion_brain.agents.qu_fast.resolve_entity",
            return_value=("primark", []),
        ):
            result = agent.evaluate(
                _platform(),
                ask,
                conversation_state={"clarification": clar.to_dict()},
            )
        assert isinstance(result, ReadyResult)
        assert "price" in (result.intent.metrics or [])
        assert result.intent.analysis_depth != "full"
        assert str(result.intent.filters.get("brand") or "").casefold() == "primark"
        assert result.state is None or result.state.pending_slot is None


class TestApplyValidatedIntent:
    def test_full_analysis_seeds_intention(self):
        from fashion_brain.agents.contracts import ValidatedIntent

        intention = Intention(intent="other", filters={})
        summary = IntentSummary(actual_intent="other", text="Analyze PRIMARK", clarification_required=True)
        validated = ValidatedIntent(
            entity="primark",
            filters={"brand": "primark", "category": "Sweater"},
            objective="full_analysis",
            analysis_depth="full",
            available_dimensions=["color", "gender"],
            effective_question="Full analysis of primark Sweater products",
        )
        out = apply_validated_intent(intention, validated, summary=summary)
        assert out.analysis is True
        assert out.filters["brand"] == "primark"
        assert out.filters["category"] == "Sweater"
        assert summary.clarification_required is False
        assert out.retrieve_mode == "analytical_tool"


class TestClarificationAwaiting:
    def test_caveat_flag(self):
        retrieved = Retrieved(
            text="What would you like to analyze?",
            evidence=[],
            caveats=[QueryUnderstandingAgent.AWAITING_CAVEAT],
            unavailable=[],
        )
        assert clarification_awaiting(retrieved) is True
        assert clarification_awaiting(Retrieved(text="ok", evidence=[], caveats=[], unavailable=[])) is False


class TestOrchestratorShortCircuit:
    def test_draft_catalog_returns_clarification_without_retrieve(self):
        from fashion_brain.orchestration.catalog_pipeline import draft_catalog_orchestrated
        from fashion_brain.agents.contracts import ContextDecision
        from knowledge_graph.access import Principal

        clarify = ClarifyResult(
            question="What would you like to analyze about PRIMARK?",
            state=ClarificationState(
                pending_slot="scope_or_direction",
                confirmed={"entity": "primark", "filters": {"brand": "primark"}},
                offered_options=["category", "color"],
                turn_count=1,
                original_question="Analyze PRIMARK",
            ),
            slot="scope_or_direction",
        )
        state: dict = {}
        with (
            patch(
                "fashion_brain.orchestration.catalog_pipeline.ConversationContextAgent.analyze",
                return_value=ContextDecision(decision="RETRIEVE"),
            ),
            patch(
                "fashion_brain.orchestration.catalog_pipeline.QueryUnderstandingAgent.evaluate",
                return_value=clarify,
            ),
            patch(
                "fashion_brain.orchestration.catalog_pipeline.IntentionAgent.understand"
            ) as understand,
            patch(
                "fashion_brain.orchestration.catalog_pipeline.ThinkingBrain.read"
            ) as thinking,
        ):
            retrieved, routes, intention, _ctx, choice = draft_catalog_orchestrated(
                _platform(),
                "Analyze PRIMARK",
                Principal(subject="t", clearance="internal"),
                conversation_state=state,
            )
        assert clarification_awaiting(retrieved)
        assert "clarification" in routes
        assert intention.intent == "clarification"
        assert choice is None
        assert state.get("clarification", {}).get("pending_slot") == "scope_or_direction"
        understand.assert_not_called()
        thinking.assert_not_called()

    def test_simple_count_answers_from_kg_stats_skips_thinking(self, tmp_path):
        """Named-source product totals use kg_stats for every catalog — no ThinkingBrain."""
        import json
        from fashion_brain.orchestration.catalog_pipeline import draft_catalog_orchestrated
        from fashion_brain.agents.contracts import ContextDecision, ValidatedIntent
        from knowledge_graph.access import Principal

        generated = tmp_path
        stats_dir = generated / "kg_stats"
        stats_dir.mkdir(parents=True)
        (stats_dir / "ecom_products.json").write_text(
            json.dumps(
                {
                    "version": 2,
                    "graph_id": "ecom_products",
                    "graph_name": "E-commerce",
                    "totals": {"products": 18183},
                    "unique_counts": {},
                    "value_counts": {},
                    "columns": [],
                }
            ),
            encoding="utf-8",
        )
        ready = ReadyResult(
            intent=ValidatedIntent(
                source="ecom_products",
                filters={},
                objective="execute",
                analysis_depth="unset",
                metrics=["count"],
                effective_question="how many products are in e-commerce data?",
            )
        )
        platform = SimpleNamespace(
            settings=SimpleNamespace(generated=generated, answer_mode="llm"),
            kg_registry=None,
        )
        state: dict = {"last_graph_id": "plmex_products", "last_result_id": "prior"}
        with (
            patch(
                "fashion_brain.orchestration.catalog_pipeline.ConversationContextAgent.analyze",
                return_value=ContextDecision(decision="RETRIEVE"),
            ),
            patch(
                "fashion_brain.orchestration.catalog_pipeline.QueryUnderstandingAgent.evaluate",
                return_value=ready,
            ),
            patch(
                "fashion_brain.orchestration.catalog_pipeline.IntentionAgent.understand"
            ) as understand,
            patch(
                "fashion_brain.orchestration.catalog_pipeline.ThinkingBrain.read"
            ) as thinking,
        ):
            retrieved, routes, intention, _ctx, choice = draft_catalog_orchestrated(
                platform,
                "how many products are in e-commerce data?",
                Principal(subject="t", clearance="internal"),
                history=[{"role": "user", "content": "How many products are in the PLM data?"}],
                conversation_state=state,
            )
        assert "18,183" in (retrieved.text or "")
        assert "kg_stats" in routes
        assert routes[-1] == "ecom_products"
        assert any(getattr(e, "record_id", None) == "kg_stats" for e in retrieved.evidence)
        assert state.get("last_graph_id") == "ecom_products"
        assert choice is None
        assert state.get("understanding", {}).get("validated_intent", {}).get("source") == "ecom_products"
        understand.assert_not_called()
        thinking.assert_not_called()


class TestAgenticLlmQueryUnderstanding:
    def test_long_story_count_via_llm(self):
        story = (
            "I've been reviewing our retail channels and the team keeps asking about "
            "the online catalog size. After looking at warehouse notes and some old "
            "spreadsheets, what I really need is a clear sense of how large the "
            "e-commerce product catalog is right now."
        )
        agent = QueryUnderstandingAgent()
        fake = MagicMock()
        fake.complete.return_value = json.dumps(
            {
                "status": "ready",
                "source": "ecom_products",
                "entity": None,
                "filters": {},
                "objective": "product_total",
                "metrics": ["count"],
                "group_by": None,
                "analysis_depth": "unset",
                "comparison": [],
                "effective_question": "How many products are in e-commerce?",
                "summary": "User wants e-commerce product total after a long preamble.",
                "clarification_slot": None,
                "clarification_question": None,
                "offered_options": [],
            }
        )
        with patch(
            "fashion_brain.agents.query_understanding.build_model_provider",
            return_value=fake,
        ):
            result = agent.evaluate(
                _platform(answer_mode="llm"),
                story,
                conversation_state={},
            )
        assert isinstance(result, ReadyResult)
        assert result.intent.source == "ecom_products"
        assert "count" in result.intent.metrics
        fake.complete.assert_called_once()

    def test_long_ambiguous_story_clarifies_via_llm(self):
        story = (
            "Our buying team has been discussing performance across several labels "
            "this season. We are not sure where to start. Please analyze our brand "
            "situation and help us decide what matters most in the catalog."
        )
        agent = QueryUnderstandingAgent()
        fake = MagicMock()
        fake.complete.return_value = json.dumps(
            {
                "status": "clarify",
                "source": None,
                "entity": None,
                "filters": {},
                "objective": "analysis",
                "metrics": [],
                "group_by": None,
                "analysis_depth": "unset",
                "comparison": [],
                "effective_question": story,
                "summary": "Brand/entity missing for analysis.",
                "clarification_slot": "entity",
                "clarification_question": "Which brand or buyer should I analyze?",
                "offered_options": ["category", "color"],
            }
        )
        with patch(
            "fashion_brain.agents.query_understanding.build_model_provider",
            return_value=fake,
        ):
            result = agent.evaluate(
                _platform(answer_mode="llm"),
                story,
                conversation_state={},
            )
        assert isinstance(result, ClarifyResult)
        assert result.slot == "entity"
        assert "brand" in result.question.casefold()

    def test_freeform_clarification_continue_via_llm(self):
        clar = ClarificationState(
            pending_slot="scope_or_direction",
            confirmed={"entity": "primark", "filters": {"brand": "primark"}, "analysis_depth": "unset"},
            offered_options=["category", "color", "gender"],
            turn_count=1,
            original_question="Analyze PRIMARK",
        )
        agent = QueryUnderstandingAgent()
        fake = MagicMock()
        fake.complete.return_value = json.dumps(
            {
                "status": "ready",
                "source": "ecom_products",
                "entity": "primark",
                "filters": {"brand": "primark"},
                "objective": "focused:color",
                "metrics": ["count"],
                "group_by": "color",
                "analysis_depth": "focused",
                "comparison": [],
                "effective_question": "PRIMARK color distribution",
                "summary": "User wants color focus in a free-form reply.",
                "clarification_slot": None,
                "clarification_question": None,
                "offered_options": ["color"],
            }
        )
        with (
            patch(
                "fashion_brain.agents.query_understanding.build_model_provider",
                return_value=fake,
            ),
            patch.object(agent, "_available_dimensions", return_value=_dims()),
        ):
            result = agent.evaluate(
                _platform(answer_mode="llm"),
                "I'd rather look at how colors break down across the range.",
                conversation_state={"clarification": clar.to_dict()},
            )
        assert isinstance(result, ReadyResult)
        assert result.intent.group_by == "color"
        assert result.intent.analysis_depth == "focused"
        fake.complete.assert_called_once()

    def test_model_unavailable_falls_back_to_heuristic(self):
        agent = QueryUnderstandingAgent()
        with (
            patch(
                "fashion_brain.agents.query_understanding.build_model_provider",
                side_effect=RuntimeError("down"),
            ),
            patch.object(agent, "_available_dimensions", return_value=_dims()),
            patch.object(agent, "_category_values", return_value=["Sweater"]),
            patch(
                "fashion_brain.agents.qu_fast.resolve_entity",
                return_value=("primark", []),
            ),
        ):
            result = agent.evaluate(
                _platform(answer_mode="llm"),
                "Analyze PRIMARK",
                conversation_state={},
            )
        # Heuristic + exact live brand → auto-run full analysis.
        assert isinstance(result, ReadyResult)
        assert str(result.intent.filters.get("brand") or "").casefold() == "primark"
        assert result.intent.analysis_depth == "full"

    def test_store_understanding_blob(self):
        state: dict = {}
        validated = ValidatedIntent(
            source="plmex_products",
            filters={},
            metrics=["count"],
            objective="execute",
            effective_question="How many products are in the PLM data?",
        )
        store_understanding(state, validated=validated, summary="PLM product total")
        blob = state["understanding"]
        assert blob["summary"] == "PLM product total"
        assert blob["validated_intent"]["source"] == "plmex_products"
        assert blob["pending_slot"] is None
        assert blob["updated_at"]

    def test_orchestrator_persists_understanding_on_clarification(self):
        from fashion_brain.orchestration.catalog_pipeline import draft_catalog_orchestrated
        from fashion_brain.agents.contracts import ContextDecision
        from knowledge_graph.access import Principal

        clarify = ClarifyResult(
            question="Which brand should I analyze?",
            state=ClarificationState(
                pending_slot="entity",
                confirmed={"understanding_summary": "Need a brand"},
                offered_options=[],
                turn_count=1,
                original_question="Analyze our brand situation please",
            ),
            slot="entity",
        )
        state: dict = {}
        with (
            patch(
                "fashion_brain.orchestration.catalog_pipeline.ConversationContextAgent.analyze",
                return_value=ContextDecision(decision="RETRIEVE"),
            ),
            patch(
                "fashion_brain.orchestration.catalog_pipeline.QueryUnderstandingAgent.evaluate",
                return_value=clarify,
            ),
        ):
            retrieved, routes, *_rest = draft_catalog_orchestrated(
                _platform(),
                "Analyze our brand situation please",
                Principal(subject="t", clearance="internal"),
                conversation_state=state,
            )
        assert clarification_awaiting(retrieved)
        assert state.get("understanding", {}).get("pending_slot") == "entity"
        assert "clarification" in routes
