from fashion_brain.agent import answer_question, plan_routes, resolve_catalog_plan
from fashion_brain.agents import (
    GraphRouterAgent,
    Intention,
    Retrieved,
    ResponseAgent,
    requests_sample_design,
    response_messages,
    sample_design_sheet,
    sample_image_prompt,
)
from fashion_brain.flow import iter_ask_events
from knowledge_graph.access import Principal
from knowledge_graph.errors import ModelUnavailableError
from knowledge_graph.models import GroundedAnswer


INTERNAL = Principal(subject="analyst", clearance="internal")


class _Provider:
    name = "openrouter"

    def __init__(self, route: str, web_error: bool = False):
        self.route = route
        self.web_error = web_error
        self.calls = []

    def chat(self, messages, *, timeout, tools=None, tool_choice=None):
        self.calls.append({"tools": tools, "tool_choice": tool_choice})
        if tool_choice == "required":
            arguments = "{}" if self.route != "search_web" else '{"query":"cotton prices"}'
            return {"choices": [{"message": {"tool_calls": [{"function": {"name": self.route, "arguments": arguments}}]}}]}
        if self.web_error:
            raise ModelUnavailableError("web search failed")
        return {
            "choices": [
                {
                    "message": {
                        "content": "Cotton prices moved this week.",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url_citation": {"url": "https://example.com/cotton", "title": "Cotton", "content": "Prices moved."},
                            }
                        ],
                    }
                }
            ]
        }

    def complete(self, prompt, *, timeout, system=None):
        self.calls.append({"complete": system, "prompt": prompt})
        if system and "own knowledge" in system:
            return "A seam joins two pieces of fabric."
        return prompt

    def iter_chat(self, messages, *, timeout, tools=None, tool_choice=None, sink=None):
        self.calls.append({"stream": True, "messages": messages, "tool_choice": tool_choice, "tools": tools})
        if tool_choice == "required":
            body = self.chat(messages, timeout=timeout, tools=tools, tool_choice=tool_choice)
            if sink is not None:
                sink["body"] = body
            yield "thinking", "Choosing."
            return
        text = "Bangladesh's garment industry is a major exporter."
        if sink is not None:
            sink["body"] = {"choices": [{"message": {"content": text}}]}
        yield "answer", text


def _llm(demo, monkeypatch, provider):
    monkeypatch.setattr(demo, "settings", demo.settings.model_copy(update={"answer_mode": "llm", "model_provider": "openrouter", "model_api_key": "test-key"}))
    monkeypatch.setattr("fashion_brain.flow.build_model_provider", lambda settings: provider)
    monkeypatch.setattr("fashion_brain.flow.build_image_provider", lambda settings: provider)
    monkeypatch.setattr("fashion_brain.ask.build_model_provider", lambda settings: provider)
    monkeypatch.setattr("fashion_brain.agent.build_model_provider", lambda settings: provider)
    monkeypatch.setattr("fashion_brain.agents.build_model_provider", lambda settings: provider)


def test_catalog_questions_can_use_every_stored_column():
    assert resolve_catalog_plan("how many products are there by category?")["group_by"] == "category"
    assert resolve_catalog_plan("breakdown by gauge")["group_by"] == "gauge"
    assert resolve_catalog_plan("how many products by knit machine type?")["group_by"] == "knit_machine_type"
    assert resolve_catalog_plan("how many products by plm main category?")["group_by"] == "plm_main_category"
    assert resolve_catalog_plan("how many products with gsm 180?")["filters"]["gsm"] == "180"
    assert resolve_catalog_plan("how many products with gauge 7?")["filters"]["gauge"] == "7"
    cotton = resolve_catalog_plan("how many products has 100% cotton?")
    assert cotton["filters"] == {"material": "100% cotton"}
    assert "garment" not in cotton["filters"]
    ranking = resolve_catalog_plan("show top 10 buyer who make 100 % cotton fabric garments")
    assert ranking["filters"] == {"material": "100% cotton"}
    assert ranking["group_by"] == "buyer"
    assert ranking["limit"] == 10
    assert "garment" not in ranking["filters"]
    plm_buyers = resolve_catalog_plan("which buyer are top product making in plm")
    assert plm_buyers["group_by"] == "buyer"
    assert plm_buyers["limit"] == 10
    assert plm_buyers["filters"] == {}
    plmex_buyers = resolve_catalog_plan("which buyer are top product making in plmex")
    assert plmex_buyers["group_by"] == "buyer"
    assert plmex_buyers["limit"] == 10
    assert plmex_buyers["filters"] == {}
    assert plan_routes("which buyer are top product making in plm") == ["product_count"]


def test_shared_material_route_selects_named_product_graphs(monkeypatch):
    rows = [
        {"id": "ecom_products", "name": "E-Commerce Knowledge Graph", "kind": "product", "status": "active"},
        {"id": "plmex_products", "name": "PLMEX Knowledge Graph", "kind": "product", "status": "active"},
        {"id": "news", "name": "News", "kind": "documents", "status": "active"},
    ]
    monkeypatch.setattr("fashion_brain.agents._impl._list_kg_registry", lambda _platform: rows)
    question = "Show materials shared across ecom and plmex."

    assert plan_routes(question) == ["shared_products"]
    choice = GraphRouterAgent().choose(
        Intention(intent="shared_products"),
        platform=object(),
        question=question,
    )
    assert choice.selected_graphs == ["ecom_products", "plmex_products"]
    assert choice.use_global_kg is False


def test_shared_material_question_bypasses_semantic_router(demo, monkeypatch):
    provider = _Provider("answer_directly")
    _llm(demo, monkeypatch, provider)
    retrieved = GroundedAnswer(
        question="Show materials shared across ecom and plmex.",
        routes=["shared_products"],
        answer="Cotton is shared.",
    )
    monkeypatch.setattr("fashion_brain.agent.answer_from_catalog", lambda *_args, **_kwargs: retrieved)

    answer = answer_question(
        demo,
        "Show materials shared across ecom and plmex.",
        INTERNAL,
    )

    assert answer is retrieved
    assert provider.calls == []


def test_shared_material_intent_is_not_misread_as_product_count():
    from fashion_brain.agents import ThinkingBrain

    summary = ThinkingBrain().read(
        None,
        "Show materials shared across ecom and plmex.",
        None,
    )

    assert "material values" in summary.actual_intent.casefold()
    assert "product count" not in summary.actual_intent.casefold()
    assert "No product filters were named." not in summary.text
    assert "material values present in every selected catalog" in summary.text.casefold()
    assert "product count" in summary.text.casefold()


def test_shared_catalog_reasoning_does_not_claim_nothing_matched():
    from fashion_brain.flow import _catalog_reasoning_text

    text = _catalog_reasoning_text(
        "Shared material values present in every selected product catalog:\n"
        "| Material | E-Commerce Products | PLMEX Products |\n"
        "| --- | ---: | ---: |\n"
        "| Polyester | 8 | 12 |",
        question="Show materials shared across ecom and plmex.",
        intention=Intention("shared_products", {}, "material", False, ""),
    )

    assert "Compared shared values" in text or "shared" in text.casefold()
    assert "shared by every selected catalog" in text.casefold()
    assert "nothing matched" not in text


def test_response_agent_polishes_fibre_percentages_and_preserves_catalog_counts(monkeypatch):
    from types import SimpleNamespace

    text = (
        "Shared material values present in every selected product catalog (product counts by catalog):\n"
        "| Material | E-Commerce Products | PLMEX Products |\n"
        "| --- | ---: | ---: |\n"
        "| polyester 22 | 3 | 975 |\n"
        "| polyester 15 | 15 | 815 |"
    )
    retrieved = Retrieved(text=text, evidence=[], caveats=[], unavailable=[])
    platform = SimpleNamespace(settings=SimpleNamespace(answer_mode="llm"))

    class Provider:
        def complete(self, prompt, *, timeout, system, temperature):
            assert "polyester 22" in prompt
            assert "different percentages as different compositions" in prompt
            assert temperature == 0.25
            return (
                "The shared materials include two distinct polyester compositions.\n\n"
                "| Material | E-Commerce Products | PLMEX Products |\n"
                "| --- | ---: | ---: |\n"
                "| Polyester 22% | 3 | 975 |\n"
                "| Polyester 15% | 15 | 815 |"
            )

    monkeypatch.setattr("fashion_brain.agents._impl.build_model_provider", lambda _settings: Provider())

    answer = ResponseAgent().compose(platform, "Show materials shared across catalogs", retrieved)

    assert "Polyester 22%" in answer
    assert "Polyester 15%" in answer
    assert "| 3 | 975 |" in answer
    assert "| 15 | 815 |" in answer
    assert not retrieved.unavailable


def test_response_agent_falls_back_if_it_changes_cross_catalog_counts(monkeypatch):
    from types import SimpleNamespace

    text = (
        "Shared material values present in every selected product catalog:\n"
        "| Material | E-Commerce Products | PLMEX Products |\n"
        "| --- | ---: | ---: |\n"
        "| polyester 22 | 3 | 975 |"
    )
    retrieved = Retrieved(text=text, evidence=[], caveats=[], unavailable=[])
    platform = SimpleNamespace(settings=SimpleNamespace(answer_mode="llm"))

    class Provider:
        def complete(self, _prompt, *, timeout, system, temperature):
            return (
                "| Material | E-Commerce Products | PLMEX Products |\n"
                "| --- | ---: | ---: |\n"
                "| Polyester 22% | 975 | 3 |"
            )

    monkeypatch.setattr("fashion_brain.agents._impl.build_model_provider", lambda _settings: Provider())

    assert ResponseAgent().compose(platform, "Show shared materials", retrieved) == text
    assert any("changed the cross-catalog comparison table" in note for note in retrieved.unavailable)


def test_cross_catalog_table_allows_percent_suffix_but_not_renamed_material():
    from fashion_brain.agents._impl import _cross_graph_facet_table_matches

    source = (
        "Shared material values present in every selected product catalog:\n"
        "| Material | E-Commerce Products | PLMEX Products |\n"
        "| --- | ---: | ---: |\n"
        "| polyester 22 | 3 | 975 |"
    )
    polished = (
        "| Material | E-Commerce Products | PLMEX Products |\n"
        "| --- | ---: | ---: |\n"
        "| Polyester 22% | 3 | 975 |"
    )
    renamed = (
        "| Material | E-Commerce Products | PLMEX Products |\n"
        "| --- | ---: | ---: |\n"
        "| Viscose 22% | 3 | 975 |"
    )

    assert _cross_graph_facet_table_matches(source, polished)
    assert not _cross_graph_facet_table_matches(source, renamed)


def test_buyer_product_analysis_is_not_a_specification():
    plan = resolve_catalog_plan("analysis PRIMARK buyer past products")
    assert plan["analysis"] is True
    assert plan["suggest"] is False
    assert plan["filters"] == {"buyer": "primark"}
    assert plan["group_by"] is None
    assert plan["limit"] == 5
    assert plan_routes("analysis PRIMARK buyer past products") == ["product_count"]


def test_primark_product_types_are_a_buyer_ranking():
    plan = resolve_catalog_plan("which type of products normally making PRIMARK")
    assert plan["filters"] == {"buyer": "primark"}
    assert plan["group_by"] == "garment_type"
    assert plan["limit"] == 10
    assert "garment" not in plan["filters"]
    history = [
        {"role": "user", "content": "Show top cotton buyers"},
        {"role": "assistant", "content": "Primark leads the cotton styles in this catalog."},
    ]
    continued = resolve_catalog_plan("which type of products normally making PRIMARK", history)
    assert continued["filters"] == {"buyer": "primark"}
    assert continued["group_by"] == "garment_type"
    assert "material" not in continued["filters"]
    named = resolve_catalog_plan("which type of products does primark make")
    assert named["filters"] == {"buyer": "primark"}
    assert named["group_by"] == "garment_type"


def test_specification_question_uses_the_buyer_and_winter_not_a_garment_phrase():
    question = "if i want a product for PPRIMARK for upcomming winter seasion, then are you suggest me a product specifications?"
    plan = resolve_catalog_plan(question)
    assert plan["suggest"] is True
    assert plan["filters"].get("buyer") == "pprimark"
    assert plan["season_name"] == "winter"
    assert plan["upcoming"] is True
    assert "garment" not in plan["filters"]
    sample = resolve_catalog_plan("make a new sample for primark for winter")
    assert sample["suggest"] is True
    assert sample["filters"]["buyer"] == "primark"
    assert sample["season_name"] == "winter"


def test_full_product_specification_requests_existing_catalog_records(demo):
    from fashion_brain.agents._impl import _deterministic_intent
    from fashion_brain.agents import IntentionAgent

    question = "Give me the full specification for a Primark sweater."
    assert plan_routes(question) == ["product_details"]
    plan = resolve_catalog_plan(question)
    assert plan["suggest"] is False
    assert plan_routes("suggest a product specification for Primark winter") == ["specification"]
    assert plan_routes("make a product specification for upcoming season") == ["specification"]
    summary = _deterministic_intent(question, None)
    assert "Retrieve relevant product details." in summary.text
    intention = IntentionAgent().understand(demo, question)
    assert intention.intent == "product_details"
    assert intention.suggest is False


def test_specification_follow_up_keeps_buyer_from_analysis_history():
    history = [
        {"role": "user", "content": "analysis PRIMARK buyer products"},
        {"role": "assistant", "content": "PRIMARK has 17860 products."},
    ]
    # Past filters apply only when the agent sets carry=True (references_prior_chat).
    plan = resolve_catalog_plan("make a product specification for upcoming season", history, carry=True)
    assert plan["suggest"] is True
    assert plan["upcoming"] is True
    assert plan["filters"] == {"buyer": "primark"}
    assert plan_routes("make a product specification for upcoming season") == ["specification"]
    fresh = resolve_catalog_plan("make a product specification for upcoming season", history, carry=False)
    assert fresh["filters"] == {}


def test_specification_question_bypasses_llm_router(demo, monkeypatch):
    """Specs must hit PLM data even when the model would prefer answer_directly."""

    provider = _Provider("answer_directly")
    _llm(demo, monkeypatch, provider)
    history = [
        {"role": "user", "content": "analysis PRIMARK buyer products"},
        {"role": "assistant", "content": "PRIMARK has many jumpers."},
    ]
    events = list(iter_ask_events(demo, "make a product specification for upcoming season", INTERNAL, history))
    answer = events[-1]["answer"]
    assert answer["routes"][0] == "catalog"
    assert "specification" in answer["routes"] or "Product specification" in answer["answer"]
    assert "unable" not in answer["answer"].lower()
    assert not any(call.get("tool_choice") == "required" for call in provider.calls)


def test_misspelled_buyer_is_corrected_and_winter_specs_come_from_styles():
    from types import SimpleNamespace

    from fashion_brain.agent import _append_catalog_answer, apply_buyer_spelling
    from knowledge_graph.models import RecordEnvelope
    from knowledge_graph.retrieval.products import FileProductRepository

    def envelope(record_id, buyer, style, garment):
        return RecordEnvelope(
            source_id="own",
            source_type="own_products",
            record_id=record_id,
            title=garment,
            content=garment,
            content_hash="0" * 64,
            synthetic=True,
            classification="internal",
            source_path="sample.csv",
            ingested_at="2026-01-01T00:00:00Z",
            lifecycle="active",
            metadata={"product": {"buyer": buyer, "garment_type": garment, "category": "Knitwear", "style_reference": style, "material": "100% Cotton", "gauge": "12GG", "gender": "Women"}},
        )

    repository = FileProductRepository(
        [
            envelope("aw", "PRIMARK", "AW26-PMK-BET", "Sweater"),
            envelope("aw2", "PRIMARK", "AW26-PMK-BET-2", "Sweater"),
            envelope("older", "PRIMARK", "AW24-PMK-BET", "Polo"),
            envelope("junk", "PRIMARK", "AW80-ODD", "Vest"),
            envelope("ss", "PRIMARK", "SS27-PMK-BET", "Tee"),
            envelope("other", "OTHER", "AW26-OTHER", "Jacket"),
        ]
    )
    holder = SimpleNamespace(filters={"buyer": "pprimark"}, corrections=[])
    note = apply_buyer_spelling(SimpleNamespace(products=repository), holder)
    assert holder.filters["buyer"] == "PRIMARK"
    assert "PRIMARK" in note
    lines = []
    evidence = []
    caveats = []
    unavailable = []
    _append_catalog_answer(
        SimpleNamespace(products=repository),
        INTERNAL,
        {
            "filters": {"buyer": "PRIMARK"},
            "group_by": None,
            "unsupported": False,
            "suggest": True,
            "season_name": "winter",
            "upcoming": True,
            "corrections": [note],
        },
        lines,
        evidence,
        caveats,
        unavailable,
    )
    text = "\n".join(lines)
    assert "Step 1. Buyer PRIMARK: 5 products." in text
    assert "Step 2. Winter styles" in text
    assert "3 products" in text
    assert "AW26 2" in text
    assert "AW80" not in text
    assert "Garment type: Sweater (2 of 3)" in text
    assert "Material: 100% Cotton (2 of 3)" in text or "Material: 100% Cotton (3 of 3)" in text
    assert "Tee" not in text
    assert "Vest" not in text
    assert "Jacket" not in text


def test_source_columns_can_be_filtered_and_grouped():
    from knowledge_graph.models import RecordEnvelope
    from knowledge_graph.retrieval.products import FileProductRepository

    envelope = RecordEnvelope(
        source_id="own",
        source_type="own_products",
        record_id="p1",
        title="Sample",
        content="sample",
        content_hash="0" * 64,
        synthetic=True,
        classification="internal",
        source_path="sample.csv",
        ingested_at="2026-01-01T00:00:00Z",
        lifecycle="active",
        metadata={"product": {"category": "Sweater", "gauge": "12", "source_columns": {"GSM": "180", "Knit Machine Type": "Flat"}}},
    )
    repository = FileProductRepository([envelope])
    assert repository.count(INTERNAL, gsm="180") == 1
    assert repository.count(INTERNAL, gsm="200") == 0
    assert repository.count_by(INTERNAL, "knit_machine_type") == [("Flat", 1)]
    assert repository.count_by(INTERNAL, "gauge") == [("12", 1)]
    cotton = RecordEnvelope(
        source_id="own",
        source_type="own_products",
        record_id="cotton",
        title="Cotton sweater",
        content="cotton",
        content_hash="1" * 64,
        synthetic=True,
        classification="internal",
        source_path="sample.csv",
        ingested_at="2026-01-01T00:00:00Z",
        lifecycle="active",
        metadata={"product": {"material": "100%Cotton", "source_columns": {"Composition": "100%Cotton"}}},
    )
    blend = RecordEnvelope(
        source_id="own",
        source_type="own_products",
        record_id="blend",
        title="Blend sweater",
        content="blend",
        content_hash="2" * 64,
        synthetic=True,
        classification="internal",
        source_path="sample.csv",
        ingested_at="2026-01-01T00:00:00Z",
        lifecycle="active",
        metadata={"product": {"material": "80%Cotton 20%Polyester", "source_columns": {"Composition": "80%Cotton 20%Polyester"}}},
    )
    materials = FileProductRepository([cotton, blend])
    assert materials.count(INTERNAL, material="100% cotton") == 1
    assert materials.count(INTERNAL, material="cotton") == 2


def test_thinking_brain_states_the_cotton_count():
    from fashion_brain.agents import ThinkingBrain

    summary = ThinkingBrain().read(None, "how many products has 100% cotton?", None)
    assert summary.filters["material"] == "100% cotton"
    assert "I understand this as calculate the product count." in summary.text
    assert "material = 100% cotton" in summary.text
    assert "I'll exclude blends" in summary.text
    assert summary.clarification_required is False


def test_thinking_brain_treats_analyze_by_brand_as_a_grouped_breakdown():
    from fashion_brain.agents import ThinkingBrain

    summary = ThinkingBrain().read(None, "Analyze product distribution by brand.", None)

    assert summary.group_by == "brand"
    assert summary.actual_intent == "Count products grouped by brand"
    assert "I understand this as count products grouped by brand." in summary.text
    assert "Count matching products within each brand group." in summary.text


def test_thinking_brain_uses_live_attributes_and_values_for_standalone_questions(demo, monkeypatch):
    from fashion_brain.agents import IntentionAgent, ThinkingBrain
    from fashion_brain.planning.kg_catalog import FieldCatalog, GraphFields, ValueHit

    catalog = FieldCatalog(
        graphs={
            "own_products": GraphFields(
                graph_id="own_products",
                attributes=["category", "garment_type", "color"],
                analysis_fields=["category", "garment_type", "color"],
            )
        },
        hits=[
            ValueHit(
                token="sweater",
                attribute="category",
                graph_id="own_products",
                matched_value="Sweater",
            ),
            ValueHit(
                token="sweater",
                attribute="garment_type",
                graph_id="own_products",
                matched_value="Sweater",
            ),
        ],
    )
    calls = []

    class _ThinkingProvider:
        def complete(self, prompt, *, timeout, system=None):
            calls.append({"prompt": prompt, "system": system})
            return (
                '{"actual_intent":"Count products of the requested type",'
                '"filters_and_conditions":{"garment":"sweater"},"group_by":null,'
                '"references_prior_chat":false,"clarification_required":false}'
            )

    monkeypatch.setattr(
        demo,
        "settings",
        demo.settings.model_copy(update={"answer_mode": "llm", "model_provider": "openrouter"}),
    )
    monkeypatch.setattr(
        "fashion_brain.agents._impl.build_model_provider",
        lambda settings: _ThinkingProvider(),
    )
    monkeypatch.setattr(
        "fashion_brain.planning.kg_catalog.build_field_catalog",
        lambda platform, question, *, probe_values=True: catalog,
    )

    summary = ThinkingBrain().read(demo, "how many sweater products are there?", None)

    assert calls == []
    assert "category, garment_type, color" not in summary.text
    assert summary.filters["garment"] == "sweater"
    assert summary.resolved_terms == [
        "sweater matches category = Sweater; garment type = Sweater"
    ]
    assert (
        "The live data supports these field matches: "
        "sweater matches category = Sweater; garment type = Sweater."
    ) in summary.text
    intention = IntentionAgent().understand(
        demo,
        "how many sweater products are there?",
        None,
        summary,
    )
    assert intention.filters["garment"] == "sweater"


def test_thinking_brain_ignores_incidental_values_when_party_is_explicit():
    from fashion_brain.agents._impl import _resolved_question_terms, _thinking_system
    from fashion_brain.planning.kg_catalog import FieldCatalog, GraphFields, ValueHit

    question = "Show product counts by category for Primark"
    catalog = FieldCatalog(
        graphs={
            "ecom_products": GraphFields(
                graph_id="ecom_products",
                attributes=["brand", "category", "material"],
            )
        },
        hits=[
            ValueHit("Primark", "brand", "ecom_products", "Primark"),
            ValueHit(
                "Primark",
                "material",
                "ecom_products",
                "and cotton from the Primark cotton project",
            ),
        ],
    )

    assert _resolved_question_terms(question, catalog) == [
        "Primark matches brand = Primark"
    ]
    system = _thinking_system(catalog, question)
    assert 'matches brand="Primark"' in system
    assert "cotton from the Primark cotton project" not in system


def test_thinking_brain_prefers_product_type_matches_over_incidental_values():
    from fashion_brain.agents._impl import _resolved_question_terms
    from fashion_brain.planning.kg_catalog import FieldCatalog, GraphFields, ValueHit

    question = "Give me the full specification for a Primark sweater."
    catalog = FieldCatalog(
        graphs={
            "ecom_products": GraphFields(
                graph_id="ecom_products",
                attributes=["brand", "category", "material"],
            )
        },
        hits=[
            ValueHit("full", "category", "ecom_products", "Full Suits"),
            ValueHit("Primark", "brand", "ecom_products", "Primark"),
            ValueHit("Primark", "material", "ecom_products", "Primark Cotton Project"),
            ValueHit("sweater", "category", "ecom_products", "Cardigans & Sweaters"),
            ValueHit("sweater", "material", "ecom_products", "Sweater Vest Outer Material"),
        ],
    )
    assert _resolved_question_terms(question, catalog) == [
        "Primark matches brand = Primark",
        "sweater matches category = Cardigans & Sweaters",
    ]


def test_thinking_brain_remaps_duplicate_buyer_to_brand_for_brand_graph(demo, monkeypatch):
    import json
    from types import SimpleNamespace

    from fashion_brain.agents import ThinkingBrain
    from fashion_brain.planning.kg_catalog import FieldCatalog, GraphFields, ValueHit

    catalog = FieldCatalog(
        graphs={
            "ecom_products": GraphFields(
                graph_id="ecom_products",
                attributes=["brand", "category"],
            )
        },
        hits=[ValueHit("Primark", "brand", "ecom_products", "Primark")],
    )
    monkeypatch.setattr(
        "fashion_brain.planning.kg_catalog.build_field_catalog",
        lambda *_args, **_kwargs: catalog,
    )
    monkeypatch.setattr("fashion_brain.ask.live_brand_names", lambda _platform: [])
    monkeypatch.setattr(
        "fashion_brain.ask.apply_buyer_spelling",
        lambda _platform, _summary: "",
    )
    monkeypatch.setattr(
        demo,
        "settings",
        demo.settings.model_copy(
            update={"answer_mode": "llm", "model_provider": "openrouter"}
        ),
    )

    class _ThinkingProvider:
        def complete(self, _prompt, *, timeout, system=None):
            return json.dumps(
                {
                    "actual_intent": "Count Primark products by category",
                    "filters_and_conditions": {
                        "brand": "Primark",
                        "buyer": "Primark",
                    },
                    "group_by": "category",
                    "references_prior_chat": False,
                    "clarification_required": False,
                }
            )

    monkeypatch.setattr(
        "fashion_brain.agents._impl.build_model_provider",
        lambda _settings: _ThinkingProvider(),
    )
    monkeypatch.setattr(
        "fashion_brain.planning.runtime_schema.get_runtime_schema",
        lambda *_args: SimpleNamespace(
            filter_keys=["brand"],
            facet_types=["Brand"],
            brand_label="Brand",
        ),
    )

    summary = ThinkingBrain().read(
        demo,
        "Show product counts by category for Primark",
        None,
    )

    assert summary.filters == {"brand": "primark"}
    assert summary.group_by == "category"
    assert "buyer =" not in summary.text


def test_reasoning_explains_interpretation_and_result_without_mechanical_headers():
    from types import SimpleNamespace

    from fashion_brain.agents.contracts import Intention
    from fashion_brain.flow import _catalog_reasoning_text, _provisional_reasoning_plan

    summary = SimpleNamespace(
        actual_intent="Count products grouped by category",
        filters={"brand": "Primark"},
        group_by="category",
        resolved_terms=["Primark matches brand = Primark"],
    )
    provisional = _provisional_reasoning_plan(
        "Show product counts by category for Primark",
        summary,
    )
    assert "Looking up: count products grouped by category." in provisional
    assert "brand = Primark" in provisional
    assert "grouped by category" in provisional
    assert "Live match:" in provisional

    final = _catalog_reasoning_text(
        "There are 1,464 products grouped by category.\n\n"
        "| Category | Count |\n|---|---:|\n| Clothes | 1,464 |",
        question="Show product counts by category for Primark",
        intention=Intention(
            intent="product_count",
            filters={"brand": "Primark"},
            group_by="category",
        ),
    )
    assert "Checked: brand = Primark; grouped by category." in final
    assert "There are 1,464 products grouped by category." in final
    assert "Introduce the result briefly" in final or "breakdown table" in final
    assert "Fetch:" not in final
    assert "Response plan:" not in final
    assert "Answer approach:" not in final


def test_direct_reasoning_shows_summary_not_model_private_reasoning(demo, monkeypatch):
    from fashion_brain.flow import _iter_direct_stream

    class _Provider:
        def iter_chat(self, *_args, **_kwargs):
            yield "reasoning", "private chain of thought"
            yield "thinking", "private scratchpad"
            yield "answer", "A seam joins pieces of fabric."

    monkeypatch.setattr("fashion_brain.flow.build_model_provider", lambda _settings: _Provider())

    events = list(_iter_direct_stream(demo, "What is a seam?"))
    visible_text = " ".join(str(event.get("text", "")) for event in events)

    assert "Using the model's own knowledge." in visible_text
    assert "A seam joins pieces of fabric." in visible_text
    assert "private chain of thought" not in visible_text
    assert "private scratchpad" not in visible_text


def test_slow_thinking_emits_live_status_updates():
    import time

    from fashion_brain.flow import _run_with_progress

    def operation():
        time.sleep(0.06)
        return "intent-ready"

    def collect():
        result = yield from _run_with_progress(
            operation,
            progress=["Checking live fields…", "Grounding filters…"],
            interval=0.01,
        )
        yield {"type": "result", "text": result}

    events = list(collect())
    assert events[-1]["type"] == "result"
    statuses = [event for event in events[:-1] if event["type"] == "status"]
    assert len(statuses) == 1
    assert events[0]["text"] == "Checking live fields…"
    assert events[-1]["text"] == "intent-ready"


def test_model_stage_timeouts_are_bounded():
    from fashion_brain.agents.text2cypher import _llm_timeout
    from fashion_brain.ask import _plan_timeout
    from fashion_brain.flow import _model_timeout

    class _Settings:
        request_timeout_seconds = None
        ollama_timeout_seconds = 180

    class _Platform:
        settings = _Settings()

    assert _plan_timeout(None) == 10
    assert _llm_timeout(None) == 20
    assert _model_timeout(_Platform()) == 20


def test_catalog_fast_route_does_not_capture_general_questions():
    from fashion_brain.flow import _is_explicit_catalog_question

    assert _is_explicit_catalog_question(
        "Show product counts by category for Primark.",
        "product_count",
    )
    assert _is_explicit_catalog_question(
        "How many sweaters does Primark have?",
        "product_count",
    )
    assert not _is_explicit_catalog_question("What is GSM?", "product_count")


def test_plain_catalog_counts_skip_text2cypher_model(demo, monkeypatch):
    from fashion_brain.agents.contracts import Intention, Retrieved
    from fashion_brain.orchestration.catalog_pipeline import _retrieve_with_text2cypher

    calls = []

    def retrieve(*args, **kwargs):
        calls.append((args, kwargs))
        return Retrieved(text="There are 12 matching products.", evidence=[], caveats=[], unavailable=[])

    monkeypatch.setattr("fashion_brain.ask._retrieve", retrieve)
    monkeypatch.setattr(
        "fashion_brain.agents.text2cypher.Text2CypherAgent.retrieve",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Text2Cypher should be skipped")),
    )
    intention = Intention(
        intent="product_count",
        filters={"brand": "Primark"},
        group_by=None,
        unsupported=False,
        thinking="count Primark products",
    )

    result = _retrieve_with_text2cypher(
        demo,
        None,
        intention,
        "How many products does Primark have?",
        graph_id="ecom_products",
        graph_ids=None,
        result_id=None,
    )

    assert result.text == "There are 12 matching products."
    assert len(calls) == 1


def test_catalog_count_stream_skips_second_response_model(demo, monkeypatch):
    from fashion_brain.agents.contracts import Intention, IntentSummary, Retrieved
    from fashion_brain.flow import _iter_catalog_stream
    from knowledge_graph.access import Principal

    retrieved = Retrieved(
        text="There are 12 matching products.",
        evidence=[],
        caveats=[],
        unavailable=[],
    )
    intention = Intention(
        intent="product_count",
        filters={"brand": "Primark"},
        thinking="count Primark products",
    )
    monkeypatch.setattr(
        "fashion_brain.agent.draft_catalog",
        lambda *_args, **_kwargs: (retrieved, ["product_count"], intention),
    )
    monkeypatch.setattr(
        "fashion_brain.flow.build_model_provider",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Response model should be skipped")),
    )

    events = list(
        _iter_catalog_stream(
            demo,
            "How many products does Primark have?",
            Principal(subject="test", clearance="internal"),
            summary=IntentSummary(
                actual_intent="Calculate the product count",
                text="",
                filters={"brand": "Primark"},
            ),
        )
    )

    streamed_answer = "".join(
        str(event.get("text") or "") for event in events if event.get("type") == "answer"
    )
    assert streamed_answer == retrieved.text
    assert events[-1]["type"] == "done"


def test_composition_compare_uses_canonical_tables_and_single_chart():
    from fashion_brain.flow import _canonical_composition_compare_answer

    answer = _canonical_composition_compare_answer(
        "Composition compare — similar t-shirt across Primark vs Zara "
        "(Primark 400, Zara 46).\n"
        "Response must include EVERY brand section below.\n"
        "**Primark** — 400 matching t-shirt\n"
        "| Material / composition | Products | Share |\n"
        "| --- | ---: | ---: |\n"
        "| Elastane 5% | 89 | 22.2% |\n"
        "**Zara** — 46 matching t-shirt\n"
        "| Material / composition | Products | Share |\n"
        "| --- | ---: | ---: |\n"
        "| Cotton | 28 | 60.9% |\n"
        "```chart\n"
        '{"type":"bar","title":"Material mix","labels":["Primark: Elastane 5%","Zara: Cotton"],"values":[89,28]}\n'
        "```\n"
        "Compare takeaway: Primark leads with Elastane 5%; Zara leads with Cotton."
    )

    assert answer.startswith(
        "Primark has 400 matching t-shirt products; Zara has 46 matching t-shirt products."
    )
    assert answer.count("| Material / composition | Products | Share |") == 2
    assert answer.count("```chart") == 1
    assert "Response must include" not in answer
    assert "Compare takeaway:" not in answer


def test_catalog_follow_up_keeps_the_earlier_filters():
    history = [
        {"role": "user", "content": "How many products are in the PLM data?"},
        {"role": "assistant", "content": "The PLM data contains 124682 products."},
        {"role": "user", "content": "how many of them are sweater"},
        {"role": "assistant", "content": "The PLM data contains 78841 products with category or garment type sweater."},
    ]
    plan = resolve_catalog_plan("can you make a breakdown by gender?", history, carry=True)
    assert plan["filters"] == {"garment": "sweater"}
    assert plan["group_by"] == "gender"
    # Agent says standalone → ignore earlier sweater filter.
    fresh = resolve_catalog_plan("can you make a breakdown by gender?", history, carry=False)
    assert fresh["filters"] == {}
    assert fresh["group_by"] == "gender"


def test_show_sweater_red_uses_catalog_count_not_materials_listing(demo):
    """Show/list wording must not wipe garment/color filters into the Neo4j materials template."""

    from fashion_brain.agents import IntentionAgent, ThinkingBrain

    question = "show a sweater product with red color"
    assert plan_routes(question) == ["products"]
    summary = ThinkingBrain().read(demo, question, None)
    intention = IntentionAgent().understand(demo, question, None, summary)
    assert intention.intent == "product_count"
    assert intention.filters.get("garment") == "sweater"
    assert intention.filters.get("color") == "red"


def test_response_agent_receives_the_question_and_graph_context():
    retrieved = Retrieved(
        text="There are 6 products.",
        evidence=[],
        caveats=[],
        unavailable=[],
    )
    messages = response_messages("How many products are there?", retrieved)
    assert "briefing" in messages[0]["content"]
    assert "How many products are there?" in messages[1]["content"]
    assert "There are 6 products." in messages[1]["content"]
    assert "PLMEX" not in messages[0]["content"] or "Do not add any source footer" in messages[0]["content"]


def test_shared_material_response_prompt_requires_polished_grounded_output():
    retrieved = Retrieved(
        text=(
            "Materials shared by more than one product (same Material node in the product graph):\n"
            "- cotton 100 (material:cotton-100): 254 products — raw-product-1, raw-product-2.\n"
            "- Upper (material:upper): 413 products — raw-product-3, raw-product-4.\n"
            "…and 1,300 more shared materials."
        ),
        evidence=[],
        caveats=[],
        unavailable=[],
    )

    messages = response_messages("Which products share a material?", retrieved)
    prompt = messages[1]["content"]
    assert "ranked markdown table with Material and Products columns" in prompt
    assert "'Cotton 100%'" in prompt
    assert "Omit canonical IDs and sample product IDs" in prompt
    assert "Exclude component/garment part labels" in prompt
    assert "Mention brands only when the findings explicitly identify them" in prompt
    assert "never extrapolate a total" in prompt


def test_follow_up_includes_the_previous_reply(demo, monkeypatch):
    provider = _Provider("search_web")
    _llm(demo, monkeypatch, provider)
    history = [
        {"role": "user", "content": "Study Bangladeshi garments and write a summary."},
        {"role": "assistant", "content": "Bangladesh exports about 38 billion dollars of garments."},
    ]
    answer = answer_question(demo, "make it in a single sentence", INTERNAL, history)
    assert answer.routes == ["direct"]
    prompts = [call["prompt"] for call in provider.calls if "prompt" in call]
    assert "38 billion" in prompts[-1]
    assert "make it in a single sentence" in prompts[-1]
    assert not any(call.get("tools") and call["tools"][0].get("type") == "openrouter:web_search" for call in provider.calls)


def test_streamed_follow_up_uses_the_previous_reply(demo, monkeypatch):
    provider = _Provider("search_web")
    _llm(demo, monkeypatch, provider)
    history = [{"role": "assistant", "content": "Long garment report."}]
    events = list(iter_ask_events(demo, "make it in a single sentence", INTERNAL, history))
    streamed = [call for call in provider.calls if call.get("stream")]
    assert streamed
    assert streamed[0]["tool_choice"] is None
    contents = " ".join(message["content"] for message in streamed[0]["messages"])
    assert "Long garment report." in contents
    assert events[-1]["type"] == "done"
    assert events[-1]["answer"]["routes"] == ["direct"]
    assert "garment" in events[-1]["answer"]["answer"].lower()


def test_visualize_follow_up_transforms_previous_table_without_model_or_catalog(demo, monkeypatch):
    import json

    from fashion_brain.flow import continues_previous

    history = [
        {"role": "user", "content": "Which categories are seeing decreasing price trends?"},
        {
            "role": "assistant",
            "content": (
                "Categories by average same-currency price change:\n\n"
                "| Category | Avg change | Events | Products |\n"
                "| --- | ---: | ---: | ---: |\n"
                "| lingerie | -53.49% | 1 | 1 |\n"
                "| sale | -33.34% | 1 | 1 |\n"
                "| socks | -25.66% | 3 | 3 |\n"
            ),
        },
    ]
    monkeypatch.setattr(
        "fashion_brain.flow.build_model_provider",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Model should not be called")),
    )
    monkeypatch.setattr(
        "fashion_brain.agent.plan_routes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Router should not be called")),
    )

    question = "make it visualization"
    assert continues_previous(question, history)
    events = list(iter_ask_events(demo, question, INTERNAL, history))
    answer = events[-1]["answer"]
    assert answer["routes"] == ["direct"]
    assert "no new data was retrieved" in answer["caveats"][0]
    chart_text = "".join(
        event.get("text", "") for event in events if event.get("type") == "answer"
    )
    assert '"labels": ["lingerie", "sale", "socks"]' in chart_text
    assert '"values": [-53.49, -33.34, -25.66]' in chart_text
    assert len([event for event in events if event.get("type") == "done"]) == 1

    chart_history = history + [
        {"role": "assistant", "content": answer["answer"]},
    ]
    pie_events = list(iter_ask_events(demo, "with pie chart", INTERNAL, chart_history))
    pie_answer = pie_events[-1]["answer"]
    assert pie_answer["routes"] == ["direct"]
    pie_text = pie_answer["answer"]
    pie_data = json.loads(pie_text.split("```chart\n", 1)[1].split("\n```", 1)[0])
    assert pie_data["type"] == "pie"
    assert pie_data["title"].startswith("Share of absolute")
    assert pie_data["values"] == [-53.49, -33.34, -25.66]


SPEC_HISTORY = [
    {"role": "user", "content": "suggest a product specification for PRIMARK winter"},
    {
        "role": "assistant",
        "content": (
            "Product specification\n"
            "Buyer: PRIMARK\n"
            "Season: winter\n"
            "Garment type: Jumper (4 of 6)\n"
            "Category: Sweater\n"
            "Material: 100%POLYESTER\n"
            "Color: BROWN\n"
            "Gender: Women\n"
            "Design for: Top\n"
            "Gauge: 7GG\n"
            "GSM: 200.00\n"
            "Knit machine type: No Machine Selected\n"
            "Knitting time: 28\n"
            "Sample collection type: Development Sample\n"
        ),
    },
]


def test_sample_design_uses_only_the_previous_specification():
    sheet = sample_design_sheet(SPEC_HISTORY[1]["content"])
    assert "PRIMARK" in sheet
    assert "Jumper" in sheet
    assert "100%POLYESTER" in sheet
    assert "7GG" in sheet
    assert "BROWN" in sheet
    assert "(4 of 6)" not in sheet
    assert requests_sample_design("make this garment sample", SPEC_HISTORY)
    assert not requests_sample_design("make a new sample for primark for winter", SPEC_HISTORY)
    assert not requests_sample_design("make it in a single sentence", SPEC_HISTORY)


def test_pasted_specification_routes_to_sample_design_not_catalog_suggest():
    question = (
        "A Multicolour Women's Pullover Sweater designed for primark winter.\n\n"
        "Identity\n"
        "Buyer: primark\n"
        "Season: winter\n"
        "Garment type: Jumper\n"
        "Category: Sweater\n"
        "Material: Spandex\n"
        "Color: Multicolour\n"
        "Sample collection type: Development Sample\n\n"
        "Making notes\n"
        "Create a Development Sample for primark winter.\n\n"
        "make garment sample design"
    )
    assert requests_sample_design(question, None)
    assert "Jumper" in sample_design_sheet(question)
    assert "Spandex" in sample_image_prompt(question)


def test_make_this_garment_sample_uses_the_sample_design_agent(demo, monkeypatch):
    provider = _Provider("answer_directly")
    _llm(demo, monkeypatch, provider)
    events = list(iter_ask_events(demo, "make this garment sample", INTERNAL, SPEC_HISTORY))
    assert events[-1]["answer"]["routes"] == ["sample_design"]
    streamed = [call for call in provider.calls if call.get("stream")]
    assert streamed
    contents = " ".join(message["content"] for message in streamed[0]["messages"])
    assert "Sample Design" in contents
    assert "Jumper" in contents
    assert "7GG" in contents
    assert not any(call.get("tool_choice") == "required" for call in provider.calls)


def test_sample_image_prompt_uses_the_visual_specification_fields():
    prompt = sample_image_prompt(SPEC_HISTORY[1]["content"])
    assert "PRIMARK" in prompt
    assert "Jumper" in prompt
    assert "BROWN" in prompt
    assert "100%POLYESTER" in prompt
    assert "7GG" in prompt
    assert "Women" in prompt
    assert "woven label" in prompt.casefold()
    assert "No Machine" not in prompt


def test_sample_image_prompt_accepts_lowercase_neo4j_field_labels():
    prompt = sample_image_prompt(
        "Product specification\n"
        "buyer: PRIMARK\n"
        "garment type: Jumper\n"
        "material: 100%POLYESTER\n"
        "color: BROWN\n"
        "gender: Women\n"
        "gauge: 7GG\n"
    )
    assert "PRIMARK" in prompt
    assert "Jumper" in prompt
    assert "BROWN" in prompt
    assert "100%POLYESTER" in prompt
    assert "7GG" in prompt
    assert "woven label" in prompt.casefold()


def test_sample_images_use_openrouter_even_when_chat_is_ollama(demo, monkeypatch):
    from fashion_brain.llm import OPENROUTER_DEFAULT_IMAGE_MODEL, build_image_provider, resolve_image_model

    monkeypatch.setattr(
        demo,
        "settings",
        demo.settings.model_copy(
            update={
                "model_provider": "ollama",
                "model_api_key": "test-openrouter-key",
                "image_model": "google/gemini-3.1-flash-image",
            }
        ),
    )
    provider = build_image_provider(demo.settings)
    assert provider.name == "openrouter"
    assert resolve_image_model(demo.settings) == OPENROUTER_DEFAULT_IMAGE_MODEL


def test_sample_design_draws_an_image_from_the_specification(demo, monkeypatch):
    class _ImageProvider(_Provider):
        def generate_image(self, prompt, *, model, timeout):
            self.calls.append({"image_prompt": prompt, "model": model})
            return b"\x89PNG\r\n\x1a\n", "image/png"

    provider = _ImageProvider("answer_directly")
    _llm(demo, monkeypatch, provider)
    events = list(iter_ask_events(demo, "make this garment sample", INTERNAL, SPEC_HISTORY))
    answer = events[-1]["answer"]["answer"]
    assert "![Sample design](/samples/" in answer
    name = answer.split("/samples/", 1)[1].split(")", 1)[0]
    assert (demo.settings.generated / "samples" / name).read_bytes().startswith(b"\x89PNG")
    image_call = next(call for call in provider.calls if "image_prompt" in call)
    assert "Jumper" in image_call["image_prompt"]
    assert image_call["model"] == demo.settings.image_model


def test_sample_design_keeps_the_sheet_when_the_image_fails(demo, monkeypatch):
    class _ImageProvider(_Provider):
        def generate_image(self, prompt, *, model, timeout):
            raise ModelUnavailableError("OpenRouter returned HTTP 404.")

    _llm(demo, monkeypatch, _ImageProvider("answer_directly"))
    events = list(iter_ask_events(demo, "make this garment sample", INTERNAL, SPEC_HISTORY))
    answer = events[-1]["answer"]["answer"]
    assert "The sample image was not generated." in answer
    assert "/samples/" not in answer


def test_template_mode_stays_on_the_catalog(demo):
    answer = answer_question(demo, "How many synthetic demo products are in the catalog?", INTERNAL)
    assert answer.routes == ["product_count", "own_products"]
    assert "6" in answer.answer
    assert "clearance" not in answer.answer.lower()


def test_model_can_choose_the_catalog(demo, monkeypatch):
    provider = _Provider("catalog")
    _llm(demo, monkeypatch, provider)
    answer = answer_question(demo, "How many synthetic demo products are in the catalog?", INTERNAL)
    assert answer.routes[0] == "catalog"
    assert "product_count" in answer.routes
    assert "6" in answer.answer


def test_model_can_answer_directly(demo, monkeypatch):
    provider = _Provider("answer_directly")
    _llm(demo, monkeypatch, provider)
    answer = answer_question(demo, "What is a seam?", INTERNAL)
    assert answer.routes == ["direct"]
    assert answer.answer == "A seam joins two pieces of fabric."
    assert answer.evidence == []
    assert "product records" in answer.caveats[0]


def test_model_can_search_the_web(demo, monkeypatch):
    provider = _Provider("search_web")
    _llm(demo, monkeypatch, provider)
    answer = answer_question(demo, "What is the latest cotton price?", INTERNAL)
    assert answer.routes == ["web_search"]
    assert answer.answer == "Cotton prices moved this week."
    assert answer.evidence[0].url == "https://example.com/cotton"
    web_call = next(call for call in provider.calls if call.get("tools") and call.get("tool_choice") is None)
    assert web_call["tools"][0]["type"] == "openrouter:web_search"


def test_catalog_decline_continues_with_web_search(demo, monkeypatch):
    provider = _Provider("catalog")
    _llm(demo, monkeypatch, provider)
    question = "study about bangladeshi garments and make a summary report."
    declined = GroundedAnswer(question=question, routes=["shared"], answer="No count was estimated. Ask a catalog question.")
    def _declined(platform, question, principal, history=None, graph_id=None):
        return declined

    monkeypatch.setattr("fashion_brain.ask.answer_from_catalog", _declined)
    monkeypatch.setattr("fashion_brain.agent.answer_from_catalog", _declined)

    answer = answer_question(demo, question, INTERNAL)

    assert answer.routes == ["catalog", "web_search"]
    assert answer.answer == "Cotton prices moved this week."
    assert answer.unavailable[0] == "Our products do not cover this question."


def test_web_search_falls_back_to_the_model(demo, monkeypatch):
    provider = _Provider("search_web", web_error=True)
    _llm(demo, monkeypatch, provider)
    answer = answer_question(demo, "What is the latest cotton price?", INTERNAL)
    assert answer.routes == ["web_search", "direct"]
    assert answer.answer == "A seam joins two pieces of fabric."
    assert answer.unavailable
