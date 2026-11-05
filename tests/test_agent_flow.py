from fashion_brain.agent import answer_question, plan_routes, resolve_catalog_plan
from fashion_brain.agents import Retrieved, requests_sample_design, response_messages, sample_design_sheet, sample_image_prompt
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
    assert "Intent: Calculate the product count." in summary.text
    assert "material = 100% cotton" in summary.text
    assert "blend is excluded" in summary.text
    assert summary.clarification_required is False


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
    assert "Jumper" in prompt
    assert "BROWN" in prompt
    assert "100%POLYESTER" in prompt
    assert "7GG" in prompt
    assert "Women" in prompt
    assert "No Machine" not in prompt


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
