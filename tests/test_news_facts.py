"""News articles keep trade figures on allowlisted edges."""

from knowledge_graph.construction.news_facts import extract_trade_facts, reject_news_relationship, segment_articles
from knowledge_graph.construction.tools.metadata import ExtractMetadataTool
from knowledge_graph.ontology import load_ontology
from knowledge_graph.config import load_settings

NEWS = """Bangladesh retains EU’s top denim supplier position despite export decline
Bangladesh plans to make creative industries a mainstream economic sector
Bangladesh approves six LNG cargoes from BP Singapore and QatarEnergy
Bangladesh remained the European Union’s largest denim supplier in January-August 
2026, earning €869.13 million and capturing 29.40% of the bloc’s total denim imports, 
according to Eurostat data. Pakistan ranked second with €525.49 million, followed by 
Turkey with €438.25 million. However, Bangladesh’s denim exports to the EU fell 5.85% 
year-on-year, reducing its market share from 29.85% a year earlier. August exports 
rebounded 12.71% to €134.08 million, while total EU denim imports increased 11.23% to 
€430.16 million. Despite the overall decline, Bangladesh maintained its leading position, 
supported by competitive prices and a diverse product range.
Bangladesh aims to establish creative industries as a mainstream economic sector to 
support GDP growth, create jobs and strengthen the country’s global soft power, Finance 
Minister Amir Khosru Mahmud Chowdhury said on October 24. Speaking at the unveiling 
of Creative Footprints, jointly organised by Standard Chartered Bank and Bengal 
Foundation, he said the government would support artisans, painters and cultural 
workers through financial assistance, skills development, design, branding and 
marketing.
Bangladesh has approved the import of six liquefied natural gas (LNG) cargoes from BP 
Singapore Pte Limited and QatarEnergy Trading LLC to meet domestic gas demand. 
The Cabinet Committee on Government Purchase approved five cargoes from BP 
Singapore at $0.17 per million British thermal units (MMBtu) below the Japan Korea 
Marker (JKM) benchmark. One additional cargo will be purchased from QatarEnergy 
under a long-term government-to-government agreement at $0.10 per MMBtu below the 
benchmark, with delivery scheduled for December 20, 2026.
"""


def test_newsletter_splits_and_keeps_only_garment_trade_facts():
    articles = segment_articles(NEWS)
    assert [article["domain"] for article in articles] == ["garment", "out_of_domain", "out_of_domain"]
    garment = articles[0]["text"]
    facts = extract_trade_facts(articles[0]["title"], garment)
    ontology = load_ontology(load_settings().news_ontology_file)
    assert all(reject_news_relationship(item, ontology) is None for item in facts)
    by_key = {(item["type"], item["start_name"]): item for item in facts}

    bangladesh = by_key[("EXPORTS_TO", "Bangladesh")]
    assert bangladesh["end_name"] == "EU"
    assert bangladesh["properties"]["product"] == "denim"
    assert bangladesh["properties"]["value"] == "€869.13M"
    assert bangladesh["properties"]["share_pct"] == 29.4
    assert bangladesh["properties"]["period"] == "2026-01..08"
    assert bangladesh["properties"]["yoy_pct"] == -5.85
    assert by_key[("EXPORTS_TO", "Pakistan")]["properties"]["value"] == "€525.49M"
    assert by_key[("EXPORTS_TO", "Turkey")]["properties"]["value"] == "€438.25M"
    assert by_key[("RANKED_IN", "Bangladesh")]["properties"]["rank"] == 1
    assert by_key[("RANKED_IN", "Pakistan")]["properties"]["rank"] == 2
    assert by_key[("RANKED_IN", "Turkey")]["properties"]["rank"] == 3
    share = by_key[("HAS_MARKET_SHARE", "Bangladesh")]
    assert share["properties"]["share_pct"] == 29.4
    assert share["properties"]["previous_share"] == 29.85
    assert by_key[("CITES_DATA_FROM", articles[0]["title"])]["end_name"] == "Eurostat"
    assert all(item["type"] != "OCCURRED_IN" for item in facts)
    assert "January-August 2026" not in {item["end_name"] for item in facts}


def test_period_rejected_and_unlisted_triples_queued_for_ontology_review():
    from knowledge_graph.construction.news_facts import ontology_gap_warning

    ontology = load_ontology(load_settings().news_ontology_file)
    period = {
        "type": "OCCURRED_IN",
        "start_name": "Bangladesh",
        "start_type": "Country",
        "end_name": "January-August 2026",
        "end_type": "Period",
        "evidence": "January-August 2026",
        "properties": {},
    }
    approved = {
        "type": "APPROVED",
        "start_name": "Bangladesh",
        "start_type": "Country",
        "end_name": "BP Singapore",
        "end_type": "Company",
        "evidence": "Bangladesh has approved the import of 6 liquefied natural gas cargoes from BP Singapore",
        "properties": {"amount": "6"},
    }
    assert "not a news entity pair" in reject_news_relationship(period, ontology)
    assert reject_news_relationship(approved, ontology) is None
    assert "not allowlisted yet" in ontology_gap_warning(approved, ontology)


def test_written_dates_are_collected():
    metadata = ExtractMetadataTool().run({"text": NEWS, "title": "Digest", "kind": "pdf"}, None)
    assert "October 24" in metadata["dates"]
    assert "December 20, 2026" in metadata["dates"]
    assert "January-August 2026" in metadata["dates"]
