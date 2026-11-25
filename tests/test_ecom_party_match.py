from knowledge_graph.retrieval.ecom import _party_match_token, _display_label
from knowledge_graph.retrieval.kg_stats import collapse_value_bucket, resolve_facet_display


def test_c_and_a_party_aliases_match():
    assert _party_match_token("C&A") == "canda"
    assert _party_match_token("c-and-a") == "canda"
    assert _party_match_token("C and A") == "canda"
    assert _party_match_token("c & a") == "canda"


def test_hm_party_aliases_match():
    assert _party_match_token("H&M") == "handm"
    assert _party_match_token("h-and-m") == "handm"


def test_c_and_a_display_label():
    assert _display_label("C&A") == "C&A"
    assert _display_label("c-and-a") == "C&A"
    assert _display_label("canda") == "C&A"


def test_collapse_value_bucket_keeps_real_display():
    values, display = collapse_value_bucket(
        {
            "Jumper": 100,
            "jumper": 5,
            "Cut & Sew-Jersey": 50,
            "Jacket/Light Weight/Bomber": 20,
            "Hoodies & Sweatshirts": 10,
        }
    )
    assert values["jumper"] == 105
    assert display["jumper"] == "Jumper"
    assert display["cut sew jersey"] == "Cut & Sew-Jersey"
    assert display["jacket light weight bomber"] == "Jacket/Light Weight/Bomber"
    assert display["hoodies sweatshirts"] == "Hoodies & Sweatshirts"


def test_resolve_facet_display_uses_map_not_hardcoded_catalog():
    mapping = {
        "jumper": "Jumper",
        "cut sew jersey": "Cut & Sew-Jersey",
        "jacket light weight bomber": "Jacket/Light Weight/Bomber",
    }
    assert (
        resolve_facet_display("jumper", display_map=mapping) == "Jumper"
    )
    assert (
        resolve_facet_display("cut sew jersey", display_map=mapping)
        == "Cut & Sew-Jersey"
    )
    # Without a map, scrubbed keys are not invented from a giant canon.
    assert resolve_facet_display("jacket light weight bomber") is None
    # Light title-case fallback only (no slash invention without map).
    assert _display_label("top cardigan") == "Top Cardigan"
