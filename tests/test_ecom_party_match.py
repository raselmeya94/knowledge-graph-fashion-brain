from knowledge_graph.retrieval.ecom import _party_match_token, _display_label


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
