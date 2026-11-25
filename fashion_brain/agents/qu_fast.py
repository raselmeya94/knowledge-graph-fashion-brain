"""fashion_brain/agents/qu_fast.py

Fast, deterministic helpers for QueryUnderstandingAgent:
  - classify_reply: yes / no / option-number / edit / other (no LLM call)
  - resolve_entity: validate a brand against LIVE brands (typo-fix, suggestions)
  - cached: TTL cache for metadata lookups
  - budget_exhausted: stop clarification loops
"""
from __future__ import annotations

import re
import time
from typing import Any, Callable

from knowledge_graph.ontology import normalize_name

MAX_CLARIFY_TURNS = 3
AUTO_RUN_NAMED_PARTY_ANALYSIS = True  # exact live-brand match -> run now, show scope in the answer
# Initially off: do not ask "reply yes to run the final analysis" — run in one shot.
SKIP_ANALYSIS_FINAL_CONFIRM = True

# Small CLOSED-class lexicon (yes/no words are not catalog vocabulary).
_YES = frozenset(
    "yes y yeah yep yup ok okay sure proceed ahead confirm confirmed correct right accurate "
    "haan han ji jii thik thikache "
    "হ্যাঁ হ্যা হাঁ হ্যাঁ। জি ঠিক ঠিকআছে ওকে আচ্ছা".split()
)
_NO = frozenset("no n nope nah cancel stop wrong incorrect না নাহ ভুল".split())
_FILLER = frozenset(
    "please it do go that this is looks look good fine run the report thanks thank you".split()
)


def _toks(text: str) -> list[str]:
    cleaned = re.sub(r"[.,!?;:।\"']", " ", (text or "").casefold())
    return [t for t in cleaned.split() if t]


def classify_reply(text: str, options: list[str] | None = None) -> tuple[str, str | None]:
    """Returns (kind, payload). kind in: empty | yes | no | option | edit | other."""
    toks = _toks(text)
    if not toks:
        return "empty", None
    if options and len(toks) == 1 and toks[0].isdigit():
        k = int(toks[0])
        if 1 <= k <= len(options):
            return "option", options[k - 1]
    s = set(toks)
    if s & _YES and s <= (_YES | _FILLER):
        return "yes", None
    if s & _NO and s <= (_NO | _FILLER):
        return "no", None
    if toks[0] in _YES or toks[0] in _NO:
        return "edit", None  # "yes, but only sweaters" / "no, Zara" -> needs real parsing
    return "other", None


# ------------------------------------------------------------------ entity validation
def resolve_entity(
    platform, text: str, graph_id: str | None = None
) -> tuple[str | None, list[str]]:
    """(validated_brand | None, suggestions). Never accepts free text as a brand."""
    from difflib import SequenceMatcher

    from fashion_brain.ask import (
        _is_grammar_party_name,
        live_brand_names,
        live_brands_mentioned,
    )

    names = cached(("brands", graph_id), lambda: live_brand_names(platform, graph_id), ttl=600)
    if not names:
        return None, []
    hits = live_brands_mentioned(text, names)
    if len(hits) == 1:
        return hits[0], []
    if len(hits) > 1:
        return None, hits[:3]
    best: list[tuple[float, str]] = []
    for tok in normalize_name(text or "").split():
        if len(tok) < 4:
            continue
        # Never fuzzy-map measure/grammar words (price→Prince, drop→…).
        if _is_grammar_party_name(tok):
            continue
        for n in names:
            best.append((SequenceMatcher(None, tok, normalize_name(n)).ratio(), n))
    best.sort(reverse=True)
    seen: list[str] = []
    for score, n in best:
        if score >= 0.78 and n not in seen:
            seen.append(n)
        if len(seen) == 3:
            break
    if seen and best[0][0] >= 0.9:
        return best[0][1], []  # clear typo: zarra -> Zara
    return None, seen


# ------------------------------------------------------------------ cache + budget
_CACHE: dict[Any, tuple[float, Any]] = {}


def cached(key: Any, fn: Callable[[], Any], ttl: float = 300.0) -> Any:
    now = time.time()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    value = fn()
    _CACHE[key] = (now, value)
    if len(_CACHE) > 256:
        _CACHE.clear()
    return value


def budget_exhausted(clar) -> bool:
    return int(getattr(clar, "turn_count", 0) or 0) >= MAX_CLARIFY_TURNS


def clear_cache() -> None:
    """Test helper."""
    _CACHE.clear()
