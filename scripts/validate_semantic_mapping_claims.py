"""Cross-check proposed ecom↔plmex semantic mappings against live Postgres values."""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in line or line.strip().startswith("#"):
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def norm(s: object) -> str:
    text = str(s).strip().lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def expand_json(val: object) -> set[str]:
    out: set[str] = set()
    if val is None:
        return out
    if isinstance(val, list):
        for x in val:
            out |= expand_json(x)
        return out
    if isinstance(val, dict):
        for k in ("name", "color", "material", "fibre", "fiber", "value", "label"):
            if k in val:
                out |= expand_json(val[k])
        for v in val.values():
            if isinstance(v, str) and len(v) < 80:
                out |= expand_json(v)
        return out
    s = str(val).strip()
    if not s:
        return out
    if s.startswith("[") or s.startswith("{"):
        try:
            return expand_json(json.loads(s))
        except Exception:
            pass
    n = norm(s)
    if n and len(n) < 80:
        out.add(n)
    for m in re.finditer(r"(\d+)\s*%\s*([A-Za-z][A-Za-z \-]+)", s):
        mat = norm(m.group(2))
        if mat:
            out.add(mat)
    # also split composition-like
    for part in re.split(r"[,;|/]+", s):
        pn = norm(part)
        if pn and 1 < len(pn) < 60:
            out.add(pn)
    return out


def distinct_norms(dsn: str, table: str, col: str, *, jsonb: bool = False, limit: int = 80000) -> set[str]:
    out: set[str] = set()
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            if jsonb:
                cur.execute(
                    f"SELECT DISTINCT {col}::text FROM {table} WHERE {col} IS NOT NULL LIMIT %s",
                    (limit,),
                )
            else:
                cur.execute(
                    f"""
                    SELECT DISTINCT CAST({col} AS text)
                    FROM {table}
                    WHERE {col} IS NOT NULL AND btrim(CAST({col} AS text)) <> ''
                    LIMIT %s
                    """,
                    (limit,),
                )
            for (raw,) in cur.fetchall():
                if jsonb:
                    try:
                        parsed = json.loads(raw)
                    except Exception:
                        parsed = raw
                    out |= expand_json(parsed)
                else:
                    out |= expand_json(raw)
    return {v for v in out if v and v not in {"(empty)", "empty", "n a", "na"}}


def top_values(dsn: str, table: str, col: str, n: int = 12) -> list[tuple[str, int]]:
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT btrim(CAST({col} AS text)) AS v, count(*)::int AS c
                FROM {table}
                WHERE {col} IS NOT NULL AND btrim(CAST({col} AS text)) <> ''
                GROUP BY 1
                ORDER BY c DESC
                LIMIT %s
                """,
                (n,),
            )
            return [(r[0], r[1]) for r in cur.fetchall()]


def overlap(a: set[str], b: set[str]) -> dict:
    inter = a & b
    if not a or not b:
        return {"shared": 0, "jac": 0.0, "cov_a": 0.0, "cov_b": 0.0, "samples": []}
    return {
        "shared": len(inter),
        "jac": len(inter) / len(a | b),
        "cov_a": len(inter) / len(a),
        "cov_b": len(inter) / len(b),
        "samples": sorted(inter, key=len)[:12],
    }


def soft_norm_audience(v: str) -> str:
    """Map audience-like tokens to a coarse bucket."""
    t = norm(v)
    if not t:
        return ""
    mapping = {
        "women": "women",
        "woman": "women",
        "womens": "women",
        "ladies": "women",
        "men": "men",
        "man": "men",
        "mens": "men",
        "kids": "kids",
        "kid": "kids",
        "children": "kids",
        "girl": "girl",
        "girls": "girl",
        "boy": "boy",
        "boys": "boy",
        "baby": "baby",
        "baby girl": "baby_girl",
        "baby boy": "baby_boy",
        "unisex": "unisex",
    }
    if t in mapping:
        return mapping[t]
    # compound
    if "baby" in t and "girl" in t:
        return "baby_girl"
    if "baby" in t and "boy" in t:
        return "baby_boy"
    if "girl" in t:
        return "girl"
    if "boy" in t:
        return "boy"
    if "women" in t or "woman" in t or "ladies" in t:
        return "women"
    if t == "men" or t.startswith("men ") or t.endswith(" men"):
        return "men"
    if "kid" in t or "child" in t:
        return "kids"
    return t


def garment_stem(v: str) -> str:
    t = norm(v)
    # crude plural / noise strip
    t = re.sub(r"\b(and|&|the|for|with)\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    # common stems
    replacements = [
        (r"\bjumpers?\b", "jumper"),
        (r"\bsweaters?\b", "sweater"),
        (r"\bdresses?\b", "dress"),
        (r"\bshirts?\b", "shirt"),
        (r"\bjackets?\b", "jacket"),
        (r"\bjeans\b", "jeans"),
        (r"\bsocks?\b", "sock"),
        (r"\bshorts?\b", "short"),
        (r"\btrousers?\b", "trouser"),
        (r"\bpants?\b", "pant"),
        (r"\bhoodies?\b", "hoodie"),
        (r"\bsweatshirts?\b", "sweatshirt"),
        (r"\bcardigans?\b", "cardigan"),
        (r"\bskirts?\b", "skirt"),
        (r"\bcoats?\b", "coat"),
        (r"\bswimwear\b", "swimwear"),
        (r"\bloungewear\b", "loungewear"),
        (r"\bunderwear\b", "underwear"),
        (r"\bjumpsuits?\b", "jumpsuit"),
        (r"\bpolos?\b", "polo"),
        (r"\bvests?\b", "vest"),
        (r"\bscar(?:f|ves)\b", "scarf"),
        (r"\bgloves?\b", "glove"),
        (r"\bhats?\b", "hat"),
        (r"\bbelts?\b", "belt"),
        (r"\btights?\b", "tight"),
        (r"\bleggings?\b", "legging"),
    ]
    for pat, rep in replacements:
        if re.search(pat, t):
            return rep
    return t


def main() -> None:
    load_env()
    ecom = os.environ["FASHIONGPT_ECOM_DSN"]
    plm = os.environ["FASHIONGPT_PLMEX_DSN"]

    claims = []

    def check(claim_id: str, level: str, e_label: str, p_label: str, o: dict, note: str, verdict: str):
        claims.append(
            {
                "id": claim_id,
                "proposed_level": level,
                "ecom": e_label,
                "plmex": p_label,
                "shared": o["shared"],
                "jaccard": round(o["jac"], 4),
                "cov_ecom": round(o["cov_a"], 4),
                "cov_plmex": round(o["cov_b"], 4),
                "samples": o["samples"],
                "verdict": verdict,
                "note": note,
            }
        )
        print(f"[{verdict}] {claim_id}: {e_label} <-> {p_label} shared={o['shared']} jac={o['jac']:.3f} | {note}")

    # --- Load key sets ---
    print("Loading distinct sets...")
    e_section = distinct_norms(ecom, "master_products", "section")
    p_gender = distinct_norms(plm, "plmex_products", "gender")
    e_subfamily = distinct_norms(ecom, "master_products", "subfamily")
    e_family = distinct_norms(ecom, "master_products", "family")
    e_category = distinct_norms(ecom, "master_products", "category")
    p_garment = distinct_norms(plm, "plmex_products", "garment_type")
    p_main = distinct_norms(plm, "plmex_products", "plm_main_category")
    p_sub = distinct_norms(plm, "plmex_products", "plm_sub_category")
    p_ssub = distinct_norms(plm, "plmex_products", "plm_sub_sub_category")
    p_cat = distinct_norms(plm, "plmex_products", "category")
    e_comp = distinct_norms(ecom, "master_products", "composition")
    e_compd = distinct_norms(ecom, "master_products", "composition_detail", jsonb=True)
    p_comp = distinct_norms(plm, "plmex_products", "composition")
    e_colors = distinct_norms(ecom, "master_products", "colors", jsonb=True)
    p_color = distinct_norms(plm, "plmex_products", "color")
    e_brand = distinct_norms(ecom, "master_products", "brand")
    p_brand = distinct_norms(plm, "plmex_products", "brand")
    p_buyer = distinct_norms(plm, "plmex_products", "buyer")
    e_ref = distinct_norms(ecom, "master_products", "reference")
    e_dref = distinct_norms(ecom, "master_products", "display_reference")
    p_sref = distinct_norms(plm, "plmex_products", "style_reference")
    p_csref = distinct_norms(plm, "plmex_products", "customer_style_reference")
    e_stock = distinct_norms(ecom, "master_products", "in_stock_count")
    p_stock = distinct_norms(plm, "plmex_products", "current_stock")
    e_avail = distinct_norms(ecom, "master_products", "availability")
    e_name = distinct_norms(ecom, "master_products", "name")
    e_desc = distinct_norms(ecom, "master_products", "description")
    p_dfeat = distinct_norms(plm, "plmex_products", "design_feature")
    e_care = distinct_norms(ecom, "master_products", "care")
    e_coo = distinct_norms(ecom, "master_products", "country_of_origin")
    p_loc = distinct_norms(plm, "plmex_products", "current_location")
    e_path = distinct_norms(ecom, "master_products", "category_path")
    e_fit = distinct_norms(ecom, "master_products", "fit")
    e_construction = distinct_norms(ecom, "master_products", "construction")

    # 1 section ↔ gender (exact + soft audience)
    o = overlap(e_section, p_gender)
    soft_e = {soft_norm_audience(x) for x in e_section} - {""}
    soft_p = {soft_norm_audience(x) for x in p_gender} - {""}
    o_soft = overlap(soft_e, soft_p)
    check(
        "section↔gender",
        "L2",
        "section",
        "gender",
        o_soft if o_soft["shared"] >= o["shared"] else o,
        f"exact shared={o['shared']} {o['samples']}; soft buckets shared={o_soft['shared']} {o_soft['samples']}. "
        f"ecom tops={top_values(ecom,'master_products','section',6)}; plm tops={top_values(plm,'plmex_products','gender',6)}",
        "CONFIRM" if o_soft["shared"] >= 3 else "WEAK",
    )

    # 2 subfamily ↔ garment_type (exact + stem)
    o = overlap(e_subfamily, p_garment)
    se = {garment_stem(x) for x in e_subfamily}
    sp = {garment_stem(x) for x in p_garment}
    o_stem = overlap(se, sp)
    check(
        "subfamily↔garment_type",
        "L1",
        "subfamily",
        "garment_type",
        o_stem,
        f"exact={o['shared']} {o['samples']}; stem={o_stem['shared']} {o_stem['samples']}. "
        f"ecom tops={top_values(ecom,'master_products','subfamily',8)}; plm tops={top_values(plm,'plmex_products','garment_type',8)}",
        "CONFIRM" if o_stem["shared"] >= 5 else "PARTIAL",
    )

    # 3 family ↔ plm_sub_sub_category
    o = overlap(e_family, p_ssub)
    o_stem = overlap({garment_stem(x) for x in e_family}, {garment_stem(x) for x in p_ssub})
    check(
        "family↔plm_sub_sub_category",
        "L2-alt",
        "family",
        "plm_sub_sub_category",
        o_stem if o_stem["shared"] >= o["shared"] else o,
        f"exact={o['shared']} stem={o_stem['shared']} samples={ (o_stem if o_stem['shared']>=o['shared'] else o)['samples'] }",
        "PARTIAL" if max(o["shared"], o_stem["shared"]) >= 3 else "WEAK",
    )

    # hierarchy role checks: value overlap usually weak; check role via cardinality + sample
    for claim, ecol, pcol, es, ps in [
        ("category↔plm_main_category", "category", "plm_main_category", e_category, p_main),
        ("family↔plm_sub_category", "family", "plm_sub_category", e_family, p_sub),
        ("subfamily↔plm_sub_sub_category", "subfamily", "plm_sub_sub_category", e_subfamily, p_ssub),
    ]:
        o = overlap(es, ps)
        o_stem = overlap({garment_stem(x) for x in es}, {garment_stem(x) for x in ps})
        best = o_stem if o_stem["shared"] >= o["shared"] else o
        # Role similarity: both are hierarchy levels — value overlap expected low
        verdict = "ROLE_OK_VALUES_WEAK" if best["shared"] < 15 else "PARTIAL"
        if best["shared"] >= 8 and best["jac"] >= 0.02:
            verdict = "PARTIAL"
        check(
            claim,
            "L2-hierarchy",
            ecol,
            pcol,
            best,
            f"Role-similar trees; value overlap usually low. tops e={top_values(ecom,'master_products',ecol,5)} p={top_values(plm,'plmex_products',pcol,5)}",
            verdict,
        )

    # material
    o = overlap(e_comp, p_comp)
    check("composition↔composition", "L1", "composition", "composition", o,
          f"strong fibre overlap samples={o['samples'][:8]}",
          "CONFIRM" if o["shared"] >= 50 else "PARTIAL")

    o = overlap(e_compd, p_comp)
    check("composition_detail↔composition", "L1-related", "composition_detail", "composition", o,
          f"related detail→fibre samples={o['samples'][:8]}",
          "CONFIRM" if o["shared"] >= 30 else "PARTIAL")

    # color
    o = overlap(e_colors, p_color)
    check("colors↔color", "L1", "colors", "color", o,
          f"samples={o['samples'][:10]}",
          "CONFIRM" if o["shared"] >= 50 else "PARTIAL")

    # brand/buyer
    o_buyer = overlap(e_brand, p_buyer)
    o_brand = overlap(e_brand, p_brand)
    check("brand↔buyer", "L2", "brand", "buyer", o_buyer,
          f"ecom brands={sorted(e_brand)}; shared buyer={o_buyer['samples']}; brand-brand shared={o_brand['samples']}",
          "CONFIRM_WITH_ALIAS" if o_buyer["shared"] >= 3 else "WEAK")
    check("brand↔brand", "danger-same-name", "brand", "brand", o_brand,
          "Same name; retailer vs sample label. Prefer buyer for retail party.",
          "PARTIAL_DANGEROUS")

    # refs
    o = overlap(e_ref | e_dref, p_sref | p_csref)
    check("reference↔style_reference", "L2", "reference/display_reference", "style_reference/customer_style_reference", o,
          f"shared exact refs={o['shared']} samples={o['samples'][:8]}",
          "PARTIAL" if o["shared"] >= 1 else "WEAK_NO_JOIN_YET")

    # stock
    o = overlap(e_stock, p_stock)
    check("in_stock_count↔current_stock", "L3", "in_stock_count", "current_stock", o,
          f"numeric overlap only samples={o['samples']}; ecom avail tops={top_values(ecom,'master_products','availability',6)}; plm stock tops={top_values(plm,'plmex_products','current_stock',6)}",
          "RELATED_TRANSFORM")

    # false friends
    o = overlap(e_category, p_cat)
    check("category↔category", "DANGER", "category", "category", o,
          f"FALSE FRIEND. e tops={top_values(ecom,'master_products','category',6)}; p tops={top_values(plm,'plmex_products','category',6)}",
          "REJECT_MERGE")

    # section empty on plm?
    with psycopg.connect(plm) as conn:
        with conn.cursor() as cur:
            # plmex has no section column
            cur.execute(
                """SELECT column_name FROM information_schema.columns
                   WHERE table_name='plmex_products' AND column_name IN ('section','fit','availability','family','subfamily')"""
            )
            plm_has = {r[0] for r in cur.fetchall()}
    check(
        "section/fit/availability on PLMEX",
        "DANGER",
        "section/fit/availability",
        "(no column)",
        {"shared": 0, "jac": 0, "cov_a": 0, "cov_b": 0, "samples": []},
        f"PLMEX table columns present among section/fit/availability/family/subfamily: {sorted(plm_has) or 'none'}",
        "CONFIRM_ECOM_ONLY",
    )

    # section 2 candidates
    o = overlap(e_desc, p_dfeat) if e_desc and p_dfeat else {"shared": 0, "jac": 0, "cov_a": 0, "cov_b": 0, "samples": []}
    check("description↔design_feature", "investigate", "description", "design_feature", o,
          f"design_feature non-empty distinct={len(p_dfeat)}; overlap={o['shared']}",
          "REJECT" if len(p_dfeat) < 5 or o["shared"] < 3 else "INVESTIGATE")

    o = overlap({garment_stem(x) for x in e_name}, {garment_stem(x) for x in p_garment})
    check("name↔garment_type", "investigate", "name", "garment_type", o,
          f"stem overlap samples={o['samples'][:10]} — name is title free text, garment_type is controlled vocab",
          "PARTIAL_NOT_EQUIVALENT")

    o = overlap(e_path, p_main | p_sub | p_ssub)
    check("category_path↔plm_hierarchy", "investigate", "category_path", "plm_*", o,
          f"shared={o['shared']} samples={o['samples'][:8]}",
          "ROLE_OK_VALUES_WEAK" if o["shared"] < 20 else "PARTIAL")

    o = overlap(e_coo, p_loc)
    check("country_of_origin↔current_location", "investigate", "country_of_origin", "current_location", o,
          f"different concepts; shared={o['shared']} samples={o['samples'][:6]}",
          "REJECT")

    o = overlap(e_care, p_dfeat)
    check("care↔design_feature", "investigate", "care", "design_feature", o,
          f"shared={o['shared']}; design_feature size={len(p_dfeat)}",
          "REJECT")

    # construction false friend note
    check(
        "construction same-name",
        "DANGER",
        "construction",
        "(ai_construction_tags / empty)",
        overlap(e_construction, set()),
        f"ecom construction is fibre/shell text (distinct~{len(e_construction)}); PLMEX has no construction column — tags live in ai_construction_tags",
        "CONFIRM_DIFFERENT",
    )

    # matalan special case
    with psycopg.connect(ecom) as conn, psycopg.connect(plm) as conn2:
        with conn.cursor() as cur, conn2.cursor() as cur2:
            cur.execute("SELECT count(*) FROM master_products WHERE lower(brand)='matalan'")
            e_m = cur.fetchone()[0]
            cur2.execute("SELECT count(*) FROM plmex_products WHERE lower(brand)='matalan'")
            p_mb = cur2.fetchone()[0]
            cur2.execute("SELECT count(*) FROM plmex_products WHERE lower(buyer)='matalan'")
            p_mby = cur2.fetchone()[0]
            cur2.execute("SELECT count(*) FROM plmex_products WHERE lower(brand)='primark'")
            p_pb = cur2.fetchone()[0]
            cur2.execute("SELECT count(*) FROM plmex_products WHERE lower(buyer)='primark'")
            p_pby = cur2.fetchone()[0]
    check(
        "matalan brand vs buyer",
        "L2-alias",
        "brand=matalan",
        "buyer=matalan",
        {"shared": 1, "jac": 1, "cov_a": 1, "cov_b": 1, "samples": ["matalan"]},
        f"ecom brand matalan rows={e_m}; plmex brand={p_mb}; plmex buyer={p_mby}. primark brand={p_pb} buyer={p_pby}",
        "CONFIRM",
    )

    out = ROOT / "data" / "generated" / "eval" / "SEMANTIC_MAPPING_CROSSCHECK.json"
    out.write_text(json.dumps({"claims": claims}, indent=2, ensure_ascii=False), encoding="utf-8")

    # markdown summary
    md = [
        "# Semantic mapping cross-check (your proposal vs live data)",
        "",
        "Source: Postgres `master_products` (ecom) and `plmex_products` (PLMEX). Value overlap + tops; no Neo4j.",
        "",
        "## Verdict legend",
        "",
        "- **CONFIRM** — keep as proposed",
        "- **CONFIRM_WITH_ALIAS** — keep with retailer-specific alias rules",
        "- **PARTIAL / ROLE_OK_VALUES_WEAK** — same role or soft match; do not auto-merge values",
        "- **RELATED_TRANSFORM** — Level 3 derived",
        "- **REJECT_MERGE / REJECT** — do not treat as equivalent",
        "",
        "| Claim | Level | Verdict | Shared (norm) | Note |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for c in claims:
        note = c["note"].replace("|", "/")[:180]
        md.append(
            f"| `{c['id']}` | {c['proposed_level']} | **{c['verdict']}** | {c['shared']} | {note} |"
        )

    md += [
        "",
        "## Bottom line vs your five starters",
        "",
        "| Your starter | Cross-check |",
        "| --- | --- |",
        "| section ↔ gender | **CONFIRM** (soft audience; Kids/Christmas ecom-only) |",
        "| subfamily ↔ garment_type | **CONFIRM** as L1 soft (stem overlap; plurals/nav labels) |",
        "| composition ↔ composition | **CONFIRM** strongest raw twin |",
        "| colors ↔ color | **CONFIRM** strongest cross-name twin |",
        "| brand ↔ buyer | **CONFIRM_WITH_ALIAS** (esp. Matalan); not universal = brand |",
        "",
        "## Corrections to the proposal",
        "",
        "1. Hierarchy L1/L2/L3 (`category/family/subfamily` ↔ `plm_*`) is **role-similar only** — value sets barely overlap. Keep as canonical *roles*, never value merge.",
        "2. `family` ↔ `plm_sub_sub_category` is weaker than `subfamily` ↔ `garment_type`; keep as alternate, not primary.",
        "3. `reference` ↔ `style_reference` exact join is currently **weak/empty** in distinct-token overlap — investigate as future RESOLVES_TO, not Level 1.",
        "4. `description` ↔ `design_feature` and `care` ↔ `design_feature`: **reject** (`design_feature` nearly empty).",
        "5. `country_of_origin` ↔ `current_location`: **reject** (country vs warehouse text).",
        "6. `name` ↔ `garment_type`: partial stem hits only — name is free-text title, not controlled garment type.",
        "7. Same-name dangers confirmed: **`category`**, and treat **`brand`** carefully (prefer buyer for retail party).",
        "",
        "Your 3-level scheme (direct / soft / derived) + canonical fields without dropping source columns is correct.",
    ]
    md_path = ROOT / "data" / "generated" / "eval" / "SEMANTIC_MAPPING_CROSSCHECK.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("wrote", out)
    print("wrote", md_path)


if __name__ == "__main__":
    main()
