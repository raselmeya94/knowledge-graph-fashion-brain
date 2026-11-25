"""Raw Postgres value overlap: master_products vs plmex_products (no KG/relations)."""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]


def _dsn_from_env(key: str, fallback: str) -> str:
    return (os.environ.get(key) or "").strip() or fallback


# Prefer env; fallbacks match local .env (dev only).
ECOM_DSN = _dsn_from_env(
    "FASHIONGPT_ECOM_DSN",
    "postgresql://postgres@127.0.0.1:5432/fashion_brain_live",
)
PLM_DSN = _dsn_from_env(
    "FASHIONGPT_PLMEX_DSN",
    "postgresql://postgres@127.0.0.1:5432/plmex_live",
)

# Skip free-text / ids / urls / timestamps — keep categorical-ish cols.
SKIP_ECOM = {
    "master_id",
    "catalog_product_ids",
    "seo_product_id",
    "product_ids",
    "primary_product_id",
    "reference",
    "display_reference",
    "description",
    "product_url",
    "product_url_with_variant",
    "product_urls",
    "source_url",
    "image",
    "images",
    "prices_seen",
    "content_hash",
    "colorway_hash",
    "scrape_job_id",
    "created_at",
    "updated_at",
    "scraped_at",
    "scraped_date",
    "scraped_time",
    "last_changed_at",
    "price",
    "original_price",
    "price_min",
    "price_max",
}
SKIP_PLM = {
    "code",
    "style_reference",
    "customer_style_reference",
    "created_at",
    "updated_at",
    "sample_making_date",
    "shipment_date",
    "current_location",
}


def norm(s: object) -> str:
    text = str(s).strip().lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def expand_cell(val: object, col: str) -> set[str]:
    out: set[str] = set()
    if val is None:
        return out
    if isinstance(val, (list, tuple)):
        for item in val:
            out |= expand_cell(item, col)
        return out
    if isinstance(val, dict):
        for key in ("name", "color", "material", "fibre", "fiber", "value", "label"):
            if key in val and val[key] is not None:
                out |= expand_cell(val[key], col)
        for item in val.values():
            if isinstance(item, str) and len(item) < 80:
                out |= expand_cell(item, col)
        return out
    if isinstance(val, (int, float, bool)):
        n = norm(val)
        if n:
            out.add(n)
        return out

    s = str(val).strip()
    if not s or s.lower() in {"none", "null", "nan"}:
        return out
    if s.startswith("[") or s.startswith("{"):
        try:
            return expand_cell(json.loads(s), col)
        except Exception:
            pass

    if col.startswith("ai_") or col in {"colors", "composition", "care", "construction"}:
        parts = re.split(r"[,;|/]+", s)
        if len(parts) > 1:
            for part in parts:
                n = norm(part)
                if n and 1 < len(n) < 60:
                    out.add(n)

    n = norm(s)
    if n and len(n) < 80:
        out.add(n)

    for match in re.finditer(r"(\d+)\s*%\s*([A-Za-z][A-Za-z \-]+)", s):
        mat = norm(match.group(2))
        if mat:
            out.add(mat)
    return out


def load_sets(dsn: str, table: str, skip: set[str]) -> dict[str, set[str]]:
    sets: dict[str, set[str]] = defaultdict(set)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                ORDER BY ordinal_position
                """,
                (table,),
            )
            cols = [(r[0], r[1]) for r in cur.fetchall()]
            dtype_by = dict(cols)
            use = [c for c, _ in cols if c not in skip]
            print(f"{table}: {len(cols)} cols, using {len(use)}", flush=True)
            for col in use:
                dtype = dtype_by[col]
                if dtype == "jsonb":
                    cur.execute(
                        f"SELECT DISTINCT {col}::text FROM {table} WHERE {col} IS NOT NULL LIMIT 50000"
                    )
                else:
                    cur.execute(
                        f"""
                        SELECT DISTINCT CAST({col} AS text)
                        FROM {table}
                        WHERE {col} IS NOT NULL AND CAST({col} AS text) <> ''
                        LIMIT 50000
                        """
                    )
                for (raw,) in cur.fetchall():
                    if dtype == "jsonb":
                        try:
                            parsed = json.loads(raw)
                        except Exception:
                            parsed = raw
                        sets[col] |= expand_cell(parsed, col)
                    else:
                        sets[col] |= expand_cell(raw, col)
                sets[col] = {v for v in sets[col] if v and v not in {"(empty)", "empty"}}
                print(f"  {col}: {len(sets[col])} norms", flush=True)
    return dict(sets)


def main() -> None:
    # Load DSN password from .env if present and env unset.
    env_path = ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "=" not in line or line.strip().startswith("#"):
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k in {"FASHIONGPT_ECOM_DSN", "FASHIONGPT_PLMEX_DSN"} and k not in os.environ:
                os.environ[k] = v

    ecom_dsn = os.environ.get("FASHIONGPT_ECOM_DSN") or ECOM_DSN
    plm_dsn = os.environ.get("FASHIONGPT_PLMEX_DSN") or PLM_DSN

    ecom = load_sets(ecom_dsn, "master_products", SKIP_ECOM)
    plm = load_sets(plm_dsn, "plmex_products", SKIP_PLM)

    rows: list[tuple] = []
    for ec, es in ecom.items():
        if len(es) < 2:
            continue
        for pc, ps in plm.items():
            if len(ps) < 2:
                continue
            inter = es & ps
            if not inter:
                continue
            j = len(inter) / len(es | ps)
            cov_e = len(inter) / len(es)
            cov_p = len(inter) / len(ps)
            score = len(inter) * (0.5 + 0.5 * max(cov_e, cov_p))
            samples = sorted(inter, key=len)[:12]
            rows.append((score, len(inter), j, cov_e, cov_p, ec, pc, samples, len(es), len(ps)))

    rows.sort(reverse=True)

    print("\n=== TOP COLUMN PAIRS BY SHARED VALUES ===")
    hdr = f"{'ecom_col':28} {'plmex_col':28} {'shared':>6} {'jac':>6} {'covE':>6} {'covP':>6} {'|E|':>5} {'|P|':>5}"
    print(hdr)
    for score, n, j, ce, cp, ec, pc, samples, ne, np_ in rows[:60]:
        print(
            f"{ec:28} {pc:28} {n:6d} {j:6.3f} {ce:6.3f} {cp:6.3f} {ne:5d} {np_:5d} {samples[:6]}"
        )

    print("\n=== BEST PLMEX MATCH PER ECOM COLUMN ===")
    best: dict[str, tuple] = {}
    for score, n, j, ce, cp, ec, pc, samples, ne, np_ in rows:
        if ec not in best or (n, j) > (best[ec][0], best[ec][1]):
            best[ec] = (n, j, ce, cp, pc, samples, ne, np_)
    for ec, (n, j, ce, cp, pc, samples, ne, np_) in sorted(best.items(), key=lambda x: -x[1][0]):
        print(
            f"{ec:28} -> {pc:28} shared={n:4d} jac={j:.3f} covE={ce:.3f} covP={cp:.3f} eg={samples[:8]}"
        )

    print("\n=== BEST ECOM MATCH PER PLMEX COLUMN ===")
    bestp: dict[str, tuple] = {}
    for score, n, j, ce, cp, ec, pc, samples, ne, np_ in rows:
        if pc not in bestp or (n, j) > (bestp[pc][0], bestp[pc][1]):
            bestp[pc] = (n, j, ce, cp, ec, samples, ne, np_)
    for pc, (n, j, ce, cp, ec, samples, ne, np_) in sorted(bestp.items(), key=lambda x: -x[1][0]):
        print(
            f"{pc:28} -> {ec:28} shared={n:4d} jac={j:.3f} covE={ce:.3f} covP={cp:.3f} eg={samples[:8]}"
        )

    # Strong pairs: shared>=5 and (jac>=0.05 or cov>=0.15)
    strong = [
        {
            "ecom": ec,
            "plmex": pc,
            "shared": n,
            "jaccard": round(j, 4),
            "cov_ecom": round(ce, 4),
            "cov_plmex": round(cp, 4),
            "ecom_unique": ne,
            "plmex_unique": np_,
            "sample_shared": samples,
        }
        for score, n, j, ce, cp, ec, pc, samples, ne, np_ in rows
        if n >= 5 and (j >= 0.05 or ce >= 0.15 or cp >= 0.15)
    ]

    out_dir = ROOT / "data" / "generated" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "normalized distinct token overlap on Postgres columns (no Neo4j)",
        "ecom_table": "master_products",
        "plmex_table": "plmex_products",
        "strong_pairs": strong,
        "top_pairs": [
            {
                "ecom": ec,
                "plmex": pc,
                "shared": n,
                "jaccard": round(j, 4),
                "cov_ecom": round(ce, 4),
                "cov_plmex": round(cp, 4),
                "ecom_unique": ne,
                "plmex_unique": np_,
                "sample_shared": samples,
            }
            for score, n, j, ce, cp, ec, pc, samples, ne, np_ in rows[:100]
        ],
        "best_per_ecom": {
            ec: {
                "plmex": pc,
                "shared": n,
                "jaccard": round(j, 4),
                "cov_ecom": round(ce, 4),
                "cov_plmex": round(cp, 4),
                "samples": samples,
            }
            for ec, (n, j, ce, cp, pc, samples, ne, np_) in best.items()
        },
        "best_per_plmex": {
            pc: {
                "ecom": ec,
                "shared": n,
                "jaccard": round(j, 4),
                "cov_ecom": round(ce, 4),
                "cov_plmex": round(cp, 4),
                "samples": samples,
            }
            for pc, (n, j, ce, cp, ec, samples, ne, np_) in bestp.items()
        },
    }
    out = out_dir / "RAW_VALUE_OVERLAP_ECOM_PLMEX.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# Raw value overlap: `master_products` ↔ `plmex_products`",
        "",
        "Postgres distinct values only (normalized lowercase tokens). No Neo4j / relationships.",
        "",
        "## Strong pairs (shared ≥ 5 and jac≥0.05 or coverage≥0.15)",
        "",
        "| Ecom column | PLMEX column | Shared | Jaccard | Cov→E | Cov→P | Sample shared |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in strong[:80]:
        samples = ", ".join(row["sample_shared"][:6])
        md_lines.append(
            f"| `{row['ecom']}` | `{row['plmex']}` | {row['shared']} | {row['jaccard']:.3f} | "
            f"{row['cov_ecom']:.3f} | {row['cov_plmex']:.3f} | {samples} |"
        )
    md_lines += [
        "",
        "## Best PLMEX column per ecom column",
        "",
        "| Ecom | Best PLMEX | Shared | Jaccard | Samples |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for ec, (n, j, ce, cp, pc, samples, ne, np_) in sorted(best.items(), key=lambda x: -x[1][0]):
        md_lines.append(
            f"| `{ec}` | `{pc}` | {n} | {j:.3f} | {', '.join(samples[:6])} |"
        )
    md = out_dir / "RAW_VALUE_OVERLAP_ECOM_PLMEX.md"
    md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print("wrote", out)
    print("wrote", md)


if __name__ == "__main__":
    main()
