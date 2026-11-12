"""
Build a simple Excel summary:

  column headers = database field names
  each row down that column = one unique value

Example:
  category     | brand   | section
  T-shirts     | zara    | WOMAN
  Dresses      | primark | MAN
  ...
"""
from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
OUT = Path(__file__).resolve().parents[1] / "pg_db_tables_schema" / "ecom_column_study.xlsx"

# Product tables only
TABLES = [
    "master_products",
    "zara_products",
    "primark_products",
    "matalan_products",
    "c_and_a_products",
    "walmart_products",
    "calvin_klein_products",
    "zara_colorways",
    "primark_colorways",
    "matalan_colorways",
    "c_and_a_colorways",
    "walmart_colorways",
    "calvin_klein_colorways",
]

SHEET_NAME = {
    "master_products": "All brands",
    "zara_products": "Zara",
    "primark_products": "Primark",
    "matalan_products": "Matalan",
    "c_and_a_products": "C and A",
    "walmart_products": "Walmart",
    "calvin_klein_products": "Calvin Klein",
    "zara_colorways": "Zara colors",
    "primark_colorways": "Primark colors",
    "matalan_colorways": "Matalan colors",
    "c_and_a_colorways": "C and A colors",
    "walmart_colorways": "Walmart colors",
    "calvin_klein_colorways": "CK colors",
}

# Skip these as summary columns (too many ids / links / blobs — not useful as a list)
SKIP_COLS = {
    "catalog_product_ids",
    "product_ids",
    "zara_ids",
    "composition_detail",
    "care",
    "care_items",
    "product_features",
    "sizes",
    "size_names",
    "badges",
    "product_tags",
    "prices_seen",
    "images",
    "colors",
    "color_variants",
    "colorway_tiles",
    "product_urls",
    "grid_positions",
    "variant_offers",
    "specifications",
    "content_hash",
    "colorway_hash",
    "image",
    "product_url",
    "product_url_with_variant",
    "source_url",
    "description",
    "description_short",
    "description_long",
}

# Prefer these first (left side of sheet) — easy to read
PRIORITY_COLS = [
    "brand",
    "section",
    "gender",
    "category",
    "family",
    "subfamily",
    "category_path",
    "name",
    "composition",
    "construction",
    "fit",
    "fit_details",
    "color",
    "availability",
    "currency",
    "price",
    "original_price",
    "price_min",
    "price_max",
    "country_of_origin",
    "kind",
    "type",
    "reference",
    "display_reference",
]

SAME_MEANING_GROUPS = [
    ("Product type / category", ["category", "family", "subfamily", "category_path", "kind", "type"]),
    ("Who it is for", ["section", "gender"]),
    ("Brand", ["brand", "product_brand"]),
    ("Materials", ["composition"]),
    ("Fit & features", ["fit", "fit_details"]),
    ("Colors", ["color", "selected_color"]),
    ("Price", ["price", "original_price", "price_min", "price_max", "currency"]),
    ("Stock", ["availability", "in_stock_count", "out_stock_count"]),
    ("Product id / code", ["master_id", "reference", "display_reference", "seo_product_id", "primary_product_id", "primark_id", "matalan_id", "c_and_a_id", "walmart_id", "calvin_klein_id", "primary_zara_id", "sku"]),
]

ECOM_TO_PLMEX = [
    ("composition", "composition", "Same", "Both = fabric / materials"),
    ("brand", "brand", "Same", "Both = brand name"),
    ("color", "color", "Same", "Both = color name"),
    ("category", "garment_type", "Same", "Both = product type (t-shirt, dress…)"),
    ("category", "category", "Related", "Same word; ecom more detailed"),
    ("family", "plm_sub_category", "Same", "Mid-level product group"),
    ("subfamily", "plm_sub_sub_category", "Same", "Fine product group"),
    ("section", "gender", "Same", "men / women / unisex"),
    ("gender", "gender", "Same", "Gender"),
    ("reference", "style_reference", "Same", "Style / reference number"),
    ("created_at", "created_at", "Same", "Created date"),
    ("updated_at", "updated_at", "Same", "Updated date"),
    ("master_id", "code", "Similar key", "Main product key (different systems)"),
    ("name", "garment_type", "Related", "Title often names the garment"),
    ("kind", "garment_type", "Related", "Product kind ≈ garment type"),
    ("type", "garment_type", "Related", "Product type ≈ garment type"),
    ("in_stock_count", "current_stock", "Related", "How many in stock"),
    ("availability", "current_stock", "Related", "Stock status vs stock number"),
    ("fit", "style_feature", "Related", "Fit / style"),
    ("product_brand", "brand", "Related", "Label brand"),
]

GREEN = PatternFill("solid", fgColor="C6EFCE")
YELLOW = PatternFill("solid", fgColor="FFEB9C")
ORANGE = PatternFill("solid", fgColor="FCE4D6")
HEADER = PatternFill("solid", fgColor="1F4E79")
HEADER_FONT = Font(color="FFFFFF", bold=True, size=11)
TITLE = Font(bold=True, size=14, color="1F4E79")
BOLD = Font(bold=True)

# Cap only for extreme id-like columns so Excel stays usable
HARD_CAP = 5000


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def list_columns(conn, table: str) -> list[str]:
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
        return [(r[0], r[1]) for r in cur.fetchall()]


def all_unique_values(conn, table: str, column: str, dtype: str) -> list[str]:
    t, c = qident(table), qident(column)
    cast = f"{c}::text"
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        if cur.fetchone()[0] == 0:
            return []
        cur.execute(
            f"""
            SELECT {cast} AS v
            FROM {t}
            WHERE {c} IS NOT NULL AND BTRIM({cast}) <> ''
            GROUP BY 1
            ORDER BY 1
            LIMIT %s
            """,
            (HARD_CAP,),
        )
        return [r[0] for r in cur.fetchall()]


def order_columns(cols: list[str]) -> list[str]:
    cols = [c for c in cols if c not in SKIP_COLS]
    first = [c for c in PRIORITY_COLS if c in cols]
    rest = sorted(c for c in cols if c not in first)
    return first + rest


def style_header(ws, ncols: int):
    for i in range(1, ncols + 1):
        cell = ws.cell(1, i)
        cell.fill = HEADER
        cell.font = HEADER_FONT
        cell.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")


def write_unique_sheet(wb, sheet_title: str, columns: list[str], values_by_col: dict[str, list[str]]):
    """Header = column names; each column filled down with its unique values."""
    # Excel sheet name max 31 chars
    title = sheet_title[:31]
    ws = wb.create_sheet(title)
    if not columns:
        ws["A1"] = "No fields"
        return ws

    ws.append(columns)
    style_header(ws, len(columns))
    max_len = max((len(values_by_col.get(c, [])) for c in columns), default=0)

    for row_i in range(max_len):
        row = []
        for c in columns:
            vals = values_by_col.get(c, [])
            row.append(vals[row_i] if row_i < len(vals) else None)
        ws.append(row)

    # Second header row note: count of unique values
    # Put counts in row 1 comment via row height; add a thin count row above? Keep simple — put count in freeze row 2 as subtitle by inserting:
    # Actually insert after header a "count" row then values — user wants values as rows.
    # Add count in a note at top-left of used range via auto filter only.

    ws.auto_filter.ref = f"A1:{get_column_letter(len(columns))}{ws.max_row}"
    ws.freeze_panes = "A2"
    ws.row_dimensions[1].height = 28
    for i, c in enumerate(columns, start=1):
        n = len(values_by_col.get(c, []))
        ws.cell(1, i).value = f"{c}  ({n})"
        width = 18
        for v in values_by_col.get(c, [])[:40]:
            width = max(width, min(42, len(str(v)) + 2))
        ws.column_dimensions[get_column_letter(i)].width = width
    return ws


def build():
    ecom_dsn = os.getenv("FASHIONGPT_ECOM_DSN") or ""
    plmex_dsn = os.getenv("FASHIONGPT_PLMEX_DSN") or ""
    if not ecom_dsn or not plmex_dsn:
        raise SystemExit("Need FASHIONGPT_ECOM_DSN and FASHIONGPT_PLMEX_DSN")

    wb = Workbook()
    start = wb.active
    start.title = "Start here"
    start["A1"] = "How to read this file"
    start["A1"].font = TITLE
    for i, line in enumerate(
        [
            "",
            "Each brand sheet looks like this:",
            "",
            "   category          brand        section",
            "   T-shirts          zara         WOMAN",
            "   Dresses           primark      MAN",
            "   Knitwear          …            …",
            "",
            "→ Top row = field name (and how many different values)",
            "→ Every cell under it = one unique value from the database",
            "→ Values in different columns are NOT linked to each other — each column is its own list",
            "",
            "Sheets:",
            "  • All brands / Zara / Primark / …  = unique values for that table",
            "  • Same meaning                     = fields that mean the same thing",
            "  • Match to PLMEX                   = ecom field ↔ PLMEX field (colored)",
            "",
            "Colors on Match sheet:  Green=Same   Yellow=Related   Orange=Similar key",
            f"Note: lists are capped at {HARD_CAP} values for very large id fields.",
        ],
        start=2,
    ):
        start.cell(i, 1).value = line
    start["A17"].fill = GREEN
    start["A18"] = "Green = Same"
    start["A18"].fill = GREEN
    start["A19"] = "Yellow = Related"
    start["A19"].fill = YELLOW
    start["A20"] = "Orange = Similar key"
    start["A20"].fill = ORANGE
    start.column_dimensions["A"].width = 95

    # Collect uniques per table
    all_col_tables = defaultdict(set)
    master_values = {}

    print("Building unique-value summary sheets…")
    with psycopg.connect(ecom_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema='public' AND table_type='BASE TABLE'
                """
            )
            existing = {r[0] for r in cur.fetchall()}

        for table in TABLES:
            if table not in existing:
                continue
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {qident(table)}")
                n = cur.fetchone()[0]
            cols_meta = list_columns(conn, table)
            col_names = order_columns([c for c, _ in cols_meta])
            dtype = {c: d for c, d in cols_meta}
            values_by_col = {}
            print(f"  {table} ({n} rows)…")
            for col in col_names:
                vals = all_unique_values(conn, table, col, dtype[col])
                values_by_col[col] = vals
                all_col_tables[col].add(table)
                print(f"    {col}: {len(vals)} unique")
            write_unique_sheet(wb, SHEET_NAME[table], col_names, values_by_col)
            if table == "master_products":
                master_values = values_by_col

    # Same meaning
    ws2 = wb.create_sheet("Same meaning")
    ws2.append(["Group", "Field name", "Also in these tables", "Example unique values"])
    style_header(ws2, 4)
    for group, members in SAME_MEANING_GROUPS:
        for col in members:
            if col not in all_col_tables:
                continue
            examples = master_values.get(col) or []
            if not examples:
                # any table
                examples = []
            tables = " | ".join(sorted(SHEET_NAME.get(t, t) for t in all_col_tables[col]))
            idx = ws2.max_row + 1
            ws2.append(
                [
                    group,
                    col,
                    tables,
                    "  ·  ".join(str(x) for x in examples[:15]),
                ]
            )
            ws2.cell(idx, 1).fill = GREEN
    ws2.auto_filter.ref = f"A1:D{ws2.max_row}"
    ws2.freeze_panes = "A2"
    ws2.column_dimensions["A"].width = 28
    ws2.column_dimensions["B"].width = 22
    ws2.column_dimensions["C"].width = 40
    ws2.column_dimensions["D"].width = 70

    # Match to PLMEX + plmex unique summary sheet
    plmex_values = {}
    print("Reading PLMEX unique values…")
    with psycopg.connect(plmex_dsn) as conn:
        cols = list_columns(conn, "plmex_products")
        # keep useful PLMEX columns (skip empty design_feature still include)
        plmex_col_names = [c for c, _ in cols]
        dtype = {c: d for c, d in cols}
        for col in plmex_col_names:
            plmex_values[col] = all_unique_values(conn, "plmex_products", col, dtype[col])
            print(f"  plmex.{col}: {len(plmex_values[col])}")
        write_unique_sheet(wb, "PLMEX values", plmex_col_names, plmex_values)

    ws3 = wb.create_sheet("Match to PLMEX")
    ws3.append(
        [
            "Ecom field",
            "PLMEX field",
            "Match",
            "Plain English",
            "Ecom examples",
            "PLMEX examples",
        ]
    )
    style_header(ws3, 6)
    order = {"Same": 0, "Related": 1, "Similar key": 2}
    for ecol, pcol, match, reason in sorted(ECOM_TO_PLMEX, key=lambda x: (order[x[2]], x[0])):
        if ecol not in all_col_tables:
            continue
        idx = ws3.max_row + 1
        e_ex = master_values.get(ecol) or []
        p_ex = plmex_values.get(pcol) or []
        ws3.append(
            [
                ecol,
                pcol,
                match,
                reason,
                "  ·  ".join(str(x) for x in e_ex[:12]),
                "  ·  ".join(str(x) for x in p_ex[:12]),
            ]
        )
        fill = GREEN if match == "Same" else YELLOW if match == "Related" else ORANGE
        for c in (1, 2, 3):
            ws3.cell(idx, c).fill = fill
        ws3.cell(idx, 3).font = BOLD
    ws3.auto_filter.ref = f"A1:F{ws3.max_row}"
    ws3.freeze_panes = "A2"
    for col, w in zip("ABCDEF", (16, 22, 14, 45, 50, 50)):
        ws3.column_dimensions[col].width = w

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    build()
