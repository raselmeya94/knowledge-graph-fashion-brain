"""Build a PDF from pattern_benchmark_50_results.json.

Cover page: overall performance + category scorecard.
Detail pages: serial + question + answer only (no per-question verdict,
thinking, reasoning, or references).
"""
from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics.widgets.markers import makeMarker
from reportlab.platypus import (
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "data" / "generated" / "eval" / "pattern_benchmark_50_results.json"
OUT_PATH = ROOT / "data" / "generated" / "eval" / "pattern_benchmark_50_results.pdf"

INK = colors.HexColor("#111827")
MUTED = colors.HexColor("#4b5563")
LINE = colors.HexColor("#d1d5db")
PANEL = colors.HexColor("#f3f4f6")
TEAL = colors.HexColor("#0f766e")
TEAL_SOFT = colors.HexColor("#ccfbf1")
Q_BG = colors.HexColor("#ecfeff")
A_BG = colors.HexColor("#f8fafc")
VALID_BG = colors.HexColor("#dcfce7")
VALID_FG = colors.HexColor("#166534")
PARTIAL_BG = colors.HexColor("#fef3c7")
PARTIAL_FG = colors.HexColor("#92400e")
INVALID_BG = colors.HexColor("#fee2e2")
INVALID_FG = colors.HexColor("#991b1b")
HEADER_BG = colors.HexColor("#0f172a")

FONT = "Helvetica"
FONT_BOLD = "Helvetica-Bold"

# Prefer fonts that cover Latin + CJK (brand names like 安奈儿).
_FONT_CANDIDATES = (
    (r"C:\Windows\Fonts\msyh.ttc", 0, "AppYaHei", "AppYaHeiBold"),
    (r"C:\Windows\Fonts\simsun.ttc", 0, "AppSimSun", "AppSimSunBold"),
    (r"C:\Windows\Fonts\arial.ttf", None, "AppArial", "AppArialBold"),
    (r"C:\Windows\Fonts\calibri.ttf", None, "AppCalibri", "AppCalibriBold"),
)
_BOLD_FALLBACK = {
    "AppYaHei": r"C:\Windows\Fonts\msyhbd.ttc",
    "AppSimSun": r"C:\Windows\Fonts\simsun.ttc",
    "AppArial": r"C:\Windows\Fonts\arialbd.ttf",
    "AppCalibri": r"C:\Windows\Fonts\calibrib.ttf",
}

for path, sub, regular, bold_name in _FONT_CANDIDATES:
    try:
        kwargs = {"subfontIndex": sub} if sub is not None else {}
        pdfmetrics.registerFont(TTFont(regular, path, **kwargs))
        bold_path = _BOLD_FALLBACK.get(regular, path)
        try:
            pdfmetrics.registerFont(TTFont(bold_name, bold_path, **kwargs))
        except Exception:
            pdfmetrics.registerFont(TTFont(bold_name, path, **kwargs))
        FONT = regular
        FONT_BOLD = bold_name
        break
    except Exception:
        continue


def esc(s: object) -> str:
    return html.escape(str(s or ""), quote=False)


def css_hex(c) -> str:
    try:
        return f"#{int(c.red * 255):02x}{int(c.green * 255):02x}{int(c.blue * 255):02x}"
    except Exception:
        return "#111827"


def status_colors(status: str) -> tuple:
    st = (status or "").upper()
    if st == "VALID":
        return VALID_BG, VALID_FG
    if st == "PARTIAL":
        return PARTIAL_BG, PARTIAL_FG
    if st in {"INVALID", "ERROR"}:
        return INVALID_BG, INVALID_FG
    return PANEL, MUTED


def inline_md(text: str) -> str:
    """Escape + light markdown inline → ReportLab rich text."""
    s = esc(text)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"__(.+?)__", r"<b>\1</b>", s)
    s = re.sub(r"`([^`]+)`", r'<font face="Courier" size="8">\1</font>', s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", s)
    return s


def soft_trunc(text: str, limit: int) -> str:
    """Truncate without destroying newlines (cut at paragraph/line boundary when possible)."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    # Prefer breaking at blank line / newline near the end.
    for sep in ("\n\n", "\n", ". ", " "):
        idx = cut.rfind(sep)
        if idx >= int(limit * 0.6):
            cut = cut[: idx + len(sep)].rstrip()
            break
    return cut.rstrip() + "\n…"


def parse_md_table(block_lines: list[str]) -> list[list[str]] | None:
    rows: list[list[str]] = []
    for line in block_lines:
        if re.match(r"^\s*\|?\s*:?-{2,}", line):
            continue
        if not line.strip().startswith("|"):
            return None
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    return rows or None


def _as_float_list(values) -> list[float]:
    out: list[float] = []
    for v in values or []:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def render_chart_drawing(spec: dict) -> Drawing | None:
    """Render Fashion Brain ```chart JSON (bar/line) into a ReportLab Drawing."""
    if not isinstance(spec, dict):
        return None
    ctype = str(spec.get("type") or "bar").casefold()
    title = str(spec.get("title") or ctype.title())
    labels = [str(x) for x in (spec.get("labels") or [])]
    values = _as_float_list(spec.get("values"))
    series = spec.get("series")
    if series and isinstance(series, list) and series:
        first = series[0] if isinstance(series[0], dict) else None
        if first and not values:
            values = _as_float_list(first.get("values"))
        if first and first.get("name") and not title:
            title = str(first.get("name"))
    if not labels or not values:
        return None
    n = min(len(labels), len(values), 12)
    labels = labels[:n]
    values = values[:n]
    width, height = 460, 190
    drawing = Drawing(width, height)
    drawing.add(Rect(0, 0, width, height, fillColor=colors.HexColor("#f8fafc"), strokeColor=LINE, strokeWidth=0.6))
    drawing.add(String(10, height - 14, title[:70], fontName=FONT_BOLD, fontSize=9, fillColor=INK))

    if ctype == "line":
        plot = LinePlot()
        plot.x = 45
        plot.y = 35
        plot.height = 125
        plot.width = 390
        xs = list(range(n))
        plot.data = [list(zip(xs, values))]
        plot.joinedLines = 1
        plot.lines[0].strokeColor = TEAL
        plot.lines[0].strokeWidth = 1.8
        plot.lines[0].symbol = makeMarker("FilledCircle")
        plot.lines[0].symbol.size = 4
        plot.lines[0].symbol.fillColor = TEAL
        ymin = min(values)
        ymax = max(values)
        pad = max(abs(ymax - ymin) * 0.1, 1.0)
        plot.yValueAxis.valueMin = ymin - pad
        plot.yValueAxis.valueMax = ymax + pad
        plot.xValueAxis.valueMin = 0
        plot.xValueAxis.valueMax = max(n - 1, 1)
        plot.xValueAxis.labels.boxAnchor = "n"
        plot.xValueAxis.labels.angle = 40
        plot.xValueAxis.labels.fontName = FONT
        plot.xValueAxis.labels.fontSize = 6
        # Show a subset of labels to avoid clutter.
        step = max(1, n // 6)
        plot.xValueAxis.labelTextFormat = lambda v: (
            labels[int(v)][:10] if abs(v - int(v)) < 1e-6 and int(v) % step == 0 and 0 <= int(v) < n else ""
        )
        plot.yValueAxis.labels.fontName = FONT
        plot.yValueAxis.labels.fontSize = 7
        drawing.add(plot)
    else:
        chart = VerticalBarChart()
        chart.x = 40
        chart.y = 40
        chart.height = 120
        chart.width = 400
        chart.data = [values]
        chart.categoryAxis.categoryNames = [lab[:14] for lab in labels]
        chart.categoryAxis.labels.boxAnchor = "ne"
        chart.categoryAxis.labels.dx = -2
        chart.categoryAxis.labels.dy = -2
        chart.categoryAxis.labels.angle = 35
        chart.categoryAxis.labels.fontName = FONT
        chart.categoryAxis.labels.fontSize = 7
        chart.valueAxis.valueMin = 0
        chart.valueAxis.valueMax = max(values) * 1.15 if values else 1
        chart.valueAxis.labels.fontName = FONT
        chart.valueAxis.labels.fontSize = 7
        chart.barWidth = 12
        chart.groupSpacing = 12
        chart.barSpacing = 2
        chart.bars[0].fillColor = TEAL
        chart.bars[0].strokeColor = colors.HexColor("#115e59")
        drawing.add(chart)
    return drawing


def answer_to_flowables(text: str, styles, *, limit: int = 6500) -> list:
    """Turn chatbot markdown answer into readable PDF flowables (tables + charts)."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    # Pull ```chart JSON out before truncation so plots are not cut mid-JSON.
    charts: list[dict] = []

    def _store_chart(match: re.Match) -> str:
        raw = match.group(1).strip()
        try:
            charts.append(json.loads(raw))
            return f"\n%%CHART_{len(charts) - 1}%%\n"
        except Exception:
            return "\n[chart data]\n"

    text = re.sub(r"```chart\s*(\{.*?\})\s*```", _store_chart, text, flags=re.S)
    # Drop other fenced code blocks (keep a short marker).
    text = re.sub(r"```[\w-]*\s*.*?```", "\n[code omitted]\n", text, flags=re.S)
    text = soft_trunc(text, limit)
    if not text.strip() and not charts:
        return [Paragraph("—", styles["AnswerBody"])]

    flow: list = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        raw = line.rstrip()

        chart_m = re.fullmatch(r"%%CHART_(\d+)%%", raw.strip())
        if chart_m:
            idx = int(chart_m.group(1))
            if 0 <= idx < len(charts):
                drawing = render_chart_drawing(charts[idx])
                if drawing is not None:
                    flow.append(Spacer(1, 4))
                    flow.append(drawing)
                    flow.append(Spacer(1, 6))
                else:
                    flow.append(
                        Paragraph(
                            f"<i>[chart: {esc((charts[idx] or {}).get('title') or 'unavailable')}]</i>",
                            styles["Muted"],
                        )
                    )
            i += 1
            continue

        # Markdown table
        if raw.strip().startswith("|") and i + 1 < len(lines) and re.search(
            r"\|?\s*:?-{2,}", lines[i + 1]
        ):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            rows = parse_md_table(block)
            if rows:
                # Cap tall tables so a single flowable cannot exceed the page frame.
                if len(rows) > 16:
                    rows = rows[:16]
                    rows.append(["…"] * len(rows[0]))
                col_n = max(len(r) for r in rows)
                data = []
                for ri, row in enumerate(rows):
                    padded = row + [""] * (col_n - len(row))
                    style = styles["TableHead"] if ri == 0 else styles["TableCell"]
                    data.append([Paragraph(inline_md(c), style) for c in padded])
                width = 168 * mm / max(col_n, 1)
                t = Table(data, colWidths=[width] * col_n, repeatRows=1)
                t.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("GRID", (0, 0), (-1, -1), 0.3, LINE),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            (
                                "ROWBACKGROUNDS",
                                (0, 1),
                                (-1, -1),
                                [colors.white, PANEL],
                            ),
                            ("TOPPADDING", (0, 0), (-1, -1), 3),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                            ("LEFTPADDING", (0, 0), (-1, -1), 4),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ]
                    )
                )
                flow.append(t)
                flow.append(Spacer(1, 6))
            continue

        # Headings
        if raw.startswith("### "):
            flow.append(Paragraph(inline_md(raw[4:]), styles["AnswerH3"]))
            i += 1
            continue
        if raw.startswith("## "):
            flow.append(Paragraph(inline_md(raw[3:]), styles["AnswerH2"]))
            i += 1
            continue
        if raw.startswith("# "):
            flow.append(Paragraph(inline_md(raw[2:]), styles["AnswerH2"]))
            i += 1
            continue

        # Bullet / numbered list block
        if re.match(r"^\s*([-*•]|\d+\.)\s+", raw):
            items = []
            while i < len(lines) and re.match(r"^\s*([-*•]|\d+\.)\s+", lines[i]):
                item_txt = re.sub(r"^\s*([-*•]|\d+\.)\s+", "", lines[i]).strip()
                items.append(
                    ListItem(
                        Paragraph(inline_md(item_txt), styles["AnswerBody"]),
                        leftIndent=8,
                        bulletColor=TEAL,
                    )
                )
                i += 1
            flow.append(
                ListFlowable(
                    items,
                    bulletType="bullet",
                    start="•",
                    leftIndent=12,
                    bulletFontName=FONT,
                    bulletFontSize=8,
                )
            )
            flow.append(Spacer(1, 3))
            continue

        if not raw.strip():
            flow.append(Spacer(1, 4))
            i += 1
            continue

        flow.append(Paragraph(inline_md(raw), styles["AnswerBody"]))
        i += 1

    return flow or [Paragraph("—", styles["AnswerBody"])]


def section_label(title: str, styles, bg) -> Table:
    """Compact colored label that can sit above page-breakable content."""
    head = Table(
        [[Paragraph(f"<b>{esc(title)}</b>", styles["BlockLabel"])]],
        colWidths=[180 * mm],
    )
    head.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), bg),
                ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return head


def short_labeled_block(title: str, inner: list, styles, bg) -> Table:
    """Only for short content (question / short thinking)."""
    head = Paragraph(f"<b>{esc(title)}</b>", styles["BlockLabel"])
    body = Table([[x] for x in inner], colWidths=[168 * mm])
    body.setStyle(
        TableStyle(
            [
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ]
        )
    )
    wrap = Table([[head], [body]], colWidths=[172 * mm])
    wrap.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), bg),
                ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, 0), 6),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
                ("TOPPADDING", (0, 1), (-1, 1), 2),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 6),
            ]
        )
    )
    return wrap


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="CoverBrand",
            fontName=FONT_BOLD,
            fontSize=28,
            leading=34,
            textColor=colors.white,
            alignment=TA_CENTER,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverSub",
            fontName=FONT,
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#e5e7eb"),
            alignment=TA_CENTER,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionH",
            fontName=FONT_BOLD,
            fontSize=14,
            leading=18,
            textColor=INK,
            spaceBefore=10,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            fontName=FONT,
            fontSize=9,
            leading=12,
            textColor=INK,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Muted",
            fontName=FONT,
            fontSize=8,
            leading=11,
            textColor=MUTED,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CardTitle",
            fontName=FONT_BOLD,
            fontSize=11,
            leading=14,
            textColor=INK,
        )
    )
    styles.add(
        ParagraphStyle(
            name="QuestionText",
            fontName=FONT_BOLD,
            fontSize=11,
            leading=15,
            textColor=INK,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="AnswerBody",
            fontName=FONT,
            fontSize=9,
            leading=13,
            textColor=INK,
            spaceAfter=2,
        )
    )
    styles.add(
        ParagraphStyle(
            name="AnswerH2",
            fontName=FONT_BOLD,
            fontSize=10,
            leading=13,
            textColor=INK,
            spaceBefore=4,
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="AnswerH3",
            fontName=FONT_BOLD,
            fontSize=9,
            leading=12,
            textColor=MUTED,
            spaceBefore=3,
            spaceAfter=2,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BlockLabel",
            fontName=FONT_BOLD,
            fontSize=8,
            leading=10,
            textColor=TEAL,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Badge",
            fontName=FONT_BOLD,
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="StatBig",
            fontName=FONT_BOLD,
            fontSize=22,
            leading=26,
            textColor=TEAL,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="StatLabel",
            fontName=FONT,
            fontSize=8,
            leading=10,
            textColor=MUTED,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Footer",
            fontName=FONT,
            fontSize=7,
            leading=9,
            textColor=MUTED,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Cell",
            fontName=FONT,
            fontSize=8,
            leading=10,
            textColor=INK,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CellHead",
            fontName=FONT_BOLD,
            fontSize=8,
            leading=10,
            textColor=colors.white,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableCell",
            fontName=FONT,
            fontSize=7.5,
            leading=10,
            textColor=INK,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableHead",
            fontName=FONT_BOLD,
            fontSize=7.5,
            leading=10,
            textColor=colors.white,
        )
    )
    return styles


def cover_page(styles, summary: dict, by_cat: dict, meta: dict) -> list:
    story = []
    hero = Table(
        [
            [Paragraph("Fashion Brain", styles["CoverBrand"])],
            [
                Paragraph(
                    "Pattern Benchmark 50 — Live DB Validation Report",
                    styles["CoverSub"],
                )
            ],
            [
                Paragraph(
                    f"Mode: {esc(meta.get('mode') or 'ws/ask-chatbot+conversation')}  ·  "
                    f"Generated: {esc(meta.get('generated_at') or '')}",
                    styles["CoverSub"],
                )
            ],
        ],
        colWidths=[180 * mm],
    )
    hero.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), HEADER_BG),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    story.append(hero)
    story.append(Spacer(1, 14))

    total_q = int(summary.get("total") or 0) or sum(
        int((by_cat.get(k) or {}).get("total") or 0)
        for k in ("basic_meta", "multi_filter_neo4j", "analysis_neo4j", "price", "product_spec")
    )
    kpis = [
        (str(total_q or "—"), "Questions"),
        (f"{summary.get('VALID', 0)}", "VALID"),
        (f"{summary.get('PARTIAL', 0)}", "PARTIAL"),
        (f"{summary.get('INVALID', 0)}", "INVALID"),
        (f"{float(summary.get('pass_rate_valid') or 0):.0f}%", "Pass rate"),
        (f"{float(summary.get('pass_rate_usable') or 0):.0f}%", "Usable"),
    ]
    kpi_row = []
    for v, lbl in kpis:
        t = Table(
            [
                [Paragraph(esc(v), styles["StatBig"])],
                [Paragraph(esc(lbl), styles["StatLabel"])],
            ],
            colWidths=[28 * mm],
        )
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), TEAL_SOFT),
                    ("BOX", (0, 0), (-1, -1), 0.4, TEAL),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ]
            )
        )
        kpi_row.append(t)
    story.append(Table([kpi_row], colWidths=[30 * mm] * 6))
    story.append(Spacer(1, 14))

    story.append(Paragraph("Category scorecard", styles["SectionH"]))
    cat_labels = {
        "basic_meta": "Basic meta / kg_stats",
        "multi_filter_neo4j": "Multi-filter Neo4j",
        "analysis_neo4j": "Analysis",
        "price": "Price",
        "product_spec": "Product specification",
    }
    head = [
        Paragraph("Category", styles["CellHead"]),
        Paragraph("VALID", styles["CellHead"]),
        Paragraph("PARTIAL", styles["CellHead"]),
        Paragraph("INVALID", styles["CellHead"]),
        Paragraph("Score", styles["CellHead"]),
    ]
    rows = [head]
    for key, label in cat_labels.items():
        c = by_cat.get(key) or {}
        total = int(c.get("total") or 0) or 1
        valid = int(c.get("VALID") or 0)
        score = f"{valid}/{total} ({100.0 * valid / total:.0f}%)"
        rows.append(
            [
                Paragraph(esc(label), styles["Cell"]),
                Paragraph(str(c.get("VALID", 0)), styles["Cell"]),
                Paragraph(str(c.get("PARTIAL", 0)), styles["Cell"]),
                Paragraph(str(c.get("INVALID", 0)), styles["Cell"]),
                Paragraph(esc(score), styles["Cell"]),
            ]
        )
    cat_table = Table(rows, colWidths=[70 * mm, 22 * mm, 24 * mm, 24 * mm, 40 * mm])
    cat_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PANEL]),
                ("GRID", (0, 0), (-1, -1), 0.3, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(cat_table)
    story.append(Spacer(1, 12))
    story.append(Paragraph("What this report shows", styles["SectionH"]))
    for b in (
        "Cover page summarises overall VALID / PARTIAL / INVALID performance by category.",
        "Following pages list each question and its answer only (no thinking or references).",
        "Each ask was run live through Fashion Brain against the knowledge graph.",
    ):
        story.append(Paragraph(f"• {esc(b)}", styles["Body"]))
        story.append(Spacer(1, 2))
    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "Fashion Brain · Knowledge Graph R&amp;D · Pattern Benchmark 50",
            styles["Footer"],
        )
    )
    story.append(PageBreak())
    return story


def question_card(styles, r: dict, *, serial: int) -> list:
    """Minimal card: ``N. question`` then answer — no verdict / thinking / refs."""

    question = str(r.get("question") or "").strip() or "—"
    answer = str(r.get("answer") or r.get("response") or "").strip() or "—"

    pieces: list = [
        Paragraph(f"<b>{serial}. {inline_md(question)}</b>", styles["QuestionText"]),
        Spacer(1, 4),
    ]
    pieces.extend(answer_to_flowables(answer, styles, limit=5000))
    pieces.append(Spacer(1, 8))

    rule = Table([[""]], colWidths=[180 * mm], rowHeights=[1])
    rule.setStyle(
        TableStyle(
            [
                ("LINEBELOW", (0, 0), (-1, -1), 0.6, LINE),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    pieces.append(rule)
    pieces.append(Spacer(1, 10))
    return pieces


def add_page_chrome(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(HEADER_BG)
    canvas.rect(0, A4[1] - 12 * mm, A4[0], 12 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont(FONT, 8)
    canvas.drawString(14 * mm, A4[1] - 8 * mm, "Fashion Brain · Pattern Benchmark 50")
    canvas.drawRightString(A4[0] - 14 * mm, A4[1] - 8 * mm, f"Page {doc.page}")
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.4)
    canvas.line(14 * mm, 12 * mm, A4[0] - 14 * mm, 12 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont(FONT, 7)
    canvas.drawCentredString(A4[0] / 2, 7 * mm, "Q&A detail · no per-question verdict")
    canvas.restoreState()


def main() -> None:
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    summary = dict(data.get("summary") or {})
    by_cat = dict(data.get("by_category") or {})
    results = list(data.get("rows") or data.get("results") or [])
    if not summary.get("total") and results:
        summary["total"] = len(results)
    meta = {
        "mode": data.get("mode"),
        "generated_at": data.get("generated_at"),
        "fb_url": data.get("fb_url"),
    }
    styles = build_styles()

    doc = SimpleDocTemplate(
        str(OUT_PATH),
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=18 * mm,
        bottomMargin=16 * mm,
        title="Fashion Brain — Pattern Benchmark 50",
        author="Fashion Brain R&D",
    )
    story: list = []
    story.extend(cover_page(styles, summary, by_cat, meta))
    story.append(Paragraph("Questions & answers", styles["SectionH"]))
    story.append(
        Paragraph(
            "Serial number, question, and answer only.",
            styles["Muted"],
        )
    )
    story.append(Spacer(1, 6))
    for i, r in enumerate(results, start=1):
        story.extend(question_card(styles, r, serial=i))

    doc.build(story, onFirstPage=add_page_chrome, onLaterPages=add_page_chrome)
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size} bytes) font={FONT} rows={len(results)}")


if __name__ == "__main__":
    main()
