"""Convert Markdown files to PDF via reportlab."""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

FONT = "Helvetica"
for font_path in (
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
):
    try:
        pdfmetrics.registerFont(TTFont("AppArial", font_path))
        FONT = "AppArial"
        break
    except Exception:
        continue


def md_inline(s: str) -> str:
    s = html.escape(s or "")
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`([^`]+)`", r'<font face="Courier" size="8">\1</font>', s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", s)
    return s


def parse_table(lines: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in lines:
        if re.match(r"^\s*\|?\s*:?-{2,}", line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    return rows


def md_to_story(md_text: str, styles) -> list:
    story = []
    lines = md_text.splitlines()
    i = 0
    in_code = False
    code_buf: list[str] = []
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            if in_code:
                story.append(
                    Preformatted("\n".join(code_buf)[:4000], styles["CodeBlock"])
                )
                story.append(Spacer(1, 6))
                code_buf = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue
        if line.startswith("|") and i + 1 < len(lines) and re.search(
            r"\|?\s*:?-{2,}", lines[i + 1]
        ):
            tbl_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl_lines.append(lines[i])
                i += 1
            rows = parse_table(tbl_lines)
            if rows:
                data = [
                    [Paragraph(md_inline(c), styles["TableCell"]) for c in r]
                    for r in rows
                ]
                col_n = max(len(r) for r in data)
                for r in data:
                    while len(r) < col_n:
                        r.append(Paragraph("", styles["TableCell"]))
                w = 170 * mm / col_n
                t = Table(data, colWidths=[w] * col_n, repeatRows=1)
                t.setStyle(
                    TableStyle(
                        [
                            (
                                "BACKGROUND",
                                (0, 0),
                                (-1, 0),
                                colors.HexColor("#1f2937"),
                            ),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("FONTNAME", (0, 0), (-1, 0), FONT),
                            ("FONTSIZE", (0, 0), (-1, -1), 7),
                            (
                                "GRID",
                                (0, 0),
                                (-1, -1),
                                0.3,
                                colors.HexColor("#9ca3af"),
                            ),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            (
                                "ROWBACKGROUNDS",
                                (0, 1),
                                (-1, -1),
                                [colors.white, colors.HexColor("#f3f4f6")],
                            ),
                            ("LEFTPADDING", (0, 0), (-1, -1), 3),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                            ("TOPPADDING", (0, 0), (-1, -1), 3),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                        ]
                    )
                )
                story.append(t)
                story.append(Spacer(1, 8))
            continue
        if line.startswith("# "):
            story.append(Paragraph(md_inline(line[2:]), styles["MDH1"]))
            story.append(Spacer(1, 8))
            i += 1
            continue
        if line.startswith("## "):
            story.append(Spacer(1, 6))
            story.append(Paragraph(md_inline(line[3:]), styles["MDH2"]))
            story.append(Spacer(1, 4))
            i += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(md_inline(line[4:]), styles["MDH3"]))
            story.append(Spacer(1, 3))
            i += 1
            continue
        if line.startswith("#### "):
            story.append(Paragraph(md_inline(line[5:]), styles["MDH4"]))
            story.append(Spacer(1, 2))
            i += 1
            continue
        if re.match(r"^-{3,}\s*$", line):
            story.append(Spacer(1, 8))
            i += 1
            continue
        if line.startswith("- ") or line.startswith("* "):
            story.append(
                Paragraph("• " + md_inline(line[2:]), styles["MDBullet"])
            )
            i += 1
            continue
        if re.match(r"^\d+\.\s+", line):
            story.append(Paragraph(md_inline(line), styles["MDBullet"]))
            i += 1
            continue
        if not line.strip():
            story.append(Spacer(1, 4))
            i += 1
            continue
        text = line if len(line) < 2000 else line[:2000] + "…"
        story.append(Paragraph(md_inline(text), styles["MDBody"]))
        i += 1
    return story


def convert(md_path: Path, pdf_path: Path, title: str) -> None:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="MDH1",
            fontName=FONT,
            fontSize=16,
            spaceAfter=6,
            textColor=colors.HexColor("#111827"),
            leading=20,
        )
    )
    styles.add(
        ParagraphStyle(
            name="MDH2",
            fontName=FONT,
            fontSize=13,
            spaceBefore=8,
            textColor=colors.HexColor("#1f2937"),
            leading=16,
        )
    )
    styles.add(
        ParagraphStyle(
            name="MDH3",
            fontName=FONT,
            fontSize=11,
            spaceBefore=6,
            textColor=colors.HexColor("#374151"),
            leading=14,
        )
    )
    styles.add(
        ParagraphStyle(
            name="MDH4",
            fontName=FONT,
            fontSize=10,
            spaceBefore=4,
            textColor=colors.HexColor("#4b5563"),
            leading=13,
        )
    )
    styles.add(
        ParagraphStyle(name="MDBody", fontName=FONT, fontSize=9, leading=12)
    )
    styles.add(
        ParagraphStyle(
            name="MDBullet",
            fontName=FONT,
            fontSize=9,
            leading=12,
            leftIndent=12,
        )
    )
    styles.add(
        ParagraphStyle(name="TableCell", fontName=FONT, fontSize=7, leading=9)
    )
    styles.add(
        ParagraphStyle(
            name="CodeBlock",
            fontName="Courier",
            fontSize=7,
            leading=9,
            backColor=colors.HexColor("#f3f4f6"),
        )
    )
    md = md_path.read_text(encoding="utf-8")
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title=title,
    )
    story = [Paragraph(html.escape(title), styles["MDH1"]), Spacer(1, 10)]
    story.extend(md_to_story(md, styles))
    doc.build(story)
    print(f"wrote {pdf_path} ({pdf_path.stat().st_size} bytes)")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    convert(
        root / "knowledge_graph" / "README_RELATIONSHIP.md",
        root / "knowledge_graph" / "README_RELATIONSHIP.pdf",
        "Knowledge Graph — Relationships & User Columns",
    )
    convert(
        root / "data" / "generated" / "eval" / "pattern_benchmark_50_results.md",
        root / "data" / "generated" / "eval" / "pattern_benchmark_50_results.pdf",
        "Fashion Brain — Pattern Benchmark 50 Results",
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
