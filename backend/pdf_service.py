"""PDF report generation (backend-produced, spec section 16).

Professional layout consistent with the web UI (emerald/slate institutional
theme). Only includes metrics that were actually computed -- no fabricated
values. Charts are rendered with matplotlib and embedded as images.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (HRFlowable, Image, KeepTogether, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from . import config

# --- theme -----------------------------------------------------------------
EMERALD = colors.HexColor("#059669")
EMERALD_DK = colors.HexColor("#047857")
INK = colors.HexColor("#0f172a")
SLATE = colors.HexColor("#334155")
MUTED = colors.HexColor("#64748b")
BORDER = colors.HexColor("#e2e8f0")
BG_SOFT = colors.HexColor("#f8fafc")
ROSE = colors.HexColor("#e11d48")
BLUE = colors.HexColor("#2563eb")

_ss = getSampleStyleSheet()
_TITLE = ParagraphStyle("t", parent=_ss["Title"], fontName="Helvetica-Bold",
                        fontSize=18, textColor=INK, alignment=TA_LEFT, spaceAfter=2)
_H = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=11, textColor=EMERALD_DK,
                    spaceBefore=8, spaceAfter=3)
_BODY = ParagraphStyle("b", fontName="Helvetica", fontSize=9, textColor=SLATE,
                       leading=13, alignment=TA_LEFT)
_SMALL = ParagraphStyle("s", fontName="Helvetica", fontSize=7.5, textColor=MUTED, leading=10)
_CELL = ParagraphStyle("c", fontName="Helvetica", fontSize=8.5, textColor=INK, leading=11)
_CELLB = ParagraphStyle("cb", fontName="Helvetica-Bold", fontSize=8.5, textColor=INK, leading=11)


def _rp(v) -> str:
    return "-" if v is None else f"Rp {v:,.0f}"


def _make_charts(report: dict, report_id: str) -> List[Tuple[str, str]]:
    chart = report.get("chart") or {}
    dates = chart.get("dates") or []
    close = chart.get("close")
    if not dates or not close:
        return []
    d = pd.to_datetime(pd.Series(dates))
    fig, ax = plt.subplots(figsize=(7.2, 2.5), dpi=150)
    ax.plot(d, close, color="#059669", lw=1.6, label="Daily close")
    ax.fill_between(d, close, color="#059669", alpha=0.06)
    ax.set_title(f"{report['stock']} daily close over the report period",
                 fontsize=9, color="#0f172a")
    ax.tick_params(labelsize=7, colors="#64748b")
    ax.grid(True, color="#e2e8f0", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    p1 = config.CHARTS_DIR / f"{report_id}_price.png"
    fig.savefig(p1); plt.close(fig)
    return [(str(p1), "Daily close price over the report period")]


def _kv_table(rows: List[Tuple[str, str]], colw=(48 * mm, 32 * mm)) -> Table:
    data = [[Paragraph(f"<b>{k}</b>", _CELL), Paragraph(v, _CELL)] for k, v in rows]
    t = Table(data, colWidths=colw, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("BACKGROUND", (0, 0), (0, -1), BG_SOFT),
    ]))
    return t


def _metrics_table(headers: List[str], rows: List[List[str]]) -> Table:
    data = [[Paragraph(f"<b>{h}</b>", _CELLB) for h in headers]]
    for r in rows:
        data.append([Paragraph(str(c), _CELL) for c in r])
    t = Table(data, colWidths=[34 * mm] + [26 * mm] * (len(headers) - 1), hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BG_SOFT),
        ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def generate_pdf(report: dict, report_id: str, qwen_summary: Optional[str] = None) -> str:
    """Render the report payload to a PDF; returns the absolute file path."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"IDX_{report['stock']}_{report['report_type']}_{report_id}.pdf"
    path = config.REPORTS_DIR / filename

    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm,
                            title=f"IDX Intelligence - {report['stock']} {report['report_type']} report",
                            author="IDX Intelligence Platform")
    story = []
    per = report.get("period", {})
    stats = report.get("statistics", {})

    # Header
    story.append(Paragraph("IDX INTELLIGENCE", ParagraphStyle(
        "brand", fontName="Helvetica-Bold", fontSize=8, textColor=EMERALD, spaceAfter=2)))
    story.append(Paragraph(
        f"{report['stock']} &mdash; {report.get('stock_name','')} "
        f"<font size=11 color='#64748b'>| {per.get('label','')}</font>", _TITLE))
    badge = " <font color='#0284c7'>[HISTORICAL REPLAY]</font>" if report.get("replay") else ""
    story.append(Paragraph(
        f"Period {per.get('start')} to {per.get('end')}{badge} &nbsp;&bull;&nbsp; "
        f"Generated {report.get('generated_at','')} &nbsp;&bull;&nbsp; "
        f"Latest dataset date {report.get('latest_dataset_date','')}", _SMALL))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1, color=EMERALD, spaceAfter=6))

    # Summary / AI interpretation
    story.append(Paragraph("Summary", _H))
    summary = qwen_summary or report.get("summary_text", "")
    if summary:
        story.append(Paragraph(summary, _BODY))
        if qwen_summary:
            story.append(Paragraph("Interpretation generated by the Qwen AI agent from tool results.", _SMALL))
    story.append(Spacer(1, 4))

    # Statistics
    if stats.get("available"):
        story.append(Paragraph("Period statistics", _H))
        chg = stats.get("change_pct")
        chg_col = "#059669" if (chg or 0) >= 0 else "#e11d48"
        story.append(_kv_table([
            ("Trading days", str(stats.get("trading_days"))),
            ("Start price", _rp(stats.get("start_price"))),
            ("End price", _rp(stats.get("end_price"))),
            ("Change", f"<font color='{chg_col}'>{_rp(stats.get('change'))} "
                       f"({chg:+.2f}%)</font>" if chg is not None else "-"),
            ("Period high", f"{_rp(stats.get('high'))} <font color='#64748b'>({stats.get('high_date')})</font>"),
            ("Period low", f"{_rp(stats.get('low'))} <font color='#64748b'>({stats.get('low_date')})</font>"),
            ("Mean / median", f"{_rp(stats.get('mean'))} / {_rp(stats.get('median'))}"),
            ("Volatility (σ)", f"{stats.get('volatility_std')}"),
            ("Avg volume", f"{stats.get('avg_volume'):,.0f}" if stats.get("avg_volume") else "-"),
        ]))
    else:
        story.append(Paragraph("Period statistics", _H))
        story.append(Paragraph(stats.get("reason", "No data in period."), _BODY))

    # Charts
    charts = _make_charts(report, report_id)
    for cpath, caption in charts:
        try:
            img = Image(cpath)
            iw, ih = img.imageWidth, img.imageHeight
            maxw = 178 * mm
            scale = maxw / iw
            img.drawWidth, img.drawHeight = maxw, ih * scale
            story.append(Spacer(1, 6))
            story.append(KeepTogether([img, Paragraph(caption, _SMALL)]))
        except Exception:
            continue

    # Notable movements
    notable = report.get("notable_movements") or []
    if notable:
        story.append(Paragraph("Notable movements", _H))
        rows = [[n["date"], _rp(n["close"]),
                 f"<font color='{'#059669' if n['change_pct']>=0 else '#e11d48'}'>"
                 f"{n['change_pct']:+.2f}%</font>"] for n in notable]
        story.append(_metrics_table(["Date", "Close", "Change %"], rows))

    # Disclaimers
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=4))
    for dline in report.get("disclaimers", []):
        story.append(Paragraph(f"&bull; {dline}", _SMALL))

    doc.build(story)
    return str(path)
