# -*- coding: utf-8 -*-
"""
make_factsheets.py  —  SINGLE PAGE version
─────────────────────────────────────────────────────────────────────────────
Reads Z_comparison_results.xlsx and produces two single-page A4 PDF factsheets:
  1. Markowitz_Factsheet.pdf    (teal)
  2. Risk_Parity_Factsheet.pdf  (navy)

Layout (one page):
  [Header]
  [Description — 1 line]
  [Chart (left) | Key Facts + Performance Summary (right)]
  [Calendar Year table (left) | (continues right)        ]
  [Annualised table (left)    | (continues right)        ]
  [——— divider ———]
  [Top 10 Holdings (left) | Sector BICS-3 + ESG vs Universe (right)]
  [Disclaimer]
"""

import sys, io, warnings
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, Image, HRFlowable)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

W, H = A4

# ── Colours ──────────────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#1F4E79")
BLUE   = colors.HexColor("#2E75B6")
TEAL   = colors.HexColor("#1A7F64")
MGREEN = colors.HexColor("#2E9E78")   # medium green — secondary for Markowitz
LGREY  = colors.HexColor("#F2F2F2")
MGREY  = colors.HexColor("#CCCCCC")
DGREY  = colors.HexColor("#404040")
WHITE  = colors.white
GREEN  = colors.HexColor("#C6EFCE")
LGREEN = colors.HexColor("#EBF3E8")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD Z_comparison_results.xlsx
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  LOADING Z_comparison_results.xlsx")
print("=" * 60)

fin_df  = pd.read_excel("Z_comparison_results.xlsx", sheet_name="Financial_Metrics",  engine="openpyxl")
cal_df  = pd.read_excel("Z_comparison_results.xlsx", sheet_name="Calendar_Returns",   engine="openpyxl")
mon_df  = pd.read_excel("Z_comparison_results.xlsx", sheet_name="Monthly_Returns",    engine="openpyxl",
                         index_col=0, parse_dates=True)
esg_df  = pd.read_excel("Z_comparison_results.xlsx", sheet_name="ESG_Metrics",        engine="openpyxl")
mkw_h   = pd.read_excel("Z_comparison_results.xlsx", sheet_name="MKW_Holdings",       engine="openpyxl")
rp_h    = pd.read_excel("Z_comparison_results.xlsx", sheet_name="RP_Holdings",        engine="openpyxl")
sec_df  = pd.read_excel("Z_comparison_results.xlsx", sheet_name="Sector_BICS3",       engine="openpyxl")
meta_df = pd.read_excel("Z_comparison_results.xlsx", sheet_name="Metadata",           engine="openpyxl")

meta = dict(zip(meta_df["Key"], meta_df["Value"]))

def fin(metric, col):
    row = fin_df[fin_df["Metric"] == metric]
    return float(row[col].values[0]) if len(row) else float("nan")

def cal(year, col):
    row = cal_df[cal_df["Year"] == year]
    return float(row[col].values[0]) if len(row) else float("nan")

def esg(metric, col):
    row = esg_df[esg_df["Metric"] == metric]
    return float(row[col].values[0]) if len(row) else float("nan")

years = [2021, 2022, 2023, 2024, 2025]

def fmt_pct(v, signed=True):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "N/A"
    return f"{v:+.1%}" if signed else f"{v:.1%}"

def fmt_f(v, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "N/A"
    return f"{v:.{d}f}"

print(f"  Loaded: {len(fin_df)} financial metrics, {len(esg_df)} ESG metrics")
print(f"  Holdings: MKW={len(mkw_h)}, RP={len(rp_h)}  |  BICS-3 sectors: {len(sec_df)}")

# ─────────────────────────────────────────────────────────────────────────────
# BUILD CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating charts...")

def make_chart(port_col, line_color, port_label):
    df    = mon_df[[port_col, "MSCI_EAFE"]].dropna()
    cum_p = (1 + df[port_col]).cumprod()    * 100
    cum_b = (1 + df["MSCI_EAFE"]).cumprod() * 100
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(cum_p.index, cum_p.values, color=line_color, lw=2.0, label=port_label)
    ax.plot(cum_b.index, cum_b.values, color="#888888",  lw=1.5,
            linestyle="--", label="MSCI EAFE")
    ax.axhline(100, color="black", lw=0.4, linestyle=":", alpha=0.5)
    ax.set_ylabel("Growth of €100", fontsize=7)
    ax.legend(fontsize=7, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=20)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f"))
    ax.tick_params(labelsize=6.5)
    plt.tight_layout(pad=0.4)
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf

mkw_chart = make_chart("Markowitz",  "#1A7F64", "Net-Zero Transition Portfolio")
rp_chart  = make_chart("RiskParity", "#2E75B6", "Net-Zero Risk Parity Portfolio")
print("  Charts ready.")


# ─────────────────────────────────────────────────────────────────────────────
# TABLE STYLE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def ts_base(hdr_color, pad=2):
    return TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),  hdr_color),
        ("TEXTCOLOR",     (0,0),(-1,0),  WHITE),
        ("FONTNAME",      (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 7.5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, LGREY]),
        ("GRID",          (0,0),(-1,-1), 0.3, MGREY),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0),(-1,-1), pad),
        ("BOTTOMPADDING", (0,0),(-1,-1), pad),
        ("LEFTPADDING",   (0,0),(-1,-1), 4),
        ("RIGHTPADDING",  (0,0),(-1,-1), 4),
    ])

def no_pad_style():
    return TableStyle([
        ("TOPPADDING",    (0,0),(-1,-1), 0),
        ("BOTTOMPADDING", (0,0),(-1,-1), 0),
        ("LEFTPADDING",   (0,0),(-1,-1), 0),
        ("RIGHTPADDING",  (0,0),(-1,-1), 0),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# PDF BUILDER  —  single page
# ─────────────────────────────────────────────────────────────────────────────

def build_pdf(output_file, *,
              primary_color, secondary_color, title_text, description_text,
              chart_buf, port_col,
              holdings_df, weights_col, esg_port_col,
              cagr_footnote, kf_extra_rows, kf_col_widths):

    pfx = output_file[:3]

    def S(name, **kw):
        d = dict(fontName="Helvetica", fontSize=8, textColor=DGREY, leading=10, spaceAfter=1)
        d.update(kw)
        return ParagraphStyle(f"{pfx}_{name}", **d)

    sTitle  = S("Title", fontName="Helvetica-Bold", fontSize=15, textColor=WHITE, leading=19)
    sSub    = S("Sub",   fontSize=8.5, textColor=WHITE, leading=11)
    sHdr    = S("Hdr",  fontName="Helvetica-Bold", fontSize=8, textColor=NAVY,
                leading=10, spaceBefore=2, spaceAfter=1)
    sBody   = S("Body", fontSize=7.5, leading=10)
    sSm     = S("Sm",   fontSize=6.5, leading=9,  textColor=colors.HexColor("#555555"))
    sBold   = S("Bold",  fontName="Helvetica-Bold", fontSize=7.5, leading=10)
    sBoldW  = S("BoldW", fontName="Helvetica-Bold", fontSize=7.5, leading=10, textColor=WHITE)

    doc = SimpleDocTemplate(
        output_file, pagesize=A4,
        leftMargin=1.4*cm, rightMargin=1.4*cm,
        topMargin=0.2*cm,  bottomMargin=0.9*cm)
    story = []

    # ── Header ───────────────────────────────────────────────────────────
    hdr = Table([[Paragraph(title_text, sTitle),
                  Paragraph("Factsheet — April 30, 2026", sSub)]],
                colWidths=[13*cm, 5.2*cm])
    hdr.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), primary_color),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0),(-1,-1), 8),
        ("BOTTOMPADDING", (0,0),(-1,-1), 8),
        ("LEFTPADDING",   (0,0),(-1,-1), 10),
        ("RIGHTPADDING",  (0,0),(-1,-1), 10),
        ("ALIGN",         (1,0),(1,0),   "RIGHT"),
    ]))
    story.append(hdr)
    story.append(Spacer(1, 2))
    story.append(Paragraph(description_text, sBody))
    story.append(Spacer(1, 2))

    # ── TOP SECTION: chart (left)  |  key facts + perf summary (right) ──
    LW = 11.0*cm; RW = 7.2*cm

    chart_buf.seek(0)
    chart_img = Image(chart_buf, width=LW, height=4.8*cm)

    # Calendar year table
    cy_hdr = ["", "2021", "2022", "2023", "2024", "2025*"]
    cy_p   = ["Portfolio"] + [fmt_pct(cal(y, port_col))    for y in years]
    cy_b   = ["MSCI EAFE"] + [fmt_pct(cal(y, "MSCI_EAFE")) for y in years]
    cy_tbl = Table([cy_hdr, cy_p, cy_b],
                   colWidths=[3.0*cm, 1.6*cm, 1.6*cm, 1.6*cm, 1.6*cm, 1.6*cm])
    cy_ts  = ts_base(secondary_color, pad=1)
    cy_ts.add("ALIGN",   (1,0), (-1,-1), "CENTER")
    cy_ts.add("FONTNAME",(0,1), (0,-1),  "Helvetica-Bold")
    cy_tbl.setStyle(cy_ts)

    # Annualised table
    an_hdr = ["", "1 Year", "3 Year", "5 Year (BT)"]
    an_p   = ["Portfolio",
              fmt_pct(fin("1yr Annualised Return", port_col)),
              fmt_pct(fin("3yr Annualised Return", port_col)),
              fmt_pct(fin("5yr Annualised Return", port_col))]
    an_b   = ["MSCI EAFE",
              fmt_pct(fin("1yr Annualised Return", "MSCI_EAFE")),
              fmt_pct(fin("3yr Annualised Return", "MSCI_EAFE")),
              fmt_pct(fin("5yr Annualised Return", "MSCI_EAFE"))]
    an_tbl = Table([an_hdr, an_p, an_b],
                   colWidths=[3.2*cm, 2.6*cm, 2.6*cm, 2.6*cm])
    an_ts  = ts_base(secondary_color, pad=1)
    an_ts.add("ALIGN",   (1,0), (-1,-1), "CENTER")
    an_ts.add("FONTNAME",(0,1), (0,-1),  "Helvetica-Bold")
    an_tbl.setStyle(an_ts)

    left_top = Table([
        [chart_img],
        [Paragraph("CALENDAR YEAR PERFORMANCE (%)", sHdr)],
        [cy_tbl],
        [Paragraph("ANNUALISED PERFORMANCE (%)", sHdr)],
        [an_tbl],
    ], colWidths=[LW])
    left_top.setStyle(no_pad_style())

    # Key Facts
    kf_base_rows = [
        [Paragraph("KEY FACTS", sBoldW), ""],
        ["Benchmark",            meta.get("Benchmark", "MSCI EAFE (EFA)")],
        ["Inception (backtest)", "February 2021"],
        ["Rebalancing",          "Annual (rolling)"],
    ]
    kf_rows = kf_base_rows + kf_extra_rows
    kf_tbl  = Table(kf_rows, colWidths=kf_col_widths)
    kf_ts   = ts_base(primary_color, pad=3)
    kf_ts.add("SPAN",     (0,0), (1,0))
    kf_ts.add("FONTNAME", (0,1), (0,-1), "Helvetica-Bold")
    kf_ts.add("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, LGREY])
    kf_tbl.setStyle(kf_ts)

    # Performance Summary
    pf_rows = [
        [Paragraph("PERFORMANCE SUMMARY", sBoldW), "", ""],
        ["Metric",           "Portfolio",  "MSCI EAFE"],
        ["Total Ret (5yr)",  fmt_pct(fin("Total Return (5yr)",    port_col)),   fmt_pct(fin("Total Return (5yr)",    "MSCI_EAFE"))],
        ["Ann. Return",      fmt_pct(fin("Annualised Return",     port_col)),   fmt_pct(fin("Annualised Return",     "MSCI_EAFE"))],
        ["Volatility",       fmt_pct(fin("Annualised Volatility", port_col)),   fmt_pct(fin("Annualised Volatility", "MSCI_EAFE"))],
        ["Sharpe Ratio",     fmt_f(fin("Sharpe Ratio (rf=2.5%)", port_col)),   fmt_f(fin("Sharpe Ratio (rf=2.5%)", "MSCI_EAFE"))],
        ["Max Drawdown",     fmt_pct(fin("Max Drawdown",          port_col)),   fmt_pct(fin("Max Drawdown",          "MSCI_EAFE"))],
        ["Beta",             fmt_f(fin("Beta vs MSCI EAFE",       port_col)),   "1.00"],
    ]
    pf_tbl = Table(pf_rows, colWidths=[3.0*cm, 2.0*cm, 2.0*cm])
    pf_ts  = ts_base(primary_color, pad=3)
    pf_ts.add("SPAN",          (0,0), (2,0))
    pf_ts.add("BACKGROUND",    (0,1), (2,1),  secondary_color)
    pf_ts.add("TEXTCOLOR",     (0,1), (2,1),  WHITE)
    pf_ts.add("FONTNAME",      (0,1), (2,1),  "Helvetica-Bold")
    pf_ts.add("ALIGN",         (1,0), (2,-1), "CENTER")
    pf_ts.add("FONTNAME",      (0,2), (0,-1), "Helvetica-Bold")
    pf_ts.add("BACKGROUND",    (1,2), (1,-1), GREEN)
    pf_tbl.setStyle(pf_ts)

    # Compute spacer so PS table bottom aligns with annualised table bottom.
    # Left column height ≈ chart + 2×(header≈13pt + 3-row table≈33pt + note≈9pt)
    CHART_PT   = 4.8 * 28.35            # chart height in points
    LEFT_TOTAL = CHART_PT + 2 * (13 + 33 + 9)   # ≈ 246 pt
    KF_ROWS    = 4 + len(kf_extra_rows)
    PF_ROWS    = 8
    PT_PER_ROW = 15.0                   # pad=3 → 7.5 font + 3+3 pad ≈ 15 pt/row
    mid_gap = max(4, LEFT_TOTAL - (KF_ROWS + PF_ROWS) * PT_PER_ROW)

    right_top = Table([
        [kf_tbl],
        [Spacer(1, mid_gap)],
        [pf_tbl],
    ], colWidths=[RW])
    right_top.setStyle(no_pad_style())

    top_section = Table([[left_top, right_top]], colWidths=[LW, RW])
    top_section.setStyle(TableStyle([
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING",   (1,0), (1,0),   8),
        ("TOPPADDING",    (0,0), (-1,-1), 0),
        ("BOTTOMPADDING", (0,0), (-1,-1), 0),
        ("LEFTPADDING",   (0,0), (0,0),   0),
        ("RIGHTPADDING",  (0,0), (-1,-1), 0),
    ]))
    story.append(top_section)
    story.append(Spacer(1, 2))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MGREY))
    story.append(Spacer(1, 2))

    # ── BOTTOM SECTION: top 10 (left) | sectors + ESG (right) ────────────
    BL = 9.2*cm; BR = 9.0*cm

    # Top 10 holdings
    story_left = []
    story_left.append(Paragraph("TOP 10 HOLDINGS (%)", sHdr))
    top10 = holdings_df.head(10)
    t10_rows = [["#", "Company", "Sector (BICS-3)", "Wt"]]
    for _, row in top10.iterrows():
        rank = int(row["Rank"]) if "Rank" in row.index else int(_)
        nm   = str(row["company_name"])[:28]
        sec3 = str(row.get("bics_level_3_name",""))[:24]
        wt   = row[weights_col]
        t10_rows.append([str(rank), nm, sec3, f"{wt:.1%}"])
    t10_tbl = Table(t10_rows, colWidths=[0.6*cm, 3.8*cm, 3.4*cm, 1.2*cm])
    t10_ts  = ts_base(primary_color, pad=1)
    t10_ts.add("ALIGN",      (0,0), (0,-1), "CENTER")
    t10_ts.add("ALIGN",      (3,0), (3,-1), "CENTER")
    t10_ts.add("BACKGROUND", (3,1), (3,-1), GREEN)
    t10_tbl.setStyle(t10_ts)
    story_left.append(t10_tbl)

    left_bot = Table([[el] for el in story_left], colWidths=[BL])
    left_bot.setStyle(no_pad_style())

    # Sector allocation BICS-3
    story_right = []
    story_right.append(Paragraph("SECTOR ALLOCATION — BICS LEVEL 3 (%)", sHdr))
    sec_col = "Markowitz" if port_col == "Markowitz" else "Risk_Parity"
    sec_sorted = sec_df[sec_df[sec_col] > 1e-4].sort_values(sec_col, ascending=False)
    MAX_SECTORS = 12
    if len(sec_sorted) > MAX_SECTORS:
        sec_top    = sec_sorted.iloc[:MAX_SECTORS]
        other_wt   = sec_sorted.iloc[MAX_SECTORS:][sec_col].sum()
        sec_rows   = [["Sector (BICS Level 3)", "Wt"]]
        for _, row in sec_top.iterrows():
            sec_rows.append([str(row["BICS_Level_3"])[:38], f"{row[sec_col]:.1%}"])
        sec_rows.append([f"Other sectors ({len(sec_sorted)-MAX_SECTORS})", f"{other_wt:.1%}"])
    else:
        sec_rows = [["Sector (BICS Level 3)", "Wt"]]
        for _, row in sec_sorted.iterrows():
            sec_rows.append([str(row["BICS_Level_3"])[:38], f"{row[sec_col]:.1%}"])
    sec_tbl = Table(sec_rows, colWidths=[7.2*cm, 1.6*cm])
    sec_ts  = ts_base(secondary_color, pad=1)
    sec_ts.add("ALIGN", (1,0), (1,-1), "CENTER")
    sec_tbl.setStyle(sec_ts)
    story_right.append(sec_tbl)

    right_bot = Table([[el] for el in story_right], colWidths=[BR])
    right_bot.setStyle(no_pad_style())

    bot_section = Table([[left_bot, right_bot]], colWidths=[BL, BR])
    bot_section.setStyle(TableStyle([
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING",   (1,0), (1,0),   8),
        ("TOPPADDING",    (0,0), (-1,-1), 0),
        ("BOTTOMPADDING", (0,0), (-1,-1), 0),
        ("LEFTPADDING",   (0,0), (0,0),   0),
        ("RIGHTPADDING",  (0,0), (-1,-1), 0),
    ]))
    story.append(bot_section)

    # ── ESG vs Universe — full width, centred ─────────────────────────────
    story.append(Spacer(1, 2))
    story.append(Paragraph("ESG METRICS vs UNIVERSE (14,486 companies)", sHdr))

    LESG  = "LSEG ESG Score"
    LE    = "LSEG E Score"
    LS    = "LSEG S Score"
    LG    = "LSEG G Score"
    LCO2  = "LSEG Total CO2 Emissions / Million in Revenue $"
    LBIO  = "LSEG Biodiversity Due Diligence"
    LWAT  = "LSEG TOTAL WATER POLLUTANT EMISSIONS / MILLION IN REVENUE $"
    LC3   = "LSEG CAGR 3 YEARS GHG EMISSIONS INTENSITY SCOPE 1 AND SCOPE 2 AND SCOPE 3"

    def _a(p, u, hb=True):
        if np.isnan(p) or np.isnan(u): return "N/A"
        d = (p - u) / abs(u) * 100
        return f"+{d:.0f}% vs uni" if d >= 0 else f"{d:.0f}% vs uni"
    def _bio_a(p, u):
        return f"{p/u:.1f}x uni rate" if (not np.isnan(p) and u > 0) else "N/A"
    def _cagr_a(p, u):
        if np.isnan(p) or np.isnan(u): return "N/A"
        if p < 0 and u > 0: return "Declining vs uni"
        return f"{p:+.1f}% | uni {u:+.1f}%"

    ev = {m: esg(m, esg_port_col)   for m in [LESG,LE,LS,LG,LCO2,LBIO,LWAT,LC3]}
    uv = {m: esg(m, "Universe_Avg") for m in [LESG,LE,LS,LG,LCO2,LBIO,LWAT,LC3]}

    # Full-page width: 18.2 cm usable → 4 columns totalling 18.2 cm
    esg_rows = [
        ["Metric",                     "Portfolio", "Universe", "vs Universe"],
        ["ESG Total Score",            fmt_f(ev[LESG],1),  fmt_f(uv[LESG],1),  _a(ev[LESG],  uv[LESG])],
        ["E Score (Environmental)",    fmt_f(ev[LE],1),    fmt_f(uv[LE],1),    _a(ev[LE],    uv[LE])],
        ["S Score (Social)",           fmt_f(ev[LS],1),    fmt_f(uv[LS],1),    _a(ev[LS],    uv[LS])],
        ["G Score (Governance)",       fmt_f(ev[LG],1),    fmt_f(uv[LG],1),    _a(ev[LG],    uv[LG])],
        ["CO2 Intensity (t/M Rev$)",   fmt_f(ev[LCO2],1),  fmt_f(uv[LCO2],1),  _a(ev[LCO2],  uv[LCO2], False)],
        ["Water Pollutant (t/M Rev$)", fmt_f(ev[LWAT],3),  fmt_f(uv[LWAT],2),  _a(ev[LWAT],  uv[LWAT], False)],
        ["Biodiversity DD (%)",        f"{ev[LBIO]:.1f}%", f"{uv[LBIO]:.1f}%", _bio_a(ev[LBIO], uv[LBIO])],
        [u"3Y CAGR GHG (S1+S2+S3) †", f"{ev[LC3]:+.2f}%", f"{uv[LC3]:+.2f}%", _cagr_a(ev[LC3], uv[LC3])],
    ]
    esg_tbl = Table(esg_rows, colWidths=[9.2*cm, 2.6*cm, 2.6*cm, 3.8*cm],
                    hAlign="CENTER")
    esg_ts  = ts_base(primary_color, pad=1)
    esg_ts.add("ALIGN",      (1,0), (3,-1), "CENTER")
    esg_ts.add("FONTNAME",   (0,1), (0,-1), "Helvetica-Bold")
    esg_ts.add("BACKGROUND", (1,1), (1,-1), GREEN)
    esg_ts.add("BACKGROUND", (3,1), (3,-1), LGREEN)
    esg_tbl.setStyle(esg_ts)
    story.append(esg_tbl)

    # ── Limitations ───────────────────────────────────────────────────────
    story.append(Spacer(1, 2))
    story.append(HRFlowable(width="100%", thickness=0.4, color=MGREY))
    story.append(Spacer(1, 1))
    story.append(Paragraph("LIMITATIONS", sHdr))
    limitations = (
        "<b>Transaction costs:</b> All returns are gross of transaction costs. In practice, portfolio "
        "rebalancing generates brokerage commissions, bid-ask spreads and market-impact costs that would "
        "reduce net performance. &nbsp;&nbsp;"
        "<b>Management fees:</b> No management or administration fees have been deducted. Real-world "
        "vehicles typically charge annual management fees (e.g. 0.3–1.0% p.a.) that compound over time "
        "and materially affect long-run returns."
    )
    story.append(Paragraph(limitations, sSm))

    doc.build(story)
    print(f"  -> {output_file}")


# ─────────────────────────────────────────────────────────────────────────────
# BUILD PDF 1 — MARKOWITZ (teal)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  BUILDING MARKOWITZ FACTSHEET")
print("=" * 60)

build_pdf(
    "Markowitz_Factsheet.pdf",
    primary_color    = TEAL,
    secondary_color  = MGREEN,
    title_text       = "Net-Zero Transition Portfolio — Markowitz",
    description_text = (
        "The Net-Zero Transition Portfolio invests in 21 developed-market companies selected and weighted "
        "using Ledoit-Wolf mean-variance optimisation (Markowitz) to maximise risk-adjusted returns while "
        "supporting alignment with net-zero emission pathways. Holdings are ESG-screened and drawn from the "
        "global energy transition universe. Portfolio constraints enforce a maximum 10% weight per issuer "
        "and a minimum 0.5% allocation. Benchmark: MSCI EAFE Index (EFA ETF)."),
    chart_buf     = mkw_chart,
    port_col      = "Markowitz",
    holdings_df   = mkw_h,
    weights_col   = "Weight_Markowitz",
    esg_port_col  = "Markowitz_Portfolio",
    cagr_footnote = meta.get("MKW_CAGR_Footnote", ""),
    kf_extra_rows = [
        ["Max weight / issuer",     meta.get("MKW_Max_Weight", "10.0%")],
        [u"N° of holdings",         str(meta.get("MKW_N_Holdings", ""))],
        [u"N° of sectors (BICS-3)", str(meta.get("MKW_N_Sectors_BICS3", ""))],
    ],
    kf_col_widths = [3.4*cm, 3.6*cm],
)

# ─────────────────────────────────────────────────────────────────────────────
# BUILD PDF 2 — RISK PARITY (navy)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  BUILDING RISK PARITY FACTSHEET")
print("=" * 60)

build_pdf(
    "Risk_Parity_Factsheet.pdf",
    primary_color    = NAVY,
    secondary_color  = BLUE,
    title_text       = u"Net-Zero Transition Portfolio — Risk Parity",
    description_text = (
        "The Net-Zero Transition Portfolio invests in 40 developed-market companies weighted using a "
        "Risk Parity (Equal Risk Contribution) approach with Ledoit-Wolf shrinkage covariance estimation "
        "to equalise each holding's marginal contribution to total portfolio risk. Capital allocation "
        "relies solely on risk diversification, without dependence on return forecasts. Holdings are "
        "ESG-screened and drawn from the global energy transition universe. Benchmark: MSCI EAFE Index (EFA ETF)."),
    chart_buf     = rp_chart,
    port_col      = "Risk_Parity",
    holdings_df   = rp_h,
    weights_col   = "Weight_RiskParity",
    esg_port_col  = "RiskParity_Portfolio",
    cagr_footnote = meta.get("RP_CAGR_Footnote", ""),
    kf_extra_rows = [
        ["Optimisation",            "Equal Risk Contribution"],
        [u"N° of holdings",         str(meta.get("RP_N_Holdings", ""))],
        [u"N° of sectors (BICS-3)", str(meta.get("RP_N_Sectors_BICS3", ""))],
        ["Max weight / issuer",     meta.get("RP_Max_Weight", "")],
    ],
    kf_col_widths = [3.6*cm, 3.4*cm],
)

print("\n" + "=" * 60)
print("  DONE — both factsheets generated")
print("=" * 60)
