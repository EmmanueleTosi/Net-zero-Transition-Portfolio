# Net-Zero Transition Portfolio

A three-agent Python pipeline for ESG-screened portfolio construction, rolling backtesting, performance analysis, and single-page PDF factsheet generation.

Developed as part of a European Pension Fund — Decarbonisation & Energy Transition project.

---

## Project Overview

Two portfolios are constructed from a universe of 40 developed-market companies pre-screened for ESG quality, net-zero alignment, and energy transition relevance:

- **Markowitz Portfolio** — Ledoit-Wolf mean-variance optimisation with concentration constraints (21 holdings, max 10% per issuer, min 0.5%)
- **Risk Parity Portfolio** — Equal Risk Contribution approach, all 40 companies, no concentration constraints

Both portfolios are benchmarked against the **MSCI EAFE Index** (proxied by the iShares EFA ETF) and compared against the full LSEG ESG universe (~14,500 companies) on sustainability metrics.

---

## Pipeline

```
Input dataset (40 companies):
  - Monthly price history (yFinance)
  - Company metadata (name, ticker, country, BICS sectors)
  - LSEG ESG metrics (11 indicators per company)
        │
        ▼
portfolio_construction_agent.py
        │
        ├── portfolio_results.xlsx
        ├── monthly_returns.xlsx
        └── backtest_results.csv
                │
                ▼
portfolio_analysis_agent.py  ◄──  LSEG ESG universe (~14,500 companies)
        │                        (ESG scores, carbon, biodiversity, water)
        │
        │   Phase 1 — Financial Analysis
        │   Phase 2 — Climate & Biodiversity Analysis
        │   Phase 3 — Benchmarking
        │
        └── portfolio_analysis_results.xlsx
                            │
                            ▼
                    reporting_agent.py
                            │
                            ├── Markowitz_Factsheet.pdf
                            └── Risk_Parity_Factsheet.pdf
```

---

## Scripts

### `portfolio_construction_agent.py`

Reads the input dataset containing monthly price history, company metadata and LSEG ESG metrics for 40 pre-screened developed-market companies, and constructs both portfolios.

- Estimates expected returns and covariance matrix (Ledoit-Wolf shrinkage)
- Solves Markowitz mean-variance optimisation via CLARABEL solver
- Solves Risk Parity (Equal Risk Contribution) via log-barrier formulation
- Runs a rolling walk-forward backtest (5 windows, 48-month train / 12-month test)
- Downloads MSCI EAFE benchmark prices via yFinance (ticker: EFA, fallback: IEFA)

**Outputs:** `portfolio_results.xlsx`, `monthly_returns.xlsx`, `backtest_results.csv`

---

### `portfolio_analysis_agent.py`
Reads the optimisation outputs and computes all metrics reported in the factsheets across three sequential internal phases:

**Phase 1 — Financial Analysis**
Computes all financial metrics for both portfolios and the benchmark: total return, annualised return, volatility, Sharpe ratio (rf=2.5%), maximum drawdown, beta vs MSCI EAFE, and 1/3/5-year annualised returns. Also produces calendar year performance for 2021–2025.

**Phase 2 — Climate & Biodiversity Analysis**
Computes portfolio-level ESG indicators as weighted averages across holdings: LSEG ESG Total Score and E/S/G pillar scores, CO₂ intensity (WACI-equivalent), biodiversity due diligence rate, water pollutant emissions intensity, and 3-year and 5-year CAGR in GHG emissions intensity (Scope 1+2+3). All metrics are computed for both the Markowitz and Risk Parity portfolios.

**Phase 3 — Benchmarking**
Compares the portfolios against two reference points:
- **MSCI EAFE Index** (EFA ETF) for financial benchmarking — return, risk, and Sharpe ratio comparison
- **LSEG universe (~14,500 companies)** for ESG, climate and biodiversity benchmarking — portfolio weighted averages vs universe equal-weight averages across all sustainability metrics

**Output:** `portfolio_analysis_results.xlsx` (8 sheets: Financial_Metrics, Calendar_Returns, Monthly_Returns, ESG_Metrics, MKW_Holdings, RP_Holdings, Sector_BICS3, Metadata)

---

### `reporting_agent.py`
Reads `portfolio_analysis_results.xlsx` and generates two single-page A4 PDF factsheets.

- Performance chart vs MSCI EAFE benchmark
- Key Facts and Performance Summary tables
- Calendar Year and Annualised Performance tables
- Top 10 Holdings with BICS Level 3 sector classification
- Sector Allocation at BICS Level 3
- ESG Metrics vs Universe (full-width table)
- Limitations section

**Outputs:** `Markowitz_Factsheet.pdf`, `Risk_Parity_Factsheet.pdf`

---

## Requirements

```
pip install pandas numpy scipy cvxpy clarabel scikit-learn yfinance openpyxl reportlab matplotlib
```

---

## Run Order

```bash
python portfolio_construction_agent.py
python portfolio_analysis_agent.py
python reporting_agent.py
```

Each script depends only on the outputs of the previous one. To update the factsheet layout only, re-run `reporting_agent.py` alone.

---

## Data Sources

| Data | Source |
|---|---|
| Company prices & financial data | yFinance |
| ESG metrics (portfolio companies) | LSEG |
| ESG universe (~14,500 companies) | LSEG |
| Benchmark prices | yFinance (EFA ETF) |

---

## Results

See `Markowitz_Factsheet.pdf` and `Risk_Parity_Factsheet.pdf` for the full results including financial performance, ESG comparison, sector allocation and holdings.
