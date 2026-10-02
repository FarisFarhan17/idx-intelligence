# IDX Intelligence

A renewable-energy (EBT) stock **close-price forecasting terminal** for the Indonesia Stock
Exchange (IDX), built on top of a set of pre-trained **FCM-LSTM** models. It adds a
**tool-calling Qwen AI agent**, period **reporting with PDF export and SMTP email delivery**,
a persistent **report history**, and a leak-free **Historical Replay** demo mode.

> Historical dataset only (not a live IDX feed). For education/portfolio purposes, not
> investment advice.

---

## Features

- **Quantitative terminal** — 9 renewable-energy tickers, live KPIs, interactive close-price
  chart with the model's next-day (T+1) forecast overlay, keyboard-accessible watchlist.
- **Next-day close prediction** — reuses the supplied pre-trained FCM-LSTM models
  (never retrained) with the exact inference pipeline recovered from the reference notebook.
- **Honest accuracy reporting** — every prediction is compared against real actuals;
  out-of-sample metrics (MAE / RMSE / MAPE, within-±2% hit rate) are computed, not claimed.
- **Qwen AI agent** — a real tool-calling agent (12 whitelisted backend tools: data,
  prediction, prediction-vs-actual, performance, reports, PDF, email, history). It retrieves
  and computes; it never invents numbers and has no shell or filesystem access.
- **Reporting engine** — daily / weekly / monthly / yearly / custom ranges through one shared
  pipeline used by the dashboard, the agent, manual generation and replay.
- **PDF export** — backend-rendered branded reports (chart + statistics + narrative).
- **Email delivery** — generate → PDF → attach → send over SMTP, with per-report status.
- **Report history** — persistent index of every report with PDF link and email status.
- **Historical Replay** — pick any past date as the analysis cutoff; predictions use only
  information available at that cutoff (no data leakage), later actuals are used only to evaluate.

## Machine-learning pipeline

The models are **pre-trained artifacts** (`ml/*.keras` + `ml/*_params.npz`); this project
performs inference only. The recovered contract:

```
sc      = (close - lo) / (hi - lo)                  # scaler bounds from *_params.npz
ret     = [0, diff(close)/close[:-1]]
fcm_in  = [sc, clip(ret / (4*r_sd), -1, 1)]         # 2-D Fuzzy C-Means input
U       = fcm_membership(fcm_in, centers, m=2)      # 3 fuzzy memberships
F       = [sc, U0, U1, U2]                          # model input tensor [batch, 20, 4]
out     = model.predict(F[-20:])                    # normalized next-day change
pred    = close[-1] + out * d_sd * (hi - lo)        # predicted NEXT-DAY close (T+1)
```

`FCM_LSTM_Close.ipynb` is kept as the reference notebook documenting this recipe.

### Validation (reproducible)

| Check | Scope | Result |
| --- | --- | --- |
| Reproduce notebook test metrics | EBT-Update2 held-out test | MAPE 0.67 – 4.25 % |
| Leak-free prediction vs actual | EBT-Update3, full history | MAPE 0.94 – 3.52 % |
| Out-of-sample (post-training) | PGEO, after 2025-09-12 | MAPE 1.92 %, 63 % within ±2 % |
| Historical replay check | as-of 2025-06-30 → 2025-07-01 | pred 1,414.8 vs actual 1,385 (2.15 % APE) |

Run `python validate_ml.py` to reproduce the model-side checks.

## Architecture

```
Frontend (Tailwind + Chart.js, single shared design system)
        |
Flask REST API  (/api/stocks, /api/reports, /api/chat, ...)
        |
Shared analysis services  <-- one implementation, many consumers
  ├── data_service      read-only EBT workbooks (Update2 / Update3)
  ├── ml_service        FCM + LSTM inference (saved models, cached, never retrained)
  ├── analysis_service  prediction, prediction-vs-actual, performance, periods
  ├── report_service    daily/weekly/monthly/yearly/custom pipeline
  ├── report_manager    build once -> cache -> PDF -> email -> history
  ├── pdf_service       ReportLab + matplotlib
  ├── email_service     SMTP delivery (optional login; supports no-auth relays)
  └── qwen_agent        tool-calling loop over backend.tools (no shell/fs access)
```

## Quick start

Requirements: **Python 3.11** (TensorFlow 2.21 / Keras 3.15 to match the saved models).

```bash
pip install -r requirements.txt
cp .env.example .env      # optional: add QWEN_API_KEY / SMTP_* to enable agent + email
python run.py             # http://127.0.0.1:5000
```

Without `.env` everything still works except the Qwen agent and email delivery, which
degrade gracefully.

### Configuration (`.env`, never committed)

| Variable | Purpose |
| --- | --- |
| `QWEN_API_KEY` | DashScope-compatible key for the agent |
| `QWEN_MODEL`, `QWEN_BASE_URL` | agent model + OpenAI-compatible endpoint |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USE_TLS` | mail server |
| `SMTP_USERNAME`, `SMTP_PASSWORD` | optional login (omit for no-auth relays) |
| `SMTP_FROM_EMAIL` | sender address |

## Repository layout

```
backend/        Flask app + shared service layer
frontend/       single-page terminal UI (index.html + assets/)
data/           EBT historical workbooks (Update2 = training basis, Update3 = later actuals)
ml/             pre-trained FCM-LSTM models + FCM/scaler parameters
FCM_LSTM_Close.ipynb   reference training/inference notebook
validate_ml.py  reproducible model validation script
run.py          entrypoint
```

Runtime artifacts (`storage/` reports, charts, history, and logs) are generated on demand
and are gitignored.

## Security notes

- Secrets live only in `.env` (gitignored); the browser never sees them.
- The agent executes a fixed whitelist of backend tools; it cannot run shell commands or
  read arbitrary files.
- Report narrative for PDF/email is assembled strictly from computed values (no fabrication).

## Disclaimer

Data: provided historical EBT workbooks, latest observation 2025-09-30 / 2026-09-30 per file.
Models: pre-trained artifacts reused without retraining. This project is a technical
portfolio piece and is **not** investment advice.
