"""Application configuration.

All secrets come from environment variables (loaded from a local .env if present).
Nothing sensitive is ever hard-coded or exposed to the frontend.
"""
from __future__ import annotations

import os
from pathlib import Path

# Load .env if python-dotenv is available (optional dependency at import time).
try:  # pragma: no cover - trivial guard
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # dotenv not installed yet; environment variables still work
    pass

# --- Paths -----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # C:\IDX PORTO
DATA_DIR = BASE_DIR / "data"
ML_DIR = BASE_DIR / "ml"
FRONTEND_DIR = BASE_DIR / "frontend"
STORAGE_DIR = BASE_DIR / "storage"
REPORTS_DIR = STORAGE_DIR / "reports"
CHARTS_DIR = STORAGE_DIR / "charts"
HISTORY_FILE = STORAGE_DIR / "report_history.json"

# Source workbooks (read-only; never modified).
UPDATE2_XLSX = DATA_DIR / "EBT - Update2.xlsx"   # model training basis
UPDATE3_XLSX = DATA_DIR / "EBT - Update3.xlsx"   # extended actuals

for _d in (STORAGE_DIR, REPORTS_DIR, CHARTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Stock universe (matches notebook SHEETS) ------------------------------
STOCKS = ["DGIK", "JARR", "TGRA", "PTBA", "POWR", "PGEO", "MPOW", "ARKO", "APEX"]

# --- ML constants (from FCM_LSTM_Close.ipynb) ------------------------------
WINDOW = 20
N_CLUSTERS = 3
FUZZ_M = 2.0

# --- Qwen agent ------------------------------------------------------------
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "").strip()
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus").strip()
QWEN_BASE_URL = os.getenv(
    "QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
).strip()

# --- SMTP (report email delivery) -----------------------------------------
SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587") or 587)
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL", "").strip()
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "IDX Intelligence").strip()
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").strip().lower() in ("1", "true", "yes")

# --- App -------------------------------------------------------------------
HOST = os.getenv("APP_HOST", "127.0.0.1")
PORT = int(os.getenv("APP_PORT", "5000") or 5000)
DEBUG = os.getenv("APP_DEBUG", "false").strip().lower() in ("1", "true", "yes")


def smtp_configured() -> bool:
    # Host + a from-address are the minimum; username/password are optional so
    # unauthenticated local test relays (Mailpit/MailHog) also work.
    return bool(SMTP_HOST and SMTP_FROM_EMAIL)


def qwen_configured() -> bool:
    return bool(QWEN_API_KEY)
