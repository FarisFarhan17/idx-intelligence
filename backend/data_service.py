"""Data service: read-only access to the EBT workbooks.

Design notes
------------
* The original datasets are NEVER modified (Historical Replay is a read-only
  analysis context implemented by slicing, not by editing files).
* Update2 is the model's training basis; Update3 is the extended series that
  contains the later *actual* observations used for prediction-vs-actual.
* Parsed sheets are cached in-process so the dashboard, agent, reports and
  scheduler all reuse the same loaded frames (spec: do not reload per request).
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config

# --- Static reference metadata --------------------------------------------
# Company names are read from the data (StockName). Energy type / study cluster
# come from the project's own "Stats Description" sheet (inspected, not invented).
_STOCK_META: Dict[str, Dict[str, str]] = {
    "APEX": {"energy_type": "Geothermal", "cluster": "GT 1"},
    "PGEO": {"energy_type": "Geothermal", "cluster": "GT 2"},
    "ARKO": {"energy_type": "Hydropower", "cluster": "HY 1"},
    "MPOW": {"energy_type": "Hydropower", "cluster": "HY 2"},
    "TGRA": {"energy_type": "Hydropower", "cluster": "HY 3"},
    "DGIK": {"energy_type": "Hydropower", "cluster": "HY 4"},
    "POWR": {"energy_type": "Solar Panel", "cluster": "SP 1"},
    "PTBA": {"energy_type": "Solar Panel", "cluster": "SP 2"},
    "JARR": {"energy_type": "Biomass", "cluster": "BM 1"},
}

_OHLC_COLS = [
    "Date", "OpenPrice", "High", "Low", "Close", "Change",
    "Volume", "Value", "Frequency", "ForeignSell", "ForeignBuy", "StockName",
]

# --- In-process caches -----------------------------------------------------
_SHEET_CACHE: Dict[Tuple[str, str], pd.DataFrame] = {}
_NAME_CACHE: Dict[str, str] = {}


class StockNotFoundError(KeyError):
    """Raised when an unsupported ticker is requested."""


def _validate_stock(stock_code: str) -> str:
    code = str(stock_code).strip().upper()
    if code not in config.STOCKS:
        raise StockNotFoundError(
            f"Unknown stock '{stock_code}'. Supported: {', '.join(config.STOCKS)}"
        )
    return code


def _workbook(source: str):
    src = (source or "update3").lower()
    if src in ("update2", "u2", "2"):
        return config.UPDATE2_XLSX, "update2"
    if src in ("update3", "u3", "3"):
        return config.UPDATE3_XLSX, "update3"
    raise ValueError(f"Unknown data source '{source}' (use 'update2' or 'update3')")


def get_sheet(stock_code: str, source: str = "update3") -> pd.DataFrame:
    """Return the parsed, date-ascending OHLC frame for a stock (cached)."""
    code = _validate_stock(stock_code)
    path, src = _workbook(source)
    key = (code, src)
    if key in _SHEET_CACHE:
        return _SHEET_CACHE[key]

    usecols = [c for c in _OHLC_COLS]
    df = pd.read_excel(path, sheet_name=code)
    # Keep only the columns that actually exist.
    df = df[[c for c in usecols if c in df.columns]].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    for c in ["OpenPrice", "High", "Low", "Close", "Change", "Volume", "Value",
              "Frequency", "ForeignSell", "ForeignBuy"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    _SHEET_CACHE[key] = df
    if "StockName" in df.columns and code not in _NAME_CACHE:
        names = df["StockName"].dropna().unique()
        if len(names):
            _NAME_CACHE[code] = str(names[0]).strip()
    return df


def get_stock_name(stock_code: str) -> str:
    code = _validate_stock(stock_code)
    if code in _NAME_CACHE:
        return _NAME_CACHE[code]
    try:
        get_sheet(code, "update3")
    except Exception:
        pass
    return _NAME_CACHE.get(code, code)


def get_available_stocks() -> List[dict]:
    """Stock universe with lightweight metadata + coverage summary."""
    out: List[dict] = []
    for code in config.STOCKS:
        meta = _STOCK_META.get(code, {})
        rec = {
            "code": code,
            "name": get_stock_name(code),
            "energy_type": meta.get("energy_type", "Renewable Energy"),
            "cluster": meta.get("cluster", ""),
            "model_available": (config.ML_DIR / f"FCM_LSTM_{code}.keras").exists(),
        }
        try:
            df = get_sheet(code, "update3")
            rec["rows"] = int(len(df))
            rec["first_date"] = _iso(df["Date"].iloc[0])
            rec["last_date"] = _iso(df["Date"].iloc[-1])
            rec["last_close"] = float(df["Close"].iloc[-1]) if pd.notna(df["Close"].iloc[-1]) else None
        except Exception:
            rec.update({"rows": 0, "first_date": None, "last_date": None, "last_close": None})
        out.append(rec)
    return out


def get_close_series(
    stock_code: str,
    source: str = "update3",
    end_date: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (dates[int np.datetime64], close[float64]) ascending.

    If ``end_date`` (ISO 'YYYY-MM-DD') is given, the series is truncated to that
    cutoff *inclusive* -- this is how Historical Replay avoids data leakage.
    """
    df = get_sheet(stock_code, source)
    if end_date:
        cut = pd.to_datetime(end_date)
        df = df[df["Date"] <= cut]
    dates = df["Date"].values
    close = df["Close"].to_numpy(dtype="float64")
    return dates, close


def get_stock_data(
    stock_code: str,
    source: str = "update3",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[dict]:
    """OHLC records (list of dicts) for charts/tables, optionally date-bounded."""
    df = get_sheet(stock_code, source)
    if start_date:
        df = df[df["Date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["Date"] <= pd.to_datetime(end_date)]
    records = []
    for _, r in df.iterrows():
        records.append({
            "date": _iso(r["Date"]),
            "open": _num(r.get("OpenPrice")),
            "high": _num(r.get("High")),
            "low": _num(r.get("Low")),
            "close": _num(r.get("Close")),
            "change": _num(r.get("Change")),
            "volume": _num(r.get("Volume")),
            "value": _num(r.get("Value")),
            "foreign_buy": _num(r.get("ForeignBuy")),
            "foreign_sell": _num(r.get("ForeignSell")),
        })
    return records


def get_latest_data_date(stock_code: Optional[str] = None, source: str = "update3") -> Optional[str]:
    """Latest available dataset date (per stock, or across the universe)."""
    codes = [stock_code] if stock_code else config.STOCKS
    latest = None
    for code in codes:
        try:
            df = get_sheet(code, source)
            d = df["Date"].max()
            if pd.notna(d) and (latest is None or d > latest):
                latest = d
        except Exception:
            continue
    return _iso(latest) if latest is not None else None


def get_data_coverage() -> dict:
    """Coverage summary for both workbooks (used to be transparent about data)."""
    cov = {}
    for label, src in (("update2", "update2"), ("update3", "update3")):
        per = {}
        for code in config.STOCKS:
            try:
                df = get_sheet(code, src)
                per[code] = {"rows": int(len(df)),
                             "first": _iso(df["Date"].min()),
                             "last": _iso(df["Date"].max())}
            except Exception as e:
                per[code] = {"error": str(e)}
        cov[label] = per
    return cov


# --- helpers ---------------------------------------------------------------
def _iso(v) -> Optional[str]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (datetime, pd.Timestamp)):
        return v.date().isoformat()
    if isinstance(v, date):
        return v.isoformat()
    try:
        return pd.to_datetime(v).date().isoformat()
    except Exception:
        return str(v)


def _num(v) -> Optional[float]:
    try:
        if v is None or pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None
