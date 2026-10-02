"""Shared report engine.

One reusable analysis pipeline keyed by (stock, start_date, end_date, report_type)
serves the dashboard, the Qwen agent, manual generation, historical replay and
future scheduling. It composes analysis_service primitives -- it never
re-implements calculation logic -- and every number it emits is derived from the
real dataset or the real model (a rule-based text summary is generated from those
computed values, so reports are complete even when Qwen is not configured).
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import analysis_service, config, data_service

VALID_TYPES = ("daily", "weekly", "monthly", "yearly", "custom")


# ==========================================================================
# Period resolution
# ==========================================================================
def resolve_period(report_type: str, ref_date: Optional[str] = None,
                   start: Optional[str] = None, end: Optional[str] = None,
                   as_of: Optional[str] = None) -> dict:
    """Return {start, end, label, type} for a report period.

    ``ref_date`` defaults to the latest dataset date (or the replay cutoff).
    Calendar periods are capped at ``end`` of available data and, in replay mode,
    at ``as_of`` so nothing beyond the simulated cutoff is included.
    """
    rtype = (report_type or "custom").lower()
    latest = data_service.get_latest_data_date()
    cap = _min_date([d for d in (latest, as_of) if d])

    if rtype == "custom":
        if not start or not end:
            raise ValueError("Custom report requires start and end dates.")
        s, e = _iso(start), _iso(end)
        if s > e:
            raise ValueError("start_date must be on or before end_date.")
        label = f"Custom period {s} to {e}"
        return {"type": "custom", "start": s, "end": _cap(e, cap), "label": label}

    ref = pd.to_datetime(ref_date or cap)
    if rtype == "daily":
        s = e = ref
        label = f"Daily report - {ref.date().isoformat()}"
    elif rtype == "weekly":
        monday = ref - pd.Timedelta(days=ref.weekday())
        s, e = monday, monday + pd.Timedelta(days=6)
        label = f"Weekly report - week of {monday.date().isoformat()}"
    elif rtype == "monthly":
        s = ref.replace(day=1)
        e = s + pd.offsets.MonthEnd(0)
        label = f"Monthly report - {ref.strftime('%B %Y')}"
    elif rtype == "yearly":
        s = ref.replace(month=1, day=1)
        e = ref.replace(month=12, day=31)
        label = f"Yearly report - {ref.year}"
    else:
        raise ValueError(f"Unknown report_type '{report_type}'. Use one of {VALID_TYPES}.")

    return {"type": rtype, "start": _iso(s), "end": _cap(_iso(e), cap), "label": label}


def _iso(d) -> str:
    return pd.to_datetime(d).date().isoformat()


def _min_date(dates):
    ds = [pd.to_datetime(d) for d in dates if d]
    return _iso(min(ds)) if ds else None


def _cap(end_iso: str, cap: Optional[str]) -> str:
    if cap and end_iso > cap:
        return cap
    return end_iso


# ==========================================================================
# Period analytics
# ==========================================================================
def get_period_statistics(stock: str, start: str, end: str,
                          as_of: Optional[str] = None) -> dict:
    rows = analysis_service.get_historical_data(stock, start, end, as_of)
    rows = [r for r in rows if r["close"] is not None]
    if not rows:
        return {"available": False, "reason": f"No trading data in {start}..{end}.",
                "trading_days": 0}
    close = np.array([r["close"] for r in rows], float)
    dates = [r["date"] for r in rows]
    vol = np.array([r["volume"] or 0 for r in rows], float)
    val = np.array([r["value"] or 0 for r in rows], float)
    daily_pct = np.r_[0.0, np.diff(close) / close[:-1] * 100]

    start_price, end_price = float(close[0]), float(close[-1])
    change = end_price - start_price
    change_pct = (change / start_price * 100) if start_price else None
    hi_i, lo_i = int(np.argmax(close)), int(np.argmin(close))
    if len(close) > 1:
        dp = np.diff(close) / close[:-1] * 100   # dp[k] = % change on day k+1
        best_i, worst_i = int(np.argmax(dp)) + 1, int(np.argmin(dp)) + 1
        best_pct, worst_pct = float(dp[best_i - 1]), float(dp[worst_i - 1])
    else:
        best_i = worst_i = 0
        best_pct = worst_pct = 0.0

    return {
        "available": True,
        "trading_days": len(rows),
        "actual_start": dates[0],
        "actual_end": dates[-1],
        "start_price": round(start_price, 2),
        "end_price": round(end_price, 2),
        "change": round(change, 2),
        "change_pct": round(change_pct, 3) if change_pct is not None else None,
        "high": round(float(close.max()), 2),
        "high_date": dates[hi_i],
        "low": round(float(close.min()), 2),
        "low_date": dates[lo_i],
        "mean": round(float(close.mean()), 2),
        "median": round(float(np.median(close)), 2),
        "volatility_std": round(float(close.std()), 3),
        "volatility_pct": round(float(np.std(daily_pct[1:])), 3) if len(close) > 1 else None,
        "total_volume": float(vol.sum()),
        "avg_volume": round(float(vol.mean()), 0),
        "total_value": float(val.sum()),
        "best_day": {"date": dates[best_i], "change_pct": round(best_pct, 3)},
        "worst_day": {"date": dates[worst_i], "change_pct": round(worst_pct, 3)},
    }


def get_period_predictions(stock: str, start: str, end: str,
                           as_of: Optional[str] = None) -> dict:
    return analysis_service.get_prediction_vs_actual(stock, start, end, as_of)


def get_period_analysis(stock: str, start: str, end: str,
                        as_of: Optional[str] = None) -> dict:
    stats = get_period_statistics(stock, start, end, as_of)
    pva = get_period_predictions(stock, start, end, as_of)
    return {"statistics": stats, "prediction_vs_actual": pva}


def _notable_movements(stock: str, start: str, end: str,
                       as_of: Optional[str] = None, top: int = 5) -> List[dict]:
    rows = analysis_service.get_historical_data(stock, start, end, as_of)
    rows = [r for r in rows if r["close"] is not None]
    if len(rows) < 2:
        return []
    close = np.array([r["close"] for r in rows], float)
    pct = np.r_[0.0, np.diff(close) / close[:-1] * 100]
    items = [{"date": rows[i]["date"], "close": round(close[i], 2),
              "change_pct": round(float(pct[i]), 3)} for i in range(1, len(rows))]
    items.sort(key=lambda x: abs(x["change_pct"]), reverse=True)
    return items[:top]


def _chart_series(stock: str, start: str, end: str, as_of: Optional[str] = None) -> dict:
    """Aligned close / predicted / actual arrays for charts + PDF."""
    rows = analysis_service.get_historical_data(stock, start, end, as_of)
    rows = [r for r in rows if r["close"] is not None]
    pva = {r["date"]: r for r in analysis_service.get_prediction_vs_actual(
        stock, start, end, as_of).get("series", [])}
    dates = [r["date"] for r in rows]
    close = [r["close"] for r in rows]
    predicted = [pva[d]["predicted"] if d in pva else None for d in dates]
    actual = [pva[d]["actual"] if d in pva else None for d in dates]
    return {"dates": dates, "close": close, "predicted": predicted, "actual": actual}


# ==========================================================================
# Unified report payload
# ==========================================================================
def get_report_data(stock: str, report_type: str = "custom",
                    start: Optional[str] = None, end: Optional[str] = None,
                    ref_date: Optional[str] = None,
                    as_of: Optional[str] = None) -> dict:
    stock = stock.upper()
    if stock not in config.STOCKS:
        raise data_service.StockNotFoundError(f"Unsupported stock '{stock}'.")
    period = resolve_period(report_type, ref_date, start, end, as_of)
    s, e = period["start"], period["end"]

    stats = get_period_statistics(stock, s, e, as_of)
    pva = get_period_predictions(stock, s, e, as_of)
    point = analysis_service.get_prediction(stock, as_of=(as_of or e))
    perf = analysis_service.get_model_performance(stock)
    chart = _chart_series(stock, s, e, as_of)
    notable = _notable_movements(stock, s, e, as_of)

    report = {
        "stock": stock,
        "stock_name": data_service.get_stock_name(stock),
        "report_type": period["type"],
        "period": period,
        "as_of": as_of,
        "replay": bool(as_of),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "latest_dataset_date": data_service.get_latest_data_date(),
        "statistics": stats,
        "prediction": point,
        "prediction_vs_actual": {
            "count": pva["count"], "metrics": pva["metrics"], "range": pva["range"],
        },
        "model_performance": {
            "out_of_sample": perf["out_of_sample"],
            "out_of_sample_boundary": perf["out_of_sample_boundary"],
            "backtest_test": perf["backtest_test"],
        },
        "notable_movements": notable,
        "chart": chart,
        "disclaimers": [
            "Data source: provided EBT workbooks (not a live IDX feed).",
            f"Latest available dataset date: {data_service.get_latest_data_date()}.",
        ],
    }
    report["summary_text"] = build_summary_text(report)
    return report


def build_summary_text(r: dict) -> str:
    """Rule-based narrative assembled strictly from computed values."""
    s = r.get("statistics", {})
    if not s.get("available"):
        return (f"No trading data for {r['stock']} in the selected period "
                f"({r['period']['start']} to {r['period']['end']}).")
    direction = "up" if (s.get("change_pct") or 0) >= 0 else "down"
    parts = [
        f"{r['stock']} ({r.get('stock_name')}) closed the period at "
        f"Rp {s['end_price']:,.0f}, {direction} {abs(s.get('change_pct') or 0):.2f}% "
        f"from Rp {s['start_price']:,.0f} over {s['trading_days']} trading day(s).",
        f"The range was Rp {s['low']:,.0f} ({s['low_date']}) to Rp {s['high']:,.0f} "
        f"({s['high_date']}).",
    ]
    if s.get("best_day") and s.get("worst_day"):
        parts.append(
            f"The strongest session was {s['best_day']['date']} "
            f"({s['best_day']['change_pct']:+.2f}%) and the weakest was "
            f"{s['worst_day']['date']} ({s['worst_day']['change_pct']:+.2f}%).")
    if s.get("avg_volume"):
        parts.append(f"Average daily volume was {s['avg_volume']:,.0f} shares.")
    return " ".join(parts)
