"""Shared analysis engine.

Every consumer (dashboard, Qwen agent, reports, PDF, email, replay, future
scheduler) calls these functions so the analysis logic exists exactly once.

Leak-free by construction: an ``as_of`` cutoff (ISO date) truncates the input
series before any prediction is made. The *actual* outcome for the day after the
cutoff is looked up separately and only ever used for evaluation, never as model
input (spec sections 19-20).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config, data_service, ml_service

# --- caches (performance: spec section 30) --------------------------------
_PVAA_CACHE: Dict[str, List[dict]] = {}
_SNAP_CACHE: Dict[str, dict] = {}


def _ts(d) -> Optional[pd.Timestamp]:
    if d is None:
        return None
    return pd.to_datetime(d)


def _iso(d) -> Optional[str]:
    t = _ts(d)
    return None if t is None or pd.isna(t) else t.date().isoformat()


def _round(x, n=2):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return round(float(x), n)
    except Exception:
        return None


# ==========================================================================
# Universe / raw data
# ==========================================================================
def get_available_stocks() -> List[dict]:
    return data_service.get_available_stocks()


def get_stock_data(stock: str, as_of: Optional[str] = None) -> dict:
    """Latest snapshot for a stock, honouring the replay cutoff."""
    stock = stock.upper()
    df = data_service.get_sheet(stock, "update3")
    if as_of:
        df = df[df["Date"] <= _ts(as_of)]
    if df.empty:
        return {"stock": stock, "available": False, "reason": "No data up to cutoff."}
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else None
    last_close = float(last["Close"])
    prev_close = float(prev["Close"]) if prev is not None else None
    change = last_close - prev_close if prev_close else None
    change_pct = (change / prev_close * 100) if prev_close else None
    return {
        "stock": stock,
        "name": data_service.get_stock_name(stock),
        "available": True,
        "as_of": _iso(last["Date"]),
        "last_close": _round(last_close, 2),
        "previous_close": _round(prev_close, 2),
        "change": _round(change, 2),
        "change_pct": _round(change_pct, 3),
        "open": _round(last.get("OpenPrice"), 2),
        "high": _round(last.get("High"), 2),
        "low": _round(last.get("Low"), 2),
        "volume": _round(last.get("Volume"), 0),
        "value": _round(last.get("Value"), 0),
        "energy_type": _SNAP_META.get(stock, {}).get("energy_type"),
        "cluster": _SNAP_META.get(stock, {}).get("cluster"),
    }


_SNAP_META = {
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


def get_historical_data(stock: str, start: Optional[str] = None,
                        end: Optional[str] = None, as_of: Optional[str] = None,
                        source: str = "update3") -> List[dict]:
    eff_end = end
    if as_of and (end is None or _ts(as_of) < _ts(end)):
        eff_end = as_of
    return data_service.get_stock_data(stock, source, start, eff_end)


# ==========================================================================
# Prediction (leak-free) & prediction-vs-actual
# ==========================================================================
def get_prediction(stock: str, as_of: Optional[str] = None) -> dict:
    """Forecast the next trading day's close using only data up to ``as_of``.

    If the day after the cutoff already has an actual observation in Update3, it
    is attached for evaluation (this is the replay/demo payoff) -- but it is NEVER
    fed into the model.
    """
    stock = stock.upper()
    full_dates, full_close = data_service.get_close_series(stock, "update3")
    if len(full_close) == 0:
        return {"stock": stock, "available": False, "reason": "No data available."}

    dates = pd.to_datetime(pd.Series(full_dates))
    if as_of:
        mask = (dates <= _ts(as_of)).to_numpy()
    else:
        mask = np.ones(len(full_close), dtype=bool)
    idx_last = int(np.where(mask)[0].max()) if mask.any() else -1
    if idx_last < 0:
        return {"stock": stock, "available": False,
                "reason": f"No data on/before cutoff {as_of}."}

    hist_close = full_close[: idx_last + 1]
    pred = ml_service.predict_latest(stock, hist_close)
    as_of_date = _iso(full_dates[idx_last])

    result = {
        "stock": stock,
        "as_of": as_of_date,
        "as_of_close": _round(float(full_close[idx_last]), 2),
        "available": pred["available"],
        "predicted_close": _round(pred.get("predicted_close"), 2),
        "predicted_change_pct": _round(pred.get("predicted_change_pct"), 3),
        "cluster": pred.get("cluster"),
        "memberships": pred.get("memberships"),
        "membership_max": pred.get("membership_max"),
        "horizon_days": pred.get("horizon_days", 1),
        "reason": pred.get("reason"),
    }
    # Attach the later actual for evaluation if it exists (never used as input).
    nxt = idx_last + 1
    if nxt < len(full_close):
        actual = float(full_close[nxt])
        result["predicted_for_date"] = _iso(full_dates[nxt])
        result["actual_close"] = _round(actual, 2)
        if pred["available"]:
            p = pred["predicted_close"]
            result["error"] = _round(p - actual, 2)
            result["abs_error"] = _round(abs(p - actual), 2)
            result["ape"] = _round(abs(p - actual) / actual * 100, 3) if actual else None
            result["direction_correct"] = bool(
                np.sign(p - float(full_close[idx_last])) == np.sign(actual - float(full_close[idx_last]))
            )
    else:
        result["predicted_for_date"] = None
        result["actual_close"] = None
        result["note"] = ("Forecast is for the next trading day after the cutoff; "
                          "no actual observation exists yet in the dataset.")
    return result


def _pvaa_full(stock: str) -> List[dict]:
    """Cached full prediction-vs-actual series over Update3."""
    if stock in _PVAA_CACHE:
        return _PVAA_CACHE[stock]
    dates, close = data_service.get_close_series(stock, "update3")
    rows = ml_service.predict_vs_actual_series(stock, dates, close)
    _PVAA_CACHE[stock] = rows
    return rows


def _metrics(true, pred) -> dict:
    true = np.asarray(true, float)
    pred = np.asarray(pred, float)
    if len(true) == 0:
        return {"n": 0, "mae": None, "rmse": None, "mape": None}
    err = pred - true
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.where(true != 0, np.abs(err / true) * 100, np.nan)
    return {
        "n": int(len(true)),
        "mae": _round(float(np.mean(np.abs(err))), 4),
        "rmse": _round(float(np.sqrt(np.mean(err ** 2))), 4),
        "mape": _round(float(np.nanmean(ape)), 4),
    }


def get_prediction_vs_actual(stock: str, start: Optional[str] = None,
                             end: Optional[str] = None,
                             as_of: Optional[str] = None,
                             limit: Optional[int] = None) -> dict:
    """Prediction-vs-actual pairs + summary metrics within an optional range.

    In replay mode (``as_of``) only pairs whose target date is <= as_of are used,
    so the summary never reflects information beyond the simulated cutoff.
    """
    stock = stock.upper()
    rows = _pvaa_full(stock)
    eff_end = end
    if as_of and (end is None or _ts(as_of) < _ts(end)):
        eff_end = as_of
    sel = rows
    if start:
        sel = [r for r in sel if r["date"] and r["date"] >= _iso(start)]
    if eff_end:
        sel = [r for r in sel if r["date"] and r["date"] <= _iso(eff_end)]
    m = _metrics([r["actual"] for r in sel], [r["predicted"] for r in sel])
    out = {
        "stock": stock,
        "range": {"start": sel[0]["date"] if sel else None,
                  "end": sel[-1]["date"] if sel else None},
        "count": len(sel),
        "metrics": m,
        "point_in_time": get_prediction(stock, as_of=as_of or (eff_end)),
    }
    series = sel[-limit:] if limit else sel
    out["series"] = [
        {"date": r["date"], "predicted": _round(r["predicted"], 2),
         "actual": _round(r["actual"], 2), "error": _round(r["error"], 2),
         "ape": _round(r["ape"], 3)}
        for r in series
    ]
    return out


def get_model_performance(stock: str) -> dict:
    """Real, computed performance -- never fabricated.

    Two honest views:
      * backtest_test  : the notebook's held-out test segment of Update2.
      * out_of_sample  : Update3 dates AFTER Update2 ends (never seen in training).
    """
    stock = stock.upper()
    u2_dates, u2_close = data_service.get_close_series(stock, "update2")
    boundary = _iso(u2_dates[-1]) if len(u2_dates) else None
    rows = _pvaa_full(stock)
    oos = [r for r in rows if boundary and r["date"] and r["date"] > boundary]
    ins = [r for r in rows if boundary and r["date"] and r["date"] <= boundary]

    def _dir(subset):
        if not subset:
            return None
        # direction of predicted change vs actual change is not stored per row;
        # use ape-based hit rate instead: fraction of days within 2% error.
        hits = sum(1 for r in subset if r["ape"] is not None and r["ape"] <= 2.0)
        return _round(hits / len(subset) * 100, 2)

    return {
        "stock": stock,
        "model": "FCM-LSTM (Close)",
        "horizon_days": 1,
        "training_basis": "EBT-Update2",
        "evaluation_basis": "EBT-Update3",
        "out_of_sample_boundary": boundary,
        "out_of_sample": {**_metrics([r["actual"] for r in oos], [r["predicted"] for r in oos]),
                          "within_2pct_hit_rate": _dir(oos)},
        "in_sample_reference": _metrics([r["actual"] for r in ins], [r["predicted"] for r in ins]),
        "backtest_test": _backtest_test(stock, u2_close),
        "note": ("Metrics are computed from actual model predictions vs real "
                 "observations; out-of-sample covers the period after the training "
                 "data ends and was never seen during training."),
    }


def _backtest_test(stock: str, u2_close: np.ndarray) -> dict:
    """Reproduce the notebook's held-out test-segment metrics."""
    n = len(u2_close)
    if n <= config.WINDOW:
        return {"n": 0}
    n_tr, n_va = int(n * 0.8), int(n * 0.1)
    params = ml_service.load_params(stock)
    F = ml_service.build_features(u2_close, params)
    t = np.arange(config.WINDOW, n)
    te = t >= n_tr + n_va
    if not te.any():
        return {"n": 0}
    W = np.stack([F[i - config.WINDOW:i] for i in t[te]]).astype("float32")
    out = ml_service._predict_batch(stock, W)
    lo, hi, d_sd = params["lo"], params["hi"], params["d_sd"]
    pred = u2_close[t[te] - 1] + out * d_sd * (hi - lo)
    true = u2_close[t[te]]
    naive = u2_close[t[te] - 1]
    m = _metrics(true, pred)
    mn = _metrics(true, naive)
    m["naive_mape"] = mn["mape"]
    return m


def get_model_info(stock: Optional[str] = None) -> dict:
    return ml_service.get_model_info(stock)
