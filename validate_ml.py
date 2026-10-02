"""Phase 2 validation: prove the reconstructed inference matches the notebook.

Part A - reproduce the notebook's Update2 test-set metrics using the SAVED model
         + SAVED params (no retraining). If MAE/RMSE/MAPE come out sensible and
         beat the naive baseline, the reconstruction is byte-faithful.
Part B - leak-free prediction-vs-actual on Update3 (later actual observations).
Part C - the "latest prediction" (T+1 beyond the last actual) for each stock.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

sys.path.insert(0, r"C:\IDX PORTO")
from backend import config, data_service, ml_service  # noqa: E402


def metrics(true, pred):
    true = np.asarray(true, float)
    pred = np.asarray(pred, float)
    return dict(
        RMSE=float(np.sqrt(np.mean((true - pred) ** 2))),
        MAE=float(np.mean(np.abs(true - pred))),
        MAPE=float(np.mean(np.abs((true - pred) / true)) * 100),
    )


def part_a(stock):
    """Reproduce notebook test-set evaluation on Update2."""
    dates, close = data_service.get_close_series(stock, "update2")
    n = len(close)
    n_tr, n_va = int(n * 0.8), int(n * 0.1)
    params = ml_service.load_params(stock)
    F = ml_service.build_features(close, params)
    t = np.arange(config.WINDOW, n)
    te = t >= n_tr + n_va
    W = np.stack([F[i - config.WINDOW:i] for i in t[te]]).astype("float32")
    out = ml_service._predict_batch(stock, W)
    lo, hi, d_sd = params["lo"], params["hi"], params["d_sd"]
    pred = close[t[te] - 1] + out * d_sd * (hi - lo)
    true = close[t[te]]
    naive = close[t[te] - 1]
    m = metrics(true, pred)
    mn = metrics(true, naive)
    return n, int(te.sum()), m, mn


def part_b(stock):
    """Leak-free prediction-vs-actual on Update3."""
    dates, close = data_service.get_close_series(stock, "update3")
    rows = ml_service.predict_vs_actual_series(stock, dates, close)
    if not rows:
        return None, 0
    true = [r["actual"] for r in rows]
    pred = [r["predicted"] for r in rows]
    return metrics(true, pred), len(rows)


def part_c(stock):
    dates, close = data_service.get_close_series(stock, "update3")
    r = ml_service.predict_latest(stock, close)
    last = dates[-1] if len(dates) else None
    import pandas as pd
    return r, (pd.to_datetime(last).date().isoformat() if last is not None else None)


def main():
    t0 = time.time()
    print("=" * 78)
    print("PART A - Reproduce notebook Update2 test metrics (saved model + params)")
    print("=" * 78)
    print(f"{'TICK':5} {'N':>5} {'Test':>5} | {'MAE':>9} {'RMSE':>9} {'MAPE%':>7} "
          f"| {'naiveMAPE%':>10}")
    for s in config.STOCKS:
        try:
            n, nte, m, mn = part_a(s)
            print(f"{s:5} {n:5d} {nte:5d} | {m['MAE']:9.3f} {m['RMSE']:9.3f} "
                  f"{m['MAPE']:7.3f} | {mn['MAPE']:10.3f}")
        except Exception as e:
            print(f"{s:5}  ERROR: {e!r}")

    print("\n" + "=" * 78)
    print("PART B - Leak-free prediction-vs-actual on Update3 (full history)")
    print("=" * 78)
    print(f"{'TICK':5} {'pairs':>6} | {'MAE':>10} {'RMSE':>10} {'MAPE%':>8}")
    for s in config.STOCKS:
        try:
            m, cnt = part_b(s)
            if m is None:
                print(f"{s:5} {0:6d} | (insufficient data)")
            else:
                print(f"{s:5} {cnt:6d} | {m['MAE']:10.3f} {m['RMSE']:10.3f} {m['MAPE']:8.3f}")
        except Exception as e:
            print(f"{s:5}  ERROR: {e!r}")

    print("\n" + "=" * 78)
    print("PART C - Latest T+1 prediction (beyond last actual)")
    print("=" * 78)
    for s in config.STOCKS:
        try:
            r, last = part_c(s)
            if r["available"]:
                print(f"{s:5} last={last} close={r['last_close']:.1f} -> "
                      f"pred(T+1)={r['predicted_close']:.1f} "
                      f"({r['predicted_change_pct']:+.2f}%) cluster=C{r['cluster']} "
                      f"U={r['memberships']}")
            else:
                print(f"{s:5} {r['reason']}")
        except Exception as e:
            print(f"{s:5}  ERROR: {e!r}")

    print(f"\n[elapsed {time.time()-t0:.1f}s]")
    print("keras_saved_version:", ml_service._keras_saved_version())


if __name__ == "__main__":
    main()
