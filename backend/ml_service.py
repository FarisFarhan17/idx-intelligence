"""ML service: faithful reproduction of FCM_LSTM_Close.ipynb inference.

The trained model is loaded ONCE per ticker and cached. Nothing is retrained.
The pipeline below mirrors the notebook exactly:

    sc      = (close - lo) / (hi - lo)
    ret     = [0, diff(close)/close[:-1]]
    fcm_in  = [sc, clip(ret/(4*r_sd), -1, 1)]
    U       = FCM_membership(fcm_in, centers, m=2)     # (n, 3)
    F       = [sc, U0, U1, U2]                          # (n, 4)
    out     = model.predict(F[i-20:i])                  # normalized next-day delta
    pred_i  = close[i-1] + out * d_sd * (hi - lo)       # predicted close on day i

The model target is the next-day change in scaled close, so a prediction for the
day AFTER the last available close is the "latest prediction" (no actual yet).
"""
from __future__ import annotations

import json
import threading
import zipfile
from typing import Dict, List, Optional

import numpy as np

from . import config

_keras = None
_keras_lock = threading.Lock()
_MODEL_CACHE: Dict[str, object] = {}
_PARAMS_CACHE: Dict[str, dict] = {}
_INFO_CACHE: Dict[str, dict] = {}


class ModelNotAvailableError(RuntimeError):
    """Raised when the Keras runtime or a model file is unavailable."""


# --- Keras runtime ---------------------------------------------------------
def _load_keras():
    """Lazily import Keras 3 (bundled with TensorFlow or standalone)."""
    global _keras
    if _keras is not None:
        return _keras
    with _keras_lock:
        if _keras is not None:
            return _keras
        import os
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("KERAS_BACKEND", os.getenv("KERAS_BACKEND", "tensorflow"))
        try:
            import keras  # type: ignore
        except Exception:  # fall back to the TF-bundled namespace
            from tensorflow import keras  # type: ignore
        _keras = keras
        return _keras


def _model_path(stock: str):
    return config.ML_DIR / f"FCM_LSTM_{stock}.keras"


def _params_path(stock: str):
    return config.ML_DIR / f"FCM_LSTM_{stock}_params.npz"


def load_params(stock: str) -> dict:
    """Load cached FCM/scaler parameters from the .npz artifact."""
    stock = stock.upper()
    if stock in _PARAMS_CACHE:
        return _PARAMS_CACHE[stock]
    p = _params_path(stock)
    if not p.exists():
        raise ModelNotAvailableError(f"No parameter file for {stock}: {p.name}")
    with np.load(p, allow_pickle=False) as z:
        params = {
            "lo": float(z["lo"]),
            "hi": float(z["hi"]),
            "r_sd": float(z["r_sd"]),
            "d_sd": float(z["d_sd"]),
            "centers": np.asarray(z["centers"], dtype="float64"),
        }
    _PARAMS_CACHE[stock] = params
    return params


def load_model(stock: str):
    """Load (once) and cache the trained Keras model for a ticker."""
    stock = stock.upper()
    if stock in _MODEL_CACHE:
        return _MODEL_CACHE[stock]
    path = _model_path(stock)
    if not path.exists():
        raise ModelNotAvailableError(f"No model file for {stock}: {path.name}")
    try:
        keras = _load_keras()
        model = keras.models.load_model(str(path), compile=False)
    except Exception as e:  # pragma: no cover - environment dependent
        raise ModelNotAvailableError(
            f"Failed to load model for {stock}. Is TensorFlow/Keras installed? ({e})"
        ) from e
    _MODEL_CACHE[stock] = model
    return model


# --- FCM + feature engineering (exact notebook logic) ----------------------
def fcm_membership(X: np.ndarray, centers: np.ndarray, m: float = config.FUZZ_M) -> np.ndarray:
    """Fuzzy C-Means membership matrix U (n, c) for points X (n, d)."""
    d = np.linalg.norm(X[:, None, :] - centers[None], axis=2) + 1e-12
    inv = d ** (-2.0 / (m - 1))
    return inv / inv.sum(axis=1, keepdims=True)


def build_features(close: np.ndarray, params: dict) -> np.ndarray:
    """Build the (n, 4) feature matrix F = [scaled_close, u0, u1, u2]."""
    close = np.asarray(close, dtype="float64")
    lo, hi = params["lo"], params["hi"]
    r_sd = params["r_sd"]
    sc = (close - lo) / (hi - lo)
    ret = np.r_[0.0, np.diff(close) / close[:-1]]
    fcm_in = np.c_[sc, np.clip(ret / (4 * r_sd), -1, 1)]
    U = fcm_membership(fcm_in, params["centers"], config.FUZZ_M)
    return np.c_[sc, U].astype("float32")


def _predict_batch(stock: str, X: np.ndarray) -> np.ndarray:
    """Run the model on a batch of windows X (N, 20, 4) -> (N,) predictions."""
    model = load_model(stock)
    X = np.asarray(X, dtype="float32")
    out = model.predict(X, verbose=0)
    return np.asarray(out).ravel()


def _make_windows(F: np.ndarray, window: int = config.WINDOW) -> np.ndarray:
    """Windows W[j] = F[j : j+window]; predicts target index (j+window)."""
    n = len(F)
    if n < window:
        return np.empty((0, window, F.shape[1]), dtype="float32")
    idx = np.arange(n - window + 1)
    return np.stack([F[i:i + window] for i in idx]).astype("float32")


# --- Public prediction API -------------------------------------------------
def predict_latest(stock: str, close: np.ndarray) -> dict:
    """Predict the NEXT day's close after the last value in ``close``.

    Returns structured data (no fabrication): if there is not enough history the
    prediction is None with a reason.
    """
    stock = stock.upper()
    close = np.asarray(close, dtype="float64")
    params = load_params(stock)
    result = {
        "stock": stock,
        "available": False,
        "last_close": float(close[-1]) if len(close) else None,
        "predicted_close": None,
        "reason": None,
    }
    if len(close) < config.WINDOW:
        result["reason"] = (
            f"Need at least {config.WINDOW} days of history to predict "
            f"(have {len(close)})."
        )
        return result

    F = build_features(close, params)
    window = F[-config.WINDOW:][None, :, :]
    out = float(_predict_batch(stock, window)[0])
    lo, hi, d_sd = params["lo"], params["hi"], params["d_sd"]
    pred = float(close[-1] + out * d_sd * (hi - lo))

    last_memberships = F[-1][1:].astype(float)  # [u0,u1,u2] of the last day
    cluster = int(np.argmax(last_memberships))
    result.update({
        "available": True,
        "predicted_close": pred,
        "predicted_change": pred - float(close[-1]),
        "predicted_change_pct": (pred - float(close[-1])) / float(close[-1]) * 100
        if close[-1] else None,
        "normalized_delta": out,
        "cluster": cluster,
        "memberships": [round(float(x), 4) for x in last_memberships],
        "membership_max": round(float(last_memberships.max()), 4),
        "scaled_last_close": float(F[-1][0]),
        "horizon_days": 1,  # model predicts the next trading day (T+1)
    })
    return result


def predict_vs_actual_series(stock: str, dates: np.ndarray, close: np.ndarray) -> List[dict]:
    """For every day with enough prior history, predict that day's close using
    only information available up to the previous day, then pair with the real
    actual close. This is the backbone of prediction-vs-actual analysis and is
    leak-free (window ends at i-1 to predict day i).
    """
    stock = stock.upper()
    close = np.asarray(close, dtype="float64")
    n = len(close)
    if n <= config.WINDOW:
        return []
    params = load_params(stock)
    F = build_features(close, params)
    W = _make_windows(F, config.WINDOW)          # (n-window+1, 20, 4)
    preds_norm = _predict_batch(stock, W)         # (n-window+1,)
    lo, hi, d_sd = params["lo"], params["hi"], params["d_sd"]

    rows: List[dict] = []
    # W[j] uses F[j:j+window] and predicts target index t = j + window.
    for j in range(len(W)):
        t = j + config.WINDOW
        if t >= n:
            break
        base = close[t - 1]
        pred = float(base + preds_norm[j] * d_sd * (hi - lo))
        actual = float(close[t])
        err = pred - actual
        rows.append({
            "date": _iso(dates[t]) if dates is not None and t < len(dates) else None,
            "predicted": pred,
            "actual": actual,
            "error": err,
            "abs_error": abs(err),
            "ape": abs(err) / actual * 100 if actual else None,
        })
    return rows


# --- Model information (from the real artifact, never fabricated) ----------
def get_model_info(stock: Optional[str] = None) -> dict:
    """Architecture + parameter info read from the actual .keras file."""
    if stock is None:
        return {
            "architecture": "Sequential: LSTM(64, return_sequences) -> Dropout(0.2) "
                            "-> LSTM(32) -> Dense(16, relu) -> Dense(1)",
            "window": config.WINDOW,
            "features": 4,
            "feature_layout": ["scaled_close", "fcm_membership_0",
                               "fcm_membership_1", "fcm_membership_2"],
            "n_clusters": config.N_CLUSTERS,
            "fuzzifier_m": config.FUZZ_M,
            "target": "next-day change in min-max scaled close (normalized by d_sd)",
            "horizon_days": 1,
            "keras_saved_version": _keras_saved_version(),
            "tickers": config.STOCKS,
        }
    stock = stock.upper()
    if stock in _INFO_CACHE:
        return _INFO_CACHE[stock]
    info = {"stock": stock, "available": _model_path(stock).exists()}
    try:
        model = load_model(stock)
        info["params"] = {
            "lo": load_params(stock)["lo"],
            "hi": load_params(stock)["hi"],
            "r_sd": load_params(stock)["r_sd"],
            "d_sd": load_params(stock)["d_sd"],
            "centers": load_params(stock)["centers"].tolist(),
        }
        try:
            info["trainable_parameters"] = int(model.count_params())
        except Exception:
            info["trainable_parameters"] = None
        try:
            info["layers"] = [
                {"name": l.name, "type": l.__class__.__name__}
                for l in model.layers
            ]
        except Exception:
            info["layers"] = []
    except Exception as e:
        info["error"] = str(e)
    _INFO_CACHE[stock] = info
    return info


def _keras_saved_version() -> Optional[str]:
    """Read keras_version from a model's metadata.json (no TF needed)."""
    try:
        with zipfile.ZipFile(_model_path(config.STOCKS[0])) as z:
            if "metadata.json" in z.namelist():
                return json.loads(z.read("metadata.json").decode()).get("keras_version")
    except Exception:
        pass
    return None


def _iso(v) -> Optional[str]:
    import pandas as pd
    try:
        return pd.to_datetime(v).date().isoformat()
    except Exception:
        return str(v)
