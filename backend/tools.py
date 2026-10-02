"""Controlled tools exposed to the Qwen agent (spec sections 9, 25).

Every tool is a thin wrapper over the SAME shared services the REST API and
dashboard use -- no duplicate calculation logic. Tools return structured data so
the model interprets real values instead of inventing numbers. The agent has no
filesystem or shell access; it can only call these whitelisted functions.
"""
from __future__ import annotations

import os
from typing import Optional

from . import analysis_service, data_service, report_manager, report_store


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _trim_history(rows, limit=90):
    rows = rows[-limit:] if limit else rows
    return [{"date": r["date"], "close": r["close"], "volume": r.get("volume")}
            for r in rows]


# --------------------------------------------------------------------------
# tool implementations (each returns a JSON-serialisable dict)
# --------------------------------------------------------------------------
def get_available_stocks(**_):
    return {"stocks": analysis_service.get_available_stocks()}


def get_stock_data(stock_code: str, as_of: Optional[str] = None, **_):
    return analysis_service.get_stock_data(stock_code, as_of=as_of)


def get_historical_data(stock_code: str, start: Optional[str] = None,
                        end: Optional[str] = None, as_of: Optional[str] = None,
                        limit: int = 90, **_):
    rows = analysis_service.get_historical_data(stock_code, start, end, as_of)
    return {"stock": stock_code.upper(), "count": len(rows),
            "returned": min(len(rows), limit or len(rows)),
            "series": _trim_history(rows, limit)}


def get_prediction(stock_code: str, as_of: Optional[str] = None, **_):
    return analysis_service.get_prediction(stock_code, as_of=as_of)


def get_prediction_vs_actual(stock_code: str, start: Optional[str] = None,
                             end: Optional[str] = None, as_of: Optional[str] = None,
                             limit: int = 30, **_):
    r = analysis_service.get_prediction_vs_actual(stock_code, start, end, as_of, limit)
    if limit == 0:
        r.pop("series", None)
    return r


def get_model_performance(stock_code: str, **_):
    return analysis_service.get_model_performance(stock_code)


def get_model_info(stock_code: Optional[str] = None, **_):
    return analysis_service.get_model_info(stock_code)


def get_period_analysis(stock_code: str, start: str, end: str,
                        as_of: Optional[str] = None, **_):
    a = analysis_service.get_period_analysis(stock_code, start, end, as_of)
    a["stock"] = stock_code.upper()
    a["period"] = {"start": start, "end": end}
    return a


def generate_report(stock_code: str, report_type: str = "custom",
                    start: Optional[str] = None, end: Optional[str] = None,
                    ref_date: Optional[str] = None, as_of: Optional[str] = None, **_):
    rep, rec = report_manager.build_report(stock_code, report_type, start, end,
                                           ref_date=ref_date, as_of=as_of)
    return report_manager.public_view(rep, rec["id"])


def generate_pdf_report(report_id: Optional[str] = None, stock_code: Optional[str] = None,
                        report_type: str = "custom", start: Optional[str] = None,
                        end: Optional[str] = None, ref_date: Optional[str] = None,
                        as_of: Optional[str] = None, **_):
    if not report_id:
        _rep, rec = report_manager.build_report(stock_code, report_type, start, end,
                                                ref_date=ref_date, as_of=as_of)
        report_id = rec["id"]
    try:
        path = report_manager.make_pdf(report_id)
    except FileNotFoundError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"PDF generation failed: {e}"}
    rep, rec = report_manager.get_report(report_id)
    return {"success": True, "report_id": report_id,
            "pdf_filename": os.path.basename(path),
            "pdf_url": f"/api/reports/{report_id}/pdf",
            "stock": rep["stock"], "period": rep["period"]}


def send_report_email(report_id: Optional[str] = None, recipient: Optional[str] = None,
                      stock_code: Optional[str] = None, report_type: str = "custom",
                      start: Optional[str] = None, end: Optional[str] = None,
                      as_of: Optional[str] = None, **_):
    if not recipient:
        return {"success": False, "need_input": "recipient",
                "error": "A recipient email address is required. Ask the user for it."}
    if not report_id:
        _rep, rec = report_manager.build_report(stock_code, report_type, start, end,
                                                as_of=as_of)
        report_id = rec["id"]
    return report_manager.email_report(report_id, recipient)


def get_report_history(limit: int = 25, stock_code: Optional[str] = None, **_):
    reports = report_store.list_reports(limit, stock_code)
    return {"count": len(reports), "reports": reports}


# --------------------------------------------------------------------------
# registry + schemas (OpenAI-compatible function calling; Qwen supports this)
# --------------------------------------------------------------------------
TOOL_FUNCS = {
    "get_available_stocks": get_available_stocks,
    "get_stock_data": get_stock_data,
    "get_historical_data": get_historical_data,
    "get_prediction": get_prediction,
    "get_prediction_vs_actual": get_prediction_vs_actual,
    "get_model_performance": get_model_performance,
    "get_model_info": get_model_info,
    "get_period_analysis": get_period_analysis,
    "generate_report": generate_report,
    "generate_pdf_report": generate_pdf_report,
    "send_report_email": send_report_email,
    "get_report_history": get_report_history,
}

_S = {"type": "string"}


def _p(props, required=None):
    return {"type": "object", "properties": props, "required": required or []}


_TYPE_ENUM = {"type": "string",
              "enum": ["daily", "weekly", "monthly", "yearly", "custom"]}

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "get_available_stocks",
        "description": "List the supported renewable-energy stock universe with names, energy type, data coverage and whether a model exists.",
        "parameters": _p({})}},
    {"type": "function", "function": {
        "name": "get_stock_data",
        "description": "Latest snapshot for a stock: last close, previous close, change, OHLC, volume. Optional as_of date for historical replay.",
        "parameters": _p({"stock_code": _S, "as_of": _S}, ["stock_code"])}},
    {"type": "function", "function": {
        "name": "get_historical_data",
        "description": "Historical daily close/OHLC series for a stock, optionally bounded by start/end or an as_of cutoff.",
        "parameters": _p({"stock_code": _S, "start": _S, "end": _S, "as_of": _S,
                          "limit": {"type": "integer"}}, ["stock_code"])}},
    {"type": "function", "function": {
        "name": "get_prediction",
        "description": "Run the existing FCM-LSTM model to forecast the NEXT-day close using only data up to as_of. If the next day's actual exists it is returned for evaluation.",
        "parameters": _p({"stock_code": _S, "as_of": _S}, ["stock_code"])}},
    {"type": "function", "function": {
        "name": "get_prediction_vs_actual",
        "description": "Prediction-vs-actual pairs and accuracy metrics (MAE/RMSE/MAPE) for a stock, optionally within a date range or up to an as_of cutoff.",
        "parameters": _p({"stock_code": _S, "start": _S, "end": _S, "as_of": _S,
                          "limit": {"type": "integer"}}, ["stock_code"])}},
    {"type": "function", "function": {
        "name": "get_model_performance",
        "description": "Real computed model performance: out-of-sample metrics (after training data ends), backtest test-set metrics, hit rate.",
        "parameters": _p({"stock_code": _S}, ["stock_code"])}},
    {"type": "function", "function": {
        "name": "get_model_info",
        "description": "Model architecture and parameters. Pass stock_code for per-ticker scaler/FCM params, or omit for the general architecture.",
        "parameters": _p({"stock_code": _S})}},
    {"type": "function", "function": {
        "name": "get_period_analysis",
        "description": "Full analysis for a stock over an explicit date range: statistics plus prediction-vs-actual.",
        "parameters": _p({"stock_code": _S, "start": _S, "end": _S, "as_of": _S},
                         ["stock_code", "start", "end"])}},
    {"type": "function", "function": {
        "name": "generate_report",
        "description": "Generate a period report and store it in history. report_type in {daily,weekly,monthly,yearly,custom}; custom needs start+end. Returns a report_id.",
        "parameters": _p({"stock_code": _S, "report_type": _TYPE_ENUM,
                          "start": _S, "end": _S, "ref_date": _S, "as_of": _S},
                         ["stock_code"])}},
    {"type": "function", "function": {
        "name": "generate_pdf_report",
        "description": "Render a report to PDF. Provide report_id (from generate_report) OR the report parameters. Returns a pdf_url.",
        "parameters": _p({"report_id": _S, "stock_code": _S, "report_type": _TYPE_ENUM,
                          "start": _S, "end": _S, "as_of": _S})}},
    {"type": "function", "function": {
        "name": "send_report_email",
        "description": "Email a report PDF to a recipient. Requires recipient; if missing, ask the user. Provide report_id or report parameters.",
        "parameters": _p({"report_id": _S, "recipient": _S, "stock_code": _S,
                          "report_type": _TYPE_ENUM, "start": _S, "end": _S, "as_of": _S})}},
    {"type": "function", "function": {
        "name": "get_report_history",
        "description": "List previously generated reports (id, stock, type, range, status, pdf, email status).",
        "parameters": _p({"limit": {"type": "integer"}, "stock_code": _S})}},
]

# Human-friendly present-progressive activity labels for the UI.
_ACTIVITY = {
    "get_available_stocks": "Loading the stock universe...",
    "get_stock_data": "Fetching {stock_code} latest data...",
    "get_historical_data": "Getting {stock_code} historical data...",
    "get_prediction": "Running the FCM-LSTM prediction for {stock_code}...",
    "get_prediction_vs_actual": "Comparing {stock_code} prediction with actual...",
    "get_model_performance": "Reading {stock_code} model performance...",
    "get_model_info": "Reading model information...",
    "get_period_analysis": "Analysing {stock_code} for the period...",
    "generate_report": "Generating the {report_type} report for {stock_code}...",
    "generate_pdf_report": "Rendering the PDF...",
    "send_report_email": "Sending the report by email...",
    "get_report_history": "Loading report history...",
}


def activity_for(name: str, args: dict) -> str:
    tmpl = _ACTIVITY.get(name, f"Running {name}...")
    try:
        return tmpl.format(**{k: (v if v is not None else "") for k, v in (args or {}).items()})
    except Exception:
        return tmpl


def execute_tool(name: str, arguments: dict) -> dict:
    fn = TOOL_FUNCS.get(name)
    if fn is None:
        return {"ok": False, "error": f"Unknown tool '{name}'."}
    try:
        data = fn(**(arguments or {}))
        return {"ok": True, "data": data}
    except data_service.StockNotFoundError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
