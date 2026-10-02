"""Report manager: single owner of build -> cache -> PDF -> email -> history.

Both the REST API and the Qwen agent tools delegate here so a report is analysed
exactly once and reused everywhere (spec section 30). Nothing is duplicated.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

from . import email_service, pdf_service, report_service, report_store

_CACHE: dict = {}


def build_report(stock: str, report_type: str = "custom", start: Optional[str] = None,
                 end: Optional[str] = None, ref_date: Optional[str] = None,
                 as_of: Optional[str] = None, store: bool = True) -> Tuple[dict, Optional[dict]]:
    rep = report_service.get_report_data(stock, report_type, start, end,
                                         ref_date=ref_date, as_of=as_of)
    if not store:
        return rep, None
    rec = report_store.add_report(
        rep["stock"], rep["report_type"], rep["period"]["start"], rep["period"]["end"],
        as_of=rep.get("as_of"), summary_text=rep.get("summary_text", ""))
    _CACHE[rec["id"]] = rep
    return rep, rec


def get_report(report_id: str) -> Tuple[Optional[dict], Optional[dict]]:
    """Return (report_payload, history_record); rebuilds payload from cache/store."""
    rec = report_store.get_report(report_id)
    if not rec:
        return None, None
    rep = _CACHE.get(report_id)
    if rep is None:
        rep = report_service.get_report_data(
            rec["stock"], rec["report_type"], rec["start"], rec["end"],
            as_of=rec.get("as_of"))
        _CACHE[report_id] = rep
    return rep, rec


def make_pdf(report_id: str, qwen_summary: Optional[str] = None) -> str:
    rep, rec = get_report(report_id)
    if rec is None:
        raise FileNotFoundError(f"No report with id '{report_id}'.")
    summary = qwen_summary or rec.get("qwen_summary")
    if qwen_summary:
        report_store.update_report(report_id, qwen_summary=qwen_summary)
    path = pdf_service.generate_pdf(rep, report_id, qwen_summary=summary)
    report_store.update_report(report_id, status="pdf_ready",
                               pdf_filename=os.path.basename(path), pdf_path=path)
    return path


def email_report(report_id: str, recipient: str) -> dict:
    rep, rec = get_report(report_id)
    if rec is None:
        return {"success": False, "error": f"No report with id '{report_id}'."}
    pdf_path = rec.get("pdf_path")
    if not pdf_path or not os.path.exists(pdf_path):
        pdf_path = make_pdf(report_id)
    res = email_service.send_report_email(recipient, rep, pdf_path)
    report_store.update_report(
        report_id, emailed=bool(res.get("success")),
        email_to=recipient if res.get("success") else rec.get("email_to"),
        email_status="sent" if res.get("success") else f"failed: {res.get('error')}")
    res["report_id"] = report_id
    return res


def public_view(rep: dict, report_id: str) -> dict:
    """Token-friendly report view for the agent (drops large chart arrays)."""
    return {
        "report_id": report_id,
        "stock": rep["stock"],
        "stock_name": rep.get("stock_name"),
        "report_type": rep["report_type"],
        "period": rep["period"],
        "replay": rep.get("replay"),
        "as_of": rep.get("as_of"),
        "statistics": rep.get("statistics"),
        "prediction": rep.get("prediction"),
        "prediction_vs_actual": rep.get("prediction_vs_actual"),
        "model_performance": rep.get("model_performance"),
        "notable_movements": (rep.get("notable_movements") or [])[:5],
        "summary_text": rep.get("summary_text"),
    }
