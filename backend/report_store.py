"""Report history persistence.

A simple, dependency-free JSON store (spec section 18: do not introduce
unnecessarily complicated infrastructure). Thread-safe for the Flask dev server.
PDFs live in storage/reports; this file indexes them.
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from typing import List, Optional

from . import config

_LOCK = threading.Lock()


def _load() -> List[dict]:
    if not config.HISTORY_FILE.exists():
        return []
    try:
        with open(config.HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save(records: List[dict]) -> None:
    config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = config.HISTORY_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=1, default=str)
    tmp.replace(config.HISTORY_FILE)


def add_report(stock: str, report_type: str, start: str, end: str,
               as_of: Optional[str] = None, summary_text: str = "",
               status: str = "generated") -> dict:
    rec = {
        "id": uuid.uuid4().hex[:12],
        "stock": stock.upper(),
        "report_type": report_type,
        "start": start,
        "end": end,
        "as_of": as_of,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "summary_text": summary_text,
        "pdf_filename": None,
        "pdf_path": None,
        "emailed": False,
        "email_to": None,
        "email_status": None,
    }
    with _LOCK:
        records = _load()
        records.insert(0, rec)
        _save(records)
    return rec


def update_report(report_id: str, **fields) -> Optional[dict]:
    with _LOCK:
        records = _load()
        for rec in records:
            if rec["id"] == report_id:
                rec.update(fields)
                _save(records)
                return rec
    return None


def get_report(report_id: str) -> Optional[dict]:
    for rec in _load():
        if rec["id"] == report_id:
            return rec
    return None


def list_reports(limit: int = 100, stock: Optional[str] = None) -> List[dict]:
    records = _load()
    if stock:
        records = [r for r in records if r["stock"] == stock.upper()]
    return records[:limit]


def clear() -> int:
    with _LOCK:
        n = len(_load())
        _save([])
    return n
