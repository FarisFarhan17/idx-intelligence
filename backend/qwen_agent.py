"""Qwen AI agent (spec sections 7, 8, 9, 25).

A real agent: it decides which controlled backend tools to call, executes them,
feeds the structured results back, and only then answers. The API key lives in
environment variables and is never sent to the client. Uses the OpenAI-compatible
DashScope endpoint via stdlib urllib (no extra dependency).

Safety: when a Historical Replay cutoff (``as_of``) is active it is injected into
any tool that supports it, so the agent cannot accidentally leak future data even
if it forgets to pass the cutoff itself.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Dict, List, Optional

from . import config, tools

MAX_TOOL_ROUNDS = 8
_TOOLS_WITH_AS_OF = {
    "get_stock_data", "get_historical_data", "get_prediction",
    "get_prediction_vs_actual", "get_period_analysis", "generate_report",
    "generate_pdf_report", "send_report_email",
}

SYSTEM_BASE = """You are the IDX Intelligence AI agent for a renewable-energy stock
analysis platform (Indonesia Stock Exchange). You operate the analysis and
reporting workflow by calling tools -- you do not guess.

Supported stocks: DGIK, JARR, TGRA, PTBA, POWR, PGEO, MPOW, ARKO, APEX.
Model: an existing FCM-LSTM that predicts the NEXT-DAY (T+1) close price from the
daily close series. It was trained on EBT-Update2; EBT-Update3 holds later ACTUAL
observations used for prediction-vs-actual evaluation. Data is a provided
historical dataset, NOT a live IDX feed.

STRICT RULES:
- Use tool results for every number. NEVER invent prices, predictions, model
  metrics, dates, accuracy or confidence values.
- Clearly distinguish a PREDICTION from an ACTUAL observation.
- If data is unavailable (e.g. no actual for a forecast date, or a stock/period
  has no data), say so plainly instead of fabricating.
- Never claim real-time/live market data. Reference the latest dataset date.
- Be concise and useful. Prefer short answers with the key figures.
- To act (analyse, predict, compare, report, email), CALL THE APPROPRIATE TOOL.
  For "email it to me" without an address, ask the user for the recipient first.
- For reports: report_type is one of daily/weekly/monthly/yearly/custom. Custom
  needs start and end (YYYY-MM-DD). generate_report returns a report_id you can
  pass to generate_pdf_report and send_report_email."""


def _system_prompt(as_of: Optional[str], latest_date: Optional[str]) -> str:
    extra = []
    if latest_date:
        extra.append(f"The latest available dataset date is {latest_date}.")
    if as_of:
        extra.append(
            f"HISTORICAL REPLAY IS ACTIVE with cutoff as_of={as_of}. Treat {as_of} "
            f"as 'today'. Only information up to {as_of} may be used for "
            f"predictions; later actuals may be referenced only to evaluate a "
            f"prediction. Pass as_of to your tools.")
    return SYSTEM_BASE + ("\n\n" + " ".join(extra) if extra else "")


def _inject_as_of(name: str, args: Dict, as_of: Optional[str]) -> Dict:
    if as_of and name in _TOOLS_WITH_AS_OF and not args.get("as_of"):
        args = {**args, "as_of": as_of}
    return args


def _post_chat(messages: List[dict], use_tools: bool) -> dict:
    url = config.QWEN_BASE_URL.rstrip("/") + "/chat/completions"
    payload = {
        "model": config.QWEN_MODEL,
        "messages": messages,
        "temperature": 0.2,
    }
    if use_tools:
        payload["tools"] = tools.TOOL_SPECS
        payload["tool_choice"] = "auto"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.QWEN_API_KEY}",
    })
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")
        raise RuntimeError(f"Qwen API HTTP {e.code}: {body[:400]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Qwen API connection error: {e.reason}") from e


def run_agent(user_message: str, history: Optional[List[dict]] = None,
              as_of: Optional[str] = None) -> dict:
    """Run one user turn through the agent loop.

    Returns {reply, activities, tool_trace, configured}. ``activities`` is a
    list of human-readable status strings for the UI tool/activity feed.
    """
    if not config.qwen_configured():
        return {
            "configured": False,
            "reply": ("The Qwen AI agent is not configured yet. Set QWEN_API_KEY "
                      "(and optionally QWEN_MODEL / QWEN_BASE_URL) in the .env file, "
                      "then restart the server. Everything else -- dashboard, "
                      "predictions, reports, PDF and history -- works without it."),
            "activities": [],
            "tool_trace": [],
        }

    from . import data_service
    latest = data_service.get_latest_data_date()
    messages: List[dict] = [{"role": "system", "content": _system_prompt(as_of, latest)}]
    for h in (history or [])[-12:]:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_message})

    activities: List[str] = []
    tool_trace: List[dict] = []

    for _ in range(MAX_TOOL_ROUNDS):
        resp = _post_chat(messages, use_tools=True)
        choice = (resp.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            return {
                "configured": True,
                "reply": (msg.get("content") or "").strip() or
                         "(The agent returned no content.)",
                "activities": activities,
                "tool_trace": tool_trace,
            }

        # Append the assistant message that requested the tools.
        messages.append({
            "role": "assistant",
            "content": msg.get("content") or "",
            "tool_calls": tool_calls,
        })
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            args = _inject_as_of(name, args, as_of)
            activities.append(tools.activity_for(name, args))
            result = tools.execute_tool(name, args)
            tool_trace.append({"tool": name, "args": args,
                               "ok": result.get("ok", False)})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id"),
                "name": name,
                "content": json.dumps(result, default=str)[:6000],
            })

    # Ran out of rounds: ask for a final answer without tools.
    final = _post_chat(messages, use_tools=False)
    content = ((final.get("choices") or [{}])[0].get("message") or {}).get("content", "")
    return {"configured": True, "reply": content.strip() or
            "(The agent did not produce a final answer.)",
            "activities": activities, "tool_trace": tool_trace}


def summarize_report(report: dict) -> Optional[str]:
    """Ask Qwen for a short interpretation of an already-computed report.

    Used to enrich PDFs/reports. Falls back to None (the rule-based summary is
    kept) if Qwen is unavailable -- never fabricates.
    """
    if not config.qwen_configured():
        return None
    compact = {
        "stock": report.get("stock"), "period": report.get("period"),
        "statistics": report.get("statistics"),
        "notable_movements": (report.get("notable_movements") or [])[:5],
    }
    messages = [
        {"role": "system", "content":
            "You write concise market commentary for a client report. Use only the "
            "numbers provided; never invent values. Focus on price action, trend, "
            "volatility, volume and notable sessions. Do NOT mention any forecasting "
            "model, predictions, machine learning, clusters, memberships or accuracy "
            "metrics. Write 3-5 sentences."},
        {"role": "user", "content": "Report data:\n" + json.dumps(compact, default=str)[:6000]},
    ]
    try:
        resp = _post_chat(messages, use_tools=False)
        return ((resp.get("choices") or [{}])[0].get("message") or {}).get("content", "").strip() or None
    except Exception:
        return None
