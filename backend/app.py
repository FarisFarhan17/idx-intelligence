"""Flask REST API + static frontend host.

Frontend -> Backend API -> Shared analysis/data services (spec section 27).
The Qwen agent calls the SAME services via backend.tools, so there is one source
of truth for every calculation.
"""
from __future__ import annotations

import os
import threading
import traceback
from typing import Optional

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

from . import (analysis_service, config, data_service, qwen_agent,
               report_manager, report_store, tools)


def _json() -> dict:
    return request.get_json(silent=True) or {}


def _q(name: str, default=None):
    v = request.args.get(name)
    return default if v in (None, "") else v


def _int_q(name: str, default: Optional[int] = None) -> Optional[int]:
    v = request.args.get(name)
    try:
        return int(v) if v not in (None, "") else default
    except ValueError:
        return default


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(config.FRONTEND_DIR), static_url_path="")
    CORS(app)
    app.config["JSON_SORT_KEYS"] = False

    # ---- error handlers ---------------------------------------------------
    @app.errorhandler(data_service.StockNotFoundError)
    def _stock_404(e):
        return jsonify({"error": str(e)}), 404

    @app.errorhandler(ValueError)
    def _value_400(e):
        return jsonify({"error": str(e)}), 400

    @app.errorhandler(404)
    def _not_found(e):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Not found", "path": request.path}), 404
        # SPA fallback: serve the frontend shell.
        return send_from_directory(app.static_folder, "index.html")

    @app.errorhandler(Exception)
    def _server_error(e):
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    # ---- meta -------------------------------------------------------------
    @app.get("/api/health")
    def health():
        from . import ml_service
        return jsonify({
            "status": "ok",
            "keras_saved_version": ml_service._keras_saved_version(),
            "latest_dataset_date": data_service.get_latest_data_date(),
            "stocks": config.STOCKS,
            "qwen_configured": config.qwen_configured(),
            "smtp_configured": config.smtp_configured(),
        })

    @app.get("/api/coverage")
    def coverage():
        return jsonify(data_service.get_data_coverage())

    # ---- stocks / data ----------------------------------------------------
    @app.get("/api/stocks")
    def stocks():
        return jsonify({"stocks": analysis_service.get_available_stocks()})

    @app.get("/api/stocks/<code>")
    def stock_snapshot(code):
        return jsonify(analysis_service.get_stock_data(code, as_of=_q("as_of")))

    @app.get("/api/stocks/<code>/history")
    def stock_history(code):
        rows = analysis_service.get_historical_data(
            code, _q("start"), _q("end"), _q("as_of"), _q("source", "update3"))
        return jsonify({"stock": code.upper(), "count": len(rows), "series": rows})

    # ---- ML ---------------------------------------------------------------
    @app.get("/api/stocks/<code>/prediction")
    def stock_prediction(code):
        return jsonify(analysis_service.get_prediction(code, as_of=_q("as_of")))

    @app.get("/api/stocks/<code>/prediction-vs-actual")
    def stock_pva(code):
        return jsonify(analysis_service.get_prediction_vs_actual(
            code, _q("start"), _q("end"), _q("as_of"), _int_q("limit")))

    @app.get("/api/stocks/<code>/performance")
    def stock_performance(code):
        return jsonify(analysis_service.get_model_performance(code))

    @app.get("/api/model/info")
    def model_info():
        return jsonify(analysis_service.get_model_info(_q("stock_code")))

    @app.get("/api/period-analysis")
    def period_analysis():
        code = _q("stock_code")
        start, end = _q("start"), _q("end")
        if not code:
            raise ValueError("stock_code is required.")
        if not start or not end:
            raise ValueError("start and end are required.")
        return jsonify(analysis_service.get_period_analysis(code, start, end, _q("as_of")))

    # ---- reports ----------------------------------------------------------
    @app.post("/api/reports/generate")
    def report_generate():
        b = _json()
        stock = b.get("stock_code") or b.get("stock")
        if not stock:
            raise ValueError("stock_code is required.")
        rep, rec = report_manager.build_report(
            stock, b.get("report_type", "custom"), b.get("start"), b.get("end"),
            ref_date=b.get("ref_date"), as_of=b.get("as_of"))
        # Full payload (with chart series) for the UI preview.
        return jsonify({"report_id": rec["id"], "record": rec, "report": rep})

    @app.post("/api/reports/preview")
    def report_preview():
        """Analyse without storing to history (used for on-screen preview)."""
        b = _json()
        stock = b.get("stock_code") or b.get("stock")
        if not stock:
            raise ValueError("stock_code is required.")
        rep, _ = report_manager.build_report(
            stock, b.get("report_type", "custom"), b.get("start"), b.get("end"),
            ref_date=b.get("ref_date"), as_of=b.get("as_of"), store=False)
        return jsonify({"report": rep})

    @app.get("/api/reports/history")
    def report_history():
        reports = report_store.list_reports(_int_q("limit", 100), _q("stock_code"))
        return jsonify({"count": len(reports), "reports": reports})

    @app.get("/api/reports/<report_id>")
    def report_get(report_id):
        rep, rec = report_manager.get_report(report_id)
        if rec is None:
            return jsonify({"error": f"No report '{report_id}'."}), 404
        return jsonify({"record": rec, "report": rep})

    @app.post("/api/reports/<report_id>/pdf")
    def report_pdf(report_id):
        b = _json()
        qwen_summary = b.get("qwen_summary")
        if b.get("use_qwen") and not qwen_summary:
            rep, _ = report_manager.get_report(report_id)
            if rep:
                qwen_summary = qwen_agent.summarize_report(rep)
        path = report_manager.make_pdf(report_id, qwen_summary=qwen_summary)
        return jsonify({"success": True, "report_id": report_id,
                        "pdf_url": f"/api/reports/{report_id}/pdf",
                        "pdf_filename": os.path.basename(path)})

    @app.get("/api/reports/<report_id>/pdf")
    def report_pdf_download(report_id):
        _rep, rec = report_manager.get_report(report_id)
        if rec is None:
            return jsonify({"error": f"No report '{report_id}'."}), 404
        path = rec.get("pdf_path")
        if not path or not os.path.exists(path):
            path = report_manager.make_pdf(report_id)
        return send_file(path, as_attachment=request.args.get("download") == "1",
                         mimetype="application/pdf")

    @app.post("/api/reports/<report_id>/email")
    def report_email(report_id):
        b = _json()
        recipient = b.get("recipient") or b.get("email")
        if not recipient:
            return jsonify({"success": False,
                            "error": "recipient email is required."}), 400
        return jsonify(report_manager.email_report(report_id, recipient))

    # ---- Qwen agent -------------------------------------------------------
    @app.post("/api/chat")
    def chat():
        b = _json()
        message = (b.get("message") or "").strip()
        if not message:
            raise ValueError("message is required.")
        try:
            result = qwen_agent.run_agent(message, history=b.get("history"),
                                          as_of=b.get("as_of"))
        except Exception as e:
            return jsonify({
                "configured": config.qwen_configured(),
                "reply": (f"The Qwen agent could not be reached: {e}. "
                          "Check QWEN_API_KEY / QWEN_BASE_URL / QWEN_MODEL and "
                          "network access."),
                "activities": [], "tool_trace": [], "error": str(e),
            }), 200
        return jsonify(result)

    @app.get("/api/agent/tools")
    def agent_tools():
        return jsonify({"tools": [t["function"]["name"] for t in tools.TOOL_SPECS]})

    # ---- frontend ---------------------------------------------------------
    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.after_request
    def _no_cache(resp):
        # Dev-friendly: never let the browser serve stale HTML/JS/CSS.
        resp.headers["Cache-Control"] = "no-store, max-age=0"
        return resp

    return app


def _warmup():
    """Load models + prime caches in the background so first requests are fast."""
    try:
        from . import ml_service
        for s in config.STOCKS:
            try:
                ml_service.load_model(s)
                ml_service.load_params(s)
                d, c = data_service.get_close_series(s, "update3")
                ml_service.predict_vs_actual_series(s, d, c)
            except Exception:
                continue
        print("[warmup] model + cache warm-up complete")
    except Exception:
        pass


app = create_app()


def main():
    if os.getenv("WARMUP", "true").lower() in ("1", "true", "yes"):
        threading.Thread(target=_warmup, daemon=True).start()
    print(f"IDX Intelligence backend on http://{config.HOST}:{config.PORT}")
    print(f"  Qwen configured: {config.qwen_configured()} | "
          f"SMTP configured: {config.smtp_configured()}")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, threaded=True)


if __name__ == "__main__":
    main()
