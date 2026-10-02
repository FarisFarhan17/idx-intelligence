"""Email delivery for generated reports (spec section 17).

Credentials come only from environment variables. If SMTP is not configured the
service returns a clear, actionable error instead of failing silently. The agent
can trigger this via tools, and the recipient is always supplied by the user.
"""
from __future__ import annotations

import re
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
from typing import List, Optional

from . import config

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def is_valid_email(addr: str) -> bool:
    return bool(addr and _EMAIL_RE.match(addr.strip()))


def send_email(recipient: str, subject: str, body_text: str,
               attachments: Optional[List[str]] = None,
               body_html: Optional[str] = None) -> dict:
    if not config.smtp_configured():
        return {
            "success": False,
            "error": "SMTP is not configured. Set SMTP_HOST, SMTP_PORT, "
                     "SMTP_USERNAME, SMTP_PASSWORD and SMTP_FROM_EMAIL in the "
                     "environment / .env file, then retry.",
        }
    if not is_valid_email(recipient):
        return {"success": False, "error": f"Invalid recipient email address: '{recipient}'."}

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"{config.SMTP_FROM_NAME} <{config.SMTP_FROM_EMAIL}>"
    msg["To"] = recipient
    msg.set_content(body_text)
    if body_html:
        msg.add_alternative(body_html, subtype="html")

    for path in (attachments or []):
        p = Path(path)
        if not p.exists():
            return {"success": False, "error": f"Attachment not found: {p.name}"}
        msg.add_attachment(p.read_bytes(), maintype="application",
                           subtype="pdf", filename=p.name)

    try:
        if config.SMTP_USE_TLS:
            with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=30) as s:
                s.starttls(context=ssl.create_default_context())
                if config.SMTP_USERNAME and config.SMTP_PASSWORD:
                    s.login(config.SMTP_USERNAME, config.SMTP_PASSWORD)
                s.send_message(msg)
        else:
            with smtplib.SMTP_SSL(config.SMTP_HOST, config.SMTP_PORT, timeout=30,
                                  context=ssl.create_default_context()) as s:
                if config.SMTP_USERNAME and config.SMTP_PASSWORD:
                    s.login(config.SMTP_USERNAME, config.SMTP_PASSWORD)
                s.send_message(msg)
        return {"success": True, "recipient": recipient,
                "message": f"Report emailed to {recipient}."}
    except smtplib.SMTPAuthenticationError:
        return {"success": False, "error": "SMTP authentication failed (check username/password)."}
    except Exception as e:
        return {"success": False, "error": f"Email send failed: {e}"}


def send_report_email(recipient: str, report: dict, pdf_path: str) -> dict:
    stock = report.get("stock", "")
    per = report.get("period", {})
    subject = f"IDX Intelligence - {stock} {report.get('report_type','')} report ({per.get('start')} to {per.get('end')})"
    body = (
        f"Hello,\n\n"
        f"Please find attached the {report.get('report_type','')} analysis report for {stock} "
        f"({report.get('stock_name','')}).\n\n"
        f"Period: {per.get('start')} to {per.get('end')}\n"
        f"Generated: {report.get('generated_at','')}\n"
        f"Latest dataset date: {report.get('latest_dataset_date','')}\n\n"
        f"Summary:\n{report.get('summary_text','')}\n\n"
        f"--\nIDX Intelligence Platform (historical dataset; not a live IDX feed).\n"
    )
    return send_email(recipient, subject, body, attachments=[pdf_path])
