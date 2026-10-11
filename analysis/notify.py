"""Send a pre-trade report to a chat webhook (Slack, Discord or Teams-style)."""

import json
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def build_webhook_payload(url, text):
    """Discord expects ``content`` (max 2,000 chars); Slack and most others accept ``text``."""

    host = urlparse(url).netloc.lower()
    if "discord" in host:
        return {"content": text[:1990]}
    return {"text": text}


def send_webhook(url, text, timeout=15):
    """POST ``text`` to ``url``. Returns ``(ok, message)``; never raises."""

    if not url:
        return False, "No webhook URL configured (set ALERT_WEBHOOK_URL in Streamlit secrets)."
    try:
        request = Request(
            url,
            data=json.dumps(build_webhook_payload(url, text)).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 300, f"HTTP {response.status}"
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def short_summary(report):
    """Compact multi-line text suitable for a chat notification."""

    lines = [f"[{report.action}] {report.symbol} - {report.headline}"]
    plan = report.plan
    if plan is not None and plan.shares > 0:
        lines.append(
            f"Entry {plan.entry:,.2f} | Stop {plan.stop:,.2f} | Target {plan.target:,.2f} | "
            f"R/R {plan.reward_risk:.2f} | {plan.shares:,} sh | risk ${plan.dollar_risk:,.0f}"
        )
    lines += [f"+ {reason}" for reason in report.reasons_for[:4]]
    lines += [f"- {reason}" for reason in (report.reasons_against + report.warnings)[:4]]
    return "\n".join(lines)
