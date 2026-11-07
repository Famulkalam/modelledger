"""
ModelLedger: alert dispatch (log-based; email or Slack can be plugged in here)

In production, replace these stubs with actual integrations:
  - Email via SMTP or SendGrid
  - Slack via webhook URL
  - PagerDuty / OpsGenie for critical severity

All alert dispatch is logged via structured JSON logging.
"""

import json
import logging

logger = logging.getLogger(__name__)


def dispatch_alerts(alerts: list[dict]) -> None:
    """
    Dispatch fairness or integrity alerts to configured channels.

    Each alert dict has:
      - severity: 'HIGH', 'MEDIUM', 'LOW'
      - metric: e.g. 'demographic_parity', 'chain_integrity'
      - message: human-readable description

    Production replacement: send to Slack webhook, email, or PagerDuty.
    """
    for alert in alerts:
        severity = alert.get("severity", "UNKNOWN")
        metric = alert.get("metric", "unknown")
        message = alert.get("message", "")

        if severity == "HIGH":
            logger.error(
                "ALERT [%s] %s: %s", severity, metric, message,
                extra={"alert": json.dumps(alert)},
            )
        elif severity == "MEDIUM":
            logger.warning(
                "ALERT [%s] %s: %s", severity, metric, message,
                extra={"alert": json.dumps(alert)},
            )
        else:
            logger.info(
                "ALERT [%s] %s: %s", severity, metric, message,
                extra={"alert": json.dumps(alert)},
            )

    # --- Production integration stubs ---
    # _send_slack_webhook(alerts)
    # _send_email_alerts(alerts)
    # _send_pagerduty(alerts)


def _send_slack_webhook(alerts: list[dict]) -> None:
    """Stub: POST alerts to Slack incoming webhook."""
    # webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    # requests.post(webhook_url, json={"text": format_alerts(alerts)})
    pass


def _send_email_alerts(alerts: list[dict]) -> None:
    """Stub: Send alert email via SMTP / SendGrid."""
    # smtp_server = os.environ.get("SMTP_SERVER")
    # ...
    pass
