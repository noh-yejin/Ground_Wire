from __future__ import annotations

import requests

from app.config import settings
from app.models import Issue, IssueStatus


def send_issue(issue: Issue) -> dict:
    payload = _build_payload(issue)

    if issue.status != IssueStatus.READY:
        return {"sent": False, "reason": "Issue is on hold", "payload": payload}

    if not settings.slack_webhook_url:
        return {"sent": False, "reason": "SLACK_WEBHOOK_URL is not configured", "payload": payload}

    response = requests.post(settings.slack_webhook_url, json=payload, timeout=10)
    response.raise_for_status()
    return {"sent": True, "payload": payload}


def _build_payload(issue: Issue) -> dict:
    evidence_lines = [
        f"- {item.source}: {item.quote} ({item.url})"
        for item in issue.evidence[:4]
    ]
    text = (
        f"*[{issue.status.value}] {issue.topic}*\n"
        f"신뢰도: {issue.reliability.value}\n"
        f"감정: {issue.analysis.sentiment.value}\n"
        f"요약: {issue.analysis.summary}\n"
        f"근거:\n" + "\n".join(evidence_lines)
    )
    return {"text": text}
