from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any

import requests

from .models import WebhookConfig

MAX_WEBHOOK_TIMELINE_EVENTS = 400


class WebhookDispatcher:
    def __init__(self, config: WebhookConfig) -> None:
        self.config = config

    def send_manager_event(
        self,
        *,
        run_id: str,
        run_context: dict[str, Any],
        event: dict[str, Any],
        sequence: int,
        timeline: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": "1.0",
            "source": "local_agent_studio",
            "kind": "manager_event",
            "sent_at": datetime.now(timezone.utc).isoformat(),
            "sequence": sequence,
            "run": {
                "run_id": run_id,
                **run_context,
            },
            "is_final": str(event.get("type", "")).strip() == "manager_final",
            "event": event,
        }

        if (
            payload["is_final"]
            and self.config.include_timeline_on_final
            and isinstance(timeline, list)
        ):
            payload["timeline"] = timeline[-MAX_WEBHOOK_TIMELINE_EVENTS:]

        return self._post_json(payload=payload, force_send=False)

    def send_test_event(self) -> dict[str, Any]:
        payload = {
            "schema_version": "1.0",
            "source": "local_agent_studio",
            "kind": "webhook_test",
            "sent_at": datetime.now(timezone.utc).isoformat(),
            "event": {
                "type": "webhook_test",
                "message": "Webhook test from Local Agent Studio.",
                "ts": datetime.now(timezone.utc).isoformat(),
            },
            "run": {
                "run_id": "webhook-test",
                "channel": "configuration",
            },
            "is_final": True,
        }
        return self._post_json(payload=payload, force_send=True)

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "User-Agent": "Local-Agent-Studio/1.0",
        }

        if isinstance(self.config.headers, dict):
            for key, value in self.config.headers.items():
                header_name = str(key).strip()
                if not header_name:
                    continue
                headers[header_name] = str(value)

        token = str(self.config.auth_token or "").strip()
        if token and not any(name.lower() == "authorization" for name in headers):
            headers["Authorization"] = f"Bearer {token}"

        return headers

    def _post_json(self, *, payload: dict[str, Any], force_send: bool) -> dict[str, Any]:
        target_url = str(self.config.url or "").strip()
        if not target_url:
            return {
                "ok": False,
                "message": "Webhook URL is empty.",
                "url": "",
                "status_code": None,
                "latency_ms": None,
                "response_preview": None,
            }

        if not force_send and not self.config.enabled:
            return {
                "ok": False,
                "message": "Webhook is disabled.",
                "url": target_url,
                "status_code": None,
                "latency_ms": None,
                "response_preview": None,
            }

        started = perf_counter()
        try:
            response = requests.post(
                target_url,
                json=payload,
                headers=self._build_headers(),
                timeout=float(self.config.timeout_seconds),
                verify=bool(self.config.verify_ssl),
            )
            latency_ms = round((perf_counter() - started) * 1000)
            status_code = int(response.status_code)
            text = (response.text or "").strip()
            response_preview = text[:600] if text else None
            ok = 200 <= status_code < 300
            message = (
                f"Webhook responded with HTTP {status_code}."
                if ok
                else f"Webhook responded with HTTP {status_code}."
            )
            return {
                "ok": ok,
                "message": message,
                "url": target_url,
                "status_code": status_code,
                "latency_ms": latency_ms,
                "response_preview": response_preview,
            }
        except requests.RequestException as exc:
            latency_ms = round((perf_counter() - started) * 1000)
            return {
                "ok": False,
                "message": str(exc),
                "url": target_url,
                "status_code": None,
                "latency_ms": latency_ms,
                "response_preview": None,
            }
