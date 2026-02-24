from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

import requests
from fastapi import FastAPI, HTTPException, Request

app = FastAPI(
    title="Local Agent Studio Open WebUI Bridge",
    version="0.1.0",
    description=(
        "Receives Local Agent Studio webhook events and forwards a normalized message "
        "to an Open WebUI webhook endpoint."
    ),
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw.strip())
    except ValueError:
        return default
    if value <= 0:
        return default
    return value


def _env_int(name: str, default: int, minimum: int = 1, maximum: int = 100000) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def _truncate(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _target_url() -> str:
    return str(os.getenv("OPEN_WEBUI_WEBHOOK_URL", "")).strip()


def _build_forward_headers() -> dict[str, str]:
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "User-Agent": "Local-Agent-Studio-OpenWebUI-Bridge/1.0",
    }

    auth_token = str(os.getenv("OPEN_WEBUI_AUTH_TOKEN", "")).strip()
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    extra_headers_raw = str(os.getenv("OPEN_WEBUI_EXTRA_HEADERS_JSON", "")).strip()
    if not extra_headers_raw:
        return headers

    try:
        extra_headers = json.loads(extra_headers_raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "OPEN_WEBUI_EXTRA_HEADERS_JSON is invalid JSON."
        ) from exc

    if not isinstance(extra_headers, dict):
        raise ValueError("OPEN_WEBUI_EXTRA_HEADERS_JSON must be a JSON object.")

    for key, value in extra_headers.items():
        name = str(key).strip()
        if not name:
            continue
        headers[name] = str(value)

    return headers


def _event_preview(event: dict[str, Any]) -> list[str]:
    event_type = str(event.get("type") or "unknown").strip()
    lines: list[str] = []

    if event_type == "manager_start":
        lines.append(
            f"Question: {_truncate(event.get('question', ''), 500)}"
        )
    elif event_type == "manager_decision":
        lines.append(f"Rationale: {_truncate(event.get('rationale', ''), 240)}")
        calls = event.get("calls")
        if isinstance(calls, list) and calls:
            lines.append("Planned calls:")
            for call in calls[:4]:
                if not isinstance(call, dict):
                    continue
                agent_id = _truncate(call.get("agent_id", "unknown"), 80)
                question = _truncate(call.get("question", ""), 180)
                lines.append(f"- {agent_id}: {question}")
    elif event_type == "manager_warning":
        lines.append(f"Warning: {_truncate(event.get('message', ''), 300)}")
    elif event_type == "agent_call_started":
        lines.append(
            "Agent call started: "
            f"{_truncate(event.get('agent_name', event.get('agent_id', 'agent')), 80)} "
            f"({_truncate(event.get('agent_type', 'unknown'), 50)})"
        )
        lines.append(f"Task: {_truncate(event.get('question', ''), 220)}")
    elif event_type == "agent_call_completed":
        lines.append(
            "Agent call completed: "
            f"{_truncate(event.get('agent_name', event.get('agent_id', 'agent')), 80)} "
            f"({_truncate(event.get('agent_type', 'unknown'), 50)})"
        )
        lines.append(f"Rows: {event.get('row_count', 0)}")
        sql_text = _truncate(event.get("sql", ""), 260)
        if sql_text:
            lines.append(f"SQL: {sql_text}")
        answer_text = _truncate(event.get("answer", ""), 260)
        if answer_text:
            lines.append(f"Answer preview: {answer_text}")
    elif event_type == "agent_call_failed":
        lines.append(
            "Agent call failed: "
            f"{_truncate(event.get('agent_name', event.get('agent_id', 'agent')), 80)} "
            f"({_truncate(event.get('agent_type', 'unknown'), 50)})"
        )
        lines.append(f"Error: {_truncate(event.get('error', ''), 320)}")
    elif event_type == "agent_marked_unavailable":
        lines.append(
            "Agent marked unavailable: "
            f"{_truncate(event.get('agent_name', event.get('agent_id', 'agent')), 80)}"
        )
        lines.append(f"Reason: {_truncate(event.get('reason', ''), 320)}")
    elif event_type == "manager_final":
        lines.append(f"Status: {_truncate(event.get('status', ''), 40)}")
        lines.append(f"Final answer: {_truncate(event.get('answer', ''), 700)}")
        manager_summary = _truncate(event.get("manager_summary", ""), 1000)
        if manager_summary:
            lines.append("Manager summary:")
            lines.append(manager_summary)
        judge_verdict = event.get("judge_verdict")
        if judge_verdict:
            confidence = event.get("judge_confidence")
            lines.append(f"Judge: {judge_verdict} ({confidence}%)")
        missing = _truncate(event.get("missing_information", ""), 420)
        if missing:
            lines.append(f"Missing info: {missing}")
    else:
        lines.append(_truncate(json.dumps(event, ensure_ascii=False), 900))

    return lines


def _timeline_summary(incoming: dict[str, Any]) -> list[str]:
    if not _env_bool("OPEN_WEBUI_BRIDGE_INCLUDE_TIMELINE_SUMMARY", default=True):
        return []
    timeline = incoming.get("timeline")
    if not isinstance(timeline, list) or not timeline:
        return []

    by_type: dict[str, int] = {}
    for item in timeline:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "unknown").strip() or "unknown"
        by_type[item_type] = by_type.get(item_type, 0) + 1

    if not by_type:
        return []

    chunks = [f"{name}={count}" for name, count in sorted(by_type.items())]
    return [f"Timeline: {len(timeline)} event(s) | " + ", ".join(chunks)]


def _build_openwebui_message(incoming: dict[str, Any]) -> str:
    run = incoming.get("run")
    run_data = run if isinstance(run, dict) else {}
    event = incoming.get("event")
    event_data = event if isinstance(event, dict) else {}

    event_type = str(event_data.get("type") or "unknown").strip()
    run_id = str(run_data.get("run_id") or "unknown").strip()
    channel = str(run_data.get("channel") or "unknown").strip()
    step = event_data.get("step")

    title = f"[Local Agent] {event_type}"
    if step is not None:
        title += f" | step {step}"

    lines = [
        title,
        f"Run ID: {run_id}",
        f"Channel: {channel}",
    ]
    lines.extend(_event_preview(event_data))
    lines.extend(_timeline_summary(incoming))

    if _env_bool("OPEN_WEBUI_BRIDGE_INCLUDE_RAW_JSON", default=False):
        max_chars = _env_int("OPEN_WEBUI_BRIDGE_RAW_JSON_MAX_CHARS", default=3500)
        raw_json = _truncate(json.dumps(incoming, ensure_ascii=False), max_chars)
        lines.append("")
        lines.append("Raw payload:")
        lines.append("```json")
        lines.append(raw_json)
        lines.append("```")

    return "\n".join([line for line in lines if str(line).strip() != ""]).strip()


def _forward_to_openwebui(incoming: dict[str, Any]) -> dict[str, Any]:
    url = _target_url()
    if not url:
        raise HTTPException(
            status_code=500,
            detail="OPEN_WEBUI_WEBHOOK_URL is empty. Configure the target webhook URL.",
        )

    message = _build_openwebui_message(incoming)
    event_data = incoming.get("event")
    event = event_data if isinstance(event_data, dict) else {}
    run_data = incoming.get("run")
    run = run_data if isinstance(run_data, dict) else {}

    outgoing_payload: dict[str, Any] = {
        "content": message,
        "text": message,
        "message": message,
        "role": "assistant",
        "metadata": {
            "source": "local_agent_studio_bridge",
            "schema_version": incoming.get("schema_version", "1.0"),
            "kind": incoming.get("kind", "manager_event"),
            "event_type": event.get("type"),
            "run_id": run.get("run_id"),
            "channel": run.get("channel"),
            "is_final": bool(incoming.get("is_final")),
            "forwarded_at": datetime.now(timezone.utc).isoformat(),
        },
    }

    timeout_seconds = _env_float("OPEN_WEBUI_BRIDGE_TIMEOUT_SECONDS", default=15.0)
    verify_ssl = _env_bool("OPEN_WEBUI_BRIDGE_VERIFY_SSL", default=True)

    started = perf_counter()
    try:
        response = requests.post(
            url,
            json=outgoing_payload,
            headers=_build_forward_headers(),
            timeout=timeout_seconds,
            verify=verify_ssl,
        )
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Forward to Open WebUI failed: {exc}",
        ) from exc

    latency_ms = round((perf_counter() - started) * 1000)
    status_code = int(response.status_code)
    if not (200 <= status_code < 300):
        preview = _truncate(response.text, 500)
        raise HTTPException(
            status_code=502,
            detail=(
                f"Open WebUI responded with HTTP {status_code}. "
                f"Response preview: {preview}"
            ),
        )

    return {
        "status": "forwarded",
        "target_url": url,
        "status_code": status_code,
        "latency_ms": latency_ms,
        "event_type": event.get("type"),
        "run_id": run.get("run_id"),
        "is_final": bool(incoming.get("is_final")),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "target_configured": bool(_target_url()),
        "target_url": _target_url() or None,
        "verify_ssl": _env_bool("OPEN_WEBUI_BRIDGE_VERIFY_SSL", default=True),
        "timeout_seconds": _env_float("OPEN_WEBUI_BRIDGE_TIMEOUT_SECONDS", default=15.0),
        "include_raw_json": _env_bool("OPEN_WEBUI_BRIDGE_INCLUDE_RAW_JSON", default=False),
    }


@app.post("/webhook/agent-events")
async def relay_agent_events(request: Request) -> dict[str, Any]:
    try:
        incoming = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {exc}") from exc

    if not isinstance(incoming, dict):
        raise HTTPException(status_code=400, detail="Payload must be a JSON object.")

    return _forward_to_openwebui(incoming)


@app.post("/webhook/test")
def relay_test_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Payload must be a JSON object.")
    return _forward_to_openwebui(payload)
