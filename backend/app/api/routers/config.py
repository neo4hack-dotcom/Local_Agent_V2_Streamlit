from __future__ import annotations

import socket
from datetime import datetime, timezone
from time import perf_counter

import requests
from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_settings_repo
from app.core.llm_client import LLMClient
from app.core.models import AppSettings, LLMConfig, ManagerConfig, WebhookConfig
from app.core.storage import JSONRepository
from app.core.webhook_dispatcher import WebhookDispatcher

router = APIRouter(prefix="/config", tags=["config"])


@router.get("", response_model=AppSettings)
def get_config(
    settings_repo: JSONRepository[AppSettings] = Depends(get_settings_repo),
) -> AppSettings:
    return settings_repo.load()


@router.get("/llm", response_model=LLMConfig)
def get_llm_config(
    settings_repo: JSONRepository[AppSettings] = Depends(get_settings_repo),
) -> LLMConfig:
    settings = settings_repo.load()
    return settings.llm


@router.put("/llm", response_model=LLMConfig)
def update_llm_config(
    llm_config: LLMConfig,
    settings_repo: JSONRepository[AppSettings] = Depends(get_settings_repo),
) -> LLMConfig:
    settings = settings_repo.load()
    settings.llm = llm_config
    settings_repo.save(settings)
    return settings.llm


@router.get("/manager", response_model=ManagerConfig)
def get_manager_config(
    settings_repo: JSONRepository[AppSettings] = Depends(get_settings_repo),
) -> ManagerConfig:
    settings = settings_repo.load()
    return settings.manager


@router.put("/manager", response_model=ManagerConfig)
def update_manager_config(
    manager_config: ManagerConfig,
    settings_repo: JSONRepository[AppSettings] = Depends(get_settings_repo),
) -> ManagerConfig:
    settings = settings_repo.load()
    settings.manager = manager_config
    settings_repo.save(settings)
    return settings.manager


@router.get("/webhook", response_model=WebhookConfig)
def get_webhook_config(
    settings_repo: JSONRepository[AppSettings] = Depends(get_settings_repo),
) -> WebhookConfig:
    settings = settings_repo.load()
    return settings.webhook


@router.put("/webhook", response_model=WebhookConfig)
def update_webhook_config(
    webhook_config: WebhookConfig,
    settings_repo: JSONRepository[AppSettings] = Depends(get_settings_repo),
) -> WebhookConfig:
    settings = settings_repo.load()
    settings.webhook = webhook_config
    settings_repo.save(settings)
    return settings.webhook


@router.post("/webhook/test")
def test_webhook_delivery(webhook_config: WebhookConfig) -> dict:
    dispatcher = WebhookDispatcher(webhook_config)
    result = dispatcher.send_test_event()
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=str(result.get("message") or "Webhook test failed."))
    return {
        "status": "ok",
        "message": str(result.get("message") or "Webhook test succeeded."),
        "url": result.get("url"),
        "status_code": result.get("status_code"),
        "latency_ms": result.get("latency_ms"),
        "response_preview": result.get("response_preview"),
        "tested_at": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/llm/test")
def test_llm_connection(
    settings_repo: JSONRepository[AppSettings] = Depends(get_settings_repo),
) -> dict:
    settings = settings_repo.load()
    client = LLMClient(settings.llm)
    try:
        return client.test_connection()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/llm/models")
def list_llm_models(
    settings_repo: JSONRepository[AppSettings] = Depends(get_settings_repo),
) -> dict:
    settings = settings_repo.load()
    client = LLMClient(settings.llm)
    try:
        models = client.list_models()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "provider": settings.llm.provider,
        "models": models,
    }


@router.post("/network/test")
def test_network_access() -> dict:
    checks: list[dict] = []

    dns_start = perf_counter()
    try:
        resolved_ip = socket.gethostbyname("example.com")
        checks.append(
            {
                "name": "dns_lookup",
                "target": "example.com",
                "ok": True,
                "detail": f"Resolved example.com to {resolved_ip}",
                "status_code": None,
                "latency_ms": round((perf_counter() - dns_start) * 1000),
            }
        )
    except OSError as exc:
        checks.append(
            {
                "name": "dns_lookup",
                "target": "example.com",
                "ok": False,
                "detail": str(exc),
                "status_code": None,
                "latency_ms": round((perf_counter() - dns_start) * 1000),
            }
        )

    http_targets = [
        ("https_probe", "https://example.com"),
        ("api_probe", "https://api.github.com"),
        ("wikipedia_probe", "https://www.wikipedia.org"),
    ]
    headers = {"User-Agent": "Local-Agent-Studio/1.0"}

    for name, target in http_targets:
        started = perf_counter()
        try:
            response = requests.get(target, headers=headers, timeout=8)
            status_code = int(response.status_code)
            ok = 200 <= status_code < 500
            checks.append(
                {
                    "name": name,
                    "target": target,
                    "ok": ok,
                    "detail": f"HTTP status {status_code}",
                    "status_code": status_code,
                    "latency_ms": round((perf_counter() - started) * 1000),
                }
            )
        except requests.RequestException as exc:
            checks.append(
                {
                    "name": name,
                    "target": target,
                    "ok": False,
                    "detail": str(exc),
                    "status_code": None,
                    "latency_ms": round((perf_counter() - started) * 1000),
                }
            )

    success_count = sum(1 for check in checks if bool(check.get("ok")))
    total_count = len(checks)
    if total_count == 0 or success_count == 0:
        status = "blocked"
        message = "No outbound internet connectivity detected from backend."
    elif success_count == total_count:
        status = "ok"
        message = "Internet connectivity is available from backend."
    else:
        status = "partial"
        message = "Internet connectivity is partially available from backend."

    return {
        "status": status,
        "message": message,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "successful_checks": success_count,
        "total_checks": total_count,
        "checks": checks,
    }
