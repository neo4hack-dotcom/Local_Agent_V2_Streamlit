from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_automation_engine, get_settings_repo
from app.core.automation_engine import AutomationEngine
from app.core.models import (
    AppSettings,
    AutomationRule,
    AutomationRuleCreate,
    AutomationRuleUpdate,
    AutomationRunLog,
)
from app.core.storage import JSONRepository

router = APIRouter(prefix="/automations", tags=["automations"])


def _find_automation(settings: AppSettings, automation_id: str) -> AutomationRule:
    for automation in settings.automations:
        if automation.id == automation_id:
            return automation
    raise HTTPException(status_code=404, detail="Automation not found.")


@router.get("", response_model=list[AutomationRule])
def list_automations(
    settings_repo: JSONRepository[AppSettings] = Depends(get_settings_repo),
) -> list[AutomationRule]:
    settings = settings_repo.load()
    return settings.automations


@router.post("", response_model=AutomationRule, status_code=201)
def create_automation(
    payload: AutomationRuleCreate,
    settings_repo: JSONRepository[AppSettings] = Depends(get_settings_repo),
    automation_engine: AutomationEngine = Depends(get_automation_engine),
) -> AutomationRule:
    settings = settings_repo.load()
    automation = AutomationRule(id=uuid4().hex, **payload.model_dump())
    settings.automations.append(automation)
    settings_repo.save(settings)
    automation_engine.reset_rule_state(automation.id)
    return automation


@router.put("/{automation_id}", response_model=AutomationRule)
def update_automation(
    automation_id: str,
    payload: AutomationRuleUpdate,
    settings_repo: JSONRepository[AppSettings] = Depends(get_settings_repo),
    automation_engine: AutomationEngine = Depends(get_automation_engine),
) -> AutomationRule:
    settings = settings_repo.load()
    for index, automation in enumerate(settings.automations):
        if automation.id == automation_id:
            updated = AutomationRule(id=automation_id, **payload.model_dump())
            settings.automations[index] = updated
            settings_repo.save(settings)
            automation_engine.reset_rule_state(automation_id)
            return updated
    raise HTTPException(status_code=404, detail="Automation not found.")


@router.delete("/{automation_id}", status_code=204)
def delete_automation(
    automation_id: str,
    settings_repo: JSONRepository[AppSettings] = Depends(get_settings_repo),
    automation_engine: AutomationEngine = Depends(get_automation_engine),
) -> None:
    settings = settings_repo.load()
    _find_automation(settings, automation_id)
    settings.automations = [item for item in settings.automations if item.id != automation_id]
    settings_repo.save(settings)
    automation_engine.reset_rule_state(automation_id)


@router.get("/{automation_id}/runs", response_model=list[AutomationRunLog])
def list_automation_runs(
    automation_id: str,
    limit: int = Query(default=30, ge=1, le=200),
    settings_repo: JSONRepository[AppSettings] = Depends(get_settings_repo),
    automation_engine: AutomationEngine = Depends(get_automation_engine),
) -> list[AutomationRunLog]:
    settings = settings_repo.load()
    _find_automation(settings, automation_id)
    return automation_engine.list_runs(automation_id=automation_id, limit=limit)
