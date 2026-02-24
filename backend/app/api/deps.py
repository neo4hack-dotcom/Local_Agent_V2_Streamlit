from __future__ import annotations

from fastapi import Request

from app.core.automation_engine import AutomationEngine
from app.core.models import AgentCatalog, AppSettings
from app.core.storage import JSONRepository


def get_settings_repo(request: Request) -> JSONRepository[AppSettings]:
    return request.app.state.settings_repo


def get_agents_repo(request: Request) -> JSONRepository[AgentCatalog]:
    return request.app.state.agents_repo


def get_automation_engine(request: Request) -> AutomationEngine:
    return request.app.state.automation_engine
