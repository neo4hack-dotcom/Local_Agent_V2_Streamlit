from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import agents, automations, config, databases
from app.core.automation_engine import AutomationEngine
from app.core.models import AgentCatalog, AgentConfig, AppSettings
from app.core.settings import AGENTS_FILE, DEFAULT_FRONTEND_ORIGIN, SETTINGS_FILE
from app.core.storage import JSONRepository



def _default_settings() -> AppSettings:
    return AppSettings()



def _default_agents() -> AgentCatalog:
    return AgentCatalog(
        agents=[
            AgentConfig(
                id="default-sql-agent",
                name="SQL Analyst",
                agent_type="sql_analyst",
                description="Configurable SQL agent for ClickHouse/Oracle.",
                template_config={
                    "database_id": "",
                    "database_name": "",
                    "sql_use_case_mode": "llm_sql",
                    "sql_query_template": "",
                    "sql_parameters": [],
                },
            )
        ]
    )


settings_repo = JSONRepository(
    path=SETTINGS_FILE,
    model_cls=AppSettings,
    default_factory=_default_settings,
)
agents_repo = JSONRepository(
    path=AGENTS_FILE,
    model_cls=AgentCatalog,
    default_factory=_default_agents,
)
automation_engine = AutomationEngine(settings_repo=settings_repo, agents_repo=agents_repo)

app = FastAPI(title="Local Agent Backend", version="0.1.0")

frontend_origin = os.getenv("FRONTEND_ORIGIN", DEFAULT_FRONTEND_ORIGIN)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.settings_repo = settings_repo
app.state.agents_repo = agents_repo
app.state.automation_engine = automation_engine


@app.on_event("startup")
def startup() -> None:
    settings_repo.load()
    agents_repo.load()
    automation_engine.start()


@app.on_event("shutdown")
def shutdown() -> None:
    automation_engine.stop()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(config.router, prefix="/api")
app.include_router(databases.router, prefix="/api")
app.include_router(agents.router, prefix="/api")
app.include_router(automations.router, prefix="/api")
