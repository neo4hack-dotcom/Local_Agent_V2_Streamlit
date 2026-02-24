"""
Configuration page for the VSCode simple runner.

Edit this file directly in VSCode, then run:
    python vscode_simple_runner/run_vscode_agent_app.py

This file is intentionally verbose and heavily commented so you can quickly
understand all available options for:
1) Local LLM
2) Database connections
3) Agent manager
4) Agents

IMPORTANT
- Keep this file in valid Python syntax.
- Prefer environment variables for secrets (passwords, API keys).
- Each object below is validated at runtime by the backend pydantic models.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any


# -----------------------------------------------------------------------------
# Path helpers
# -----------------------------------------------------------------------------

# Used for file-based agents defaults.
HOME_DIR = Path.home()
DEFAULT_DATA_DIR = str(HOME_DIR / "Downloads")


# -----------------------------------------------------------------------------
# LLM CONFIGURATION
# -----------------------------------------------------------------------------
# Supported providers:
# - "ollama": local Ollama server (recommended for local usage)
# - "http": custom HTTP endpoint exposing compatible generate/model APIs
#
# Common options:
# - model: model name shown by provider (example: "llama3.1")
# - system_prompt: global prompt applied to manager + all agents
# - timeout_seconds: request timeout
# - temperature: creativity (0.0 deterministic, 1.0+ more creative)
# - headers: custom headers if needed
#
# Ollama-specific:
# - base_url: usually "http://localhost:11434"
#
# HTTP-specific:
# - endpoint: generation endpoint (optional if base_url already points to generation route)
# - api_key: optional bearer token
LLM_CONFIG: dict[str, Any] = {
    "provider": "ollama",
    "model": "llama3.1",
    "system_prompt": "",
    "base_url": os.getenv("LOCAL_AGENT_LLM_BASE_URL", "http://localhost:11434"),
    "endpoint": os.getenv("LOCAL_AGENT_LLM_ENDPOINT", None),
    "api_key": os.getenv("LOCAL_AGENT_LLM_API_KEY", None),
    "timeout_seconds": 60,
    "temperature": 0.1,
    "headers": {},
}


# -----------------------------------------------------------------------------
# AGENT MANAGER CONFIGURATION
# -----------------------------------------------------------------------------
# max_steps:
# - maximum orchestration rounds for manager reasoning.
# - each round can call one or more agents.
#
# max_agent_calls:
# - hard budget of total executed agent calls for one manager run.
MANAGER_CONFIG: dict[str, Any] = {
    "max_steps": 6,
    "max_agent_calls": 12,
}


# -----------------------------------------------------------------------------
# DATABASE CONNECTIONS
# -----------------------------------------------------------------------------
# You can define multiple profiles and route agents to one profile by:
# - template_config.database_id
# - or template_config.database_name
#
# Engines:
# - clickhouse
# - oracle
# - elasticsearch
#
# Fields:
# - id: unique technical key (required)
# - name: display name (required)
# - engine: one of clickhouse/oracle/elasticsearch (required)
# - host/port/database/username/password/dsn/secure/options as needed
#
# Notes:
# - Oracle can use "dsn" directly.
# - Elasticsearch advanced auth can be placed in options:
#   {"api_key":"...", "verify_ssl":true}
DATABASES: list[dict[str, Any]] = [
    {
        "id": "clickhouse_local",
        "name": "ClickHouse Local",
        "engine": "clickhouse",
        "host": os.getenv("CH_HOST", "localhost"),
        "port": int(os.getenv("CH_PORT", "8123")),
        "database": os.getenv("CH_DATABASE", "default"),
        "username": os.getenv("CH_USER", "default"),
        "password": os.getenv("CH_PASSWORD", ""),
        "dsn": None,
        "secure": False,
        "options": {},
    },
    # Example Oracle profile (disabled by default: fill then uncomment if needed)
    # {
    #     "id": "oracle_main",
    #     "name": "Oracle Main",
    #     "engine": "oracle",
    #     "host": os.getenv("ORACLE_HOST", ""),
    #     "port": int(os.getenv("ORACLE_PORT", "1521")),
    #     "database": os.getenv("ORACLE_SERVICE", ""),
    #     "username": os.getenv("ORACLE_USER", ""),
    #     "password": os.getenv("ORACLE_PASSWORD", ""),
    #     "dsn": os.getenv("ORACLE_DSN", None),
    #     "secure": False,
    #     "options": {},
    # },
    # Example Elasticsearch profile
    # {
    #     "id": "es_local",
    #     "name": "Elasticsearch Local",
    #     "engine": "elasticsearch",
    #     "host": os.getenv("ES_HOST", "http://localhost:9200"),
    #     "port": int(os.getenv("ES_PORT", "9200")),
    #     "database": os.getenv("ES_INDEX", ""),
    #     "username": os.getenv("ES_USER", ""),
    #     "password": os.getenv("ES_PASSWORD", ""),
    #     "dsn": None,
    #     "secure": False,
    #     "options": {
    #         "api_key": os.getenv("ES_API_KEY", ""),
    #         "verify_ssl": False,
    #     },
    # },
]

# Default active DB if agent routing does not specify one.
ACTIVE_DATABASE_ID: str | None = "clickhouse_local"


# -----------------------------------------------------------------------------
# AGENT DEFINITIONS
# -----------------------------------------------------------------------------
# Strategy:
# - Start from official backend template defaults.
# - Override only what you need.
#
# Main top-level fields per agent:
# - id: unique technical id
# - name: display name
# - agent_type: template type (already set by template defaults)
# - description: used by manager routing
# - enabled: manager can only call enabled agents
# - max_rows: output row limit (or chunks/docs depending on agent type)
# - template_config: agent-specific behavior
#
# Common template_config usage:
# - SQL agents:
#   database_id / database_name / sql_use_case_mode / sql_query_template / sql_parameters
# - File/RAG agents:
#   folder_path / file_extensions / max_files / chunk params
# - Web agents:
#   URLs, crawler depth, timeouts, language, etc.
# - Table manager:
#   safety policy booleans (protect_existing_tables, allow_row_updates, ...)
def _template_defaults(template_id: str) -> dict[str, Any]:
    """
    Import lazily to keep this config page self-contained.
    """
    import sys

    backend_root = Path(__file__).resolve().parents[1] / "backend"
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

    from app.core.agent_templates import template_defaults

    return template_defaults(template_id).model_dump()


def build_agent(
    *,
    template_id: str,
    agent_id: str,
    name: str,
    description: str | None = None,
    enabled: bool = True,
    max_rows: int | None = None,
    template_config_overrides: dict[str, Any] | None = None,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """
    Helper to create a fully valid agent payload from template defaults.
    """
    payload = deepcopy(_template_defaults(template_id))
    payload["id"] = agent_id
    payload["name"] = name
    payload["enabled"] = enabled

    if description is not None:
        payload["description"] = description
    if max_rows is not None:
        payload["max_rows"] = max_rows
    if system_prompt is not None:
        payload["system_prompt"] = system_prompt

    if template_config_overrides:
        base_cfg = payload.get("template_config", {})
        if not isinstance(base_cfg, dict):
            base_cfg = {}
        payload["template_config"] = {**base_cfg, **template_config_overrides}

    return payload


# -----------------------------------------------------------------------------
# ENABLED AGENTS FOR THIS SIMPLE RUNNER
# -----------------------------------------------------------------------------
# You can add/remove agents freely.
# Keep ids unique.
AGENTS: list[dict[str, Any]] = [
    build_agent(
        template_id="sql_analyst",
        agent_id="sql_analyst_main",
        name="SQL Analyst Main",
        description="General SQL analytics on ClickHouse/Oracle.",
        template_config_overrides={
            "database_id": "clickhouse_local",
            "sql_use_case_mode": "llm_sql",
        },
    ),
    build_agent(
        template_id="clickhouse_table_manager",
        agent_id="clickhouse_table_manager_safe",
        name="ClickHouse Table Manager (Safe)",
        description="Create/update data with table protection enabled.",
        template_config_overrides={
            "database_id": "clickhouse_local",
            "protect_existing_tables": True,
            "allow_row_inserts": True,
            "allow_row_updates": True,
            "allow_row_deletes": False,
        },
    ),
    build_agent(
        template_id="file_assistant",
        agent_id="file_assistant_docs",
        name="File Assistant",
        description="Read and answer from local documents.",
        template_config_overrides={
            "folder_path": DEFAULT_DATA_DIR,
            "file_extensions": [".txt", ".md", ".json", ".csv", ".log"],
            "max_files": 40,
            "top_k": 6,
        },
    ),
    build_agent(
        template_id="rag_context",
        agent_id="rag_context_docs",
        name="RAG Context",
        description="Retrieve context chunks from local files.",
        template_config_overrides={
            "folder_path": DEFAULT_DATA_DIR,
            "file_extensions": [".txt", ".md", ".json", ".csv"],
            "top_k_chunks": 6,
            "chunk_size": 1200,
            "chunk_overlap": 150,
        },
    ),
    build_agent(
        template_id="web_scraper",
        agent_id="web_scraper_generic",
        name="Web Scraper",
        description="HTTP scraping agent for generic sites.",
        template_config_overrides={
            "start_urls": [],
            "search_fallback": True,
            "follow_links": False,
            "same_domain_only": True,
            "max_pages": 3,
        },
    ),
    build_agent(
        template_id="wikipedia_retriever",
        agent_id="wikipedia_agent",
        name="Wikipedia Agent",
        description="Fetch facts from Wikipedia pages.",
        template_config_overrides={
            "language": "fr",
            "top_k": 5,
            "summary_sentences": 2,
        },
    ),
    build_agent(
        template_id="rss_news",
        agent_id="rss_news_briefing",
        name="RSS News Briefing",
        description="Short news briefing from configured RSS feeds.",
        template_config_overrides={
            "language_hint": "fr",
            "top_k": 5,
        },
    ),
]


# -----------------------------------------------------------------------------
# ADVANCED NOTES
# -----------------------------------------------------------------------------
# 1) To create a specialized SQL use-case agent:
#    - set sql_use_case_mode="parameterized_template"
#    - set sql_query_template (with {{parameter_name}} placeholders)
#    - set sql_parameters list with type/required/description
#
# 2) To force manager routing to one DB profile for an agent:
#    - set template_config.database_id = "your_profile_id"
#
# 3) To disable an agent without removing it:
#    - set enabled=False in that agent payload.
#
# 4) Keep this file versioned in git to track config history.
# -----------------------------------------------------------------------------
