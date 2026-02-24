#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
CONFIG_PAGE_PATH = PROJECT_ROOT / "vscode_simple_runner" / "config_page.py"
RUNTIME_STATE_PATH = PROJECT_ROOT / "vscode_simple_runner" / "runtime_config_v2.json"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from pydantic import ValidationError

from app.core.agent_executor import AgentExecutor
from app.core.agent_templates import list_agent_templates, template_defaults
from app.core.db_connectors import connector_for
from app.core.database_routing import resolve_database_for_agent
from app.core.llm_client import LLMClient
from app.core.models import (
    AgentConfig,
    ConversationTurn,
    DatabaseProfile,
    LLMConfig,
    ManagerConfig,
    ManagerRunRequest,
)
from app.core.multi_agent_manager import MultiAgentManager


MAX_MEMORY_TURNS = 12
SUPPORTED_AGENT_TYPES = [
    "sql_analyst",
    "clickhouse_table_manager",
    "unstructured_to_structured",
    "email_cleaner",
    "file_assistant",
    "text_file_manager",
    "excel_manager",
    "word_manager",
    "elasticsearch_retriever",
    "internet_search",
    "rss_news",
    "web_scraper",
    "web_navigator",
    "wikipedia_retriever",
    "rag_context",
]
SUPPORTED_ENGINES = ["clickhouse", "oracle", "elasticsearch"]


@dataclass
class RuntimeConfig:
    llm: LLMConfig
    manager: ManagerConfig
    databases: list[DatabaseProfile]
    active_database_id: str | None
    agents: list[AgentConfig]


def _load_config_module() -> Any:
    if not CONFIG_PAGE_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PAGE_PATH}")

    module_name = "vscode_simple_runner_config_page"
    spec = importlib.util.spec_from_file_location(module_name, CONFIG_PAGE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load config page: {CONFIG_PAGE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state_from_config_page() -> dict[str, Any]:
    module = _load_config_module()
    llm = getattr(module, "LLM_CONFIG", {})
    manager = getattr(module, "MANAGER_CONFIG", {})
    databases = getattr(module, "DATABASES", [])
    active_database_id = getattr(module, "ACTIVE_DATABASE_ID", None)
    agents = getattr(module, "AGENTS", [])
    return {
        "llm": copy.deepcopy(llm),
        "manager": copy.deepcopy(manager),
        "databases": copy.deepcopy(databases),
        "active_database_id": active_database_id,
        "agents": copy.deepcopy(agents),
    }


def _load_runtime_state() -> dict[str, Any]:
    if RUNTIME_STATE_PATH.exists():
        try:
            payload = json.loads(RUNTIME_STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:  # noqa: BLE001
            pass
    return _state_from_config_page()


def _save_runtime_state(state: dict[str, Any]) -> None:
    RUNTIME_STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _validate_runtime_state(state: dict[str, Any]) -> RuntimeConfig:
    llm = LLMConfig.model_validate(state.get("llm", {}))
    manager = ManagerConfig.model_validate(state.get("manager", {}))
    databases = [
        DatabaseProfile.model_validate(item)
        for item in state.get("databases", [])
    ]
    agents = [AgentConfig.model_validate(item) for item in state.get("agents", [])]
    active_database_id = state.get("active_database_id")
    if active_database_id:
        known = {db.id for db in databases}
        if active_database_id not in known:
            raise ValueError(
                f"active_database_id='{active_database_id}' not found in database ids={sorted(known)}"
            )
    return RuntimeConfig(
        llm=llm,
        manager=manager,
        databases=databases,
        active_database_id=active_database_id,
        agents=agents,
    )


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _parse_json_text(value: str, label: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} contains invalid JSON: {exc}") from exc


def _new_db_profile() -> dict[str, Any]:
    return {
        "id": "db_new",
        "name": "New Database",
        "engine": "clickhouse",
        "host": "localhost",
        "port": 8123,
        "database": "default",
        "username": "",
        "password": "",
        "dsn": None,
        "secure": False,
        "options": {},
    }


def _new_agent_from_template(template_id: str, agent_id: str, name: str) -> dict[str, Any]:
    payload = template_defaults(template_id).model_dump()
    payload["id"] = agent_id
    payload["name"] = name
    return payload


def _init_session_state() -> None:
    if "runner_state" not in st.session_state:
        st.session_state.runner_state = _load_runtime_state()
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "single_result" not in st.session_state:
        st.session_state.single_result = None
    if "manager_result" not in st.session_state:
        st.session_state.manager_result = None
    if "manager_timeline" not in st.session_state:
        st.session_state.manager_timeline = []
    if "selected_db_id" not in st.session_state:
        st.session_state.selected_db_id = ""
    if "selected_agent_id" not in st.session_state:
        st.session_state.selected_agent_id = ""


def _refresh_selection_ids() -> None:
    state = st.session_state.runner_state
    dbs = state.get("databases", [])
    agents = state.get("agents", [])

    db_ids = [str(item.get("id", "")) for item in dbs if isinstance(item, dict)]
    agent_ids = [str(item.get("id", "")) for item in agents if isinstance(item, dict)]

    if not st.session_state.selected_db_id or st.session_state.selected_db_id not in db_ids:
        st.session_state.selected_db_id = db_ids[0] if db_ids else ""
    if not st.session_state.selected_agent_id or st.session_state.selected_agent_id not in agent_ids:
        st.session_state.selected_agent_id = agent_ids[0] if agent_ids else ""


def _get_db_by_id(state: dict[str, Any], db_id: str) -> dict[str, Any] | None:
    for item in state.get("databases", []):
        if isinstance(item, dict) and str(item.get("id", "")) == db_id:
            return item
    return None


def _get_agent_by_id(state: dict[str, Any], agent_id: str) -> dict[str, Any] | None:
    for item in state.get("agents", []):
        if isinstance(item, dict) and str(item.get("id", "")) == agent_id:
            return item
    return None


def _update_db_in_state(state: dict[str, Any], payload: dict[str, Any]) -> None:
    target_id = str(payload.get("id", "")).strip()
    if not target_id:
        raise ValueError("Database id is required.")
    databases = state.get("databases", [])
    if not isinstance(databases, list):
        databases = []
        state["databases"] = databases

    for idx, item in enumerate(databases):
        if isinstance(item, dict) and str(item.get("id", "")) == target_id:
            databases[idx] = payload
            return
    databases.append(payload)


def _delete_db_in_state(state: dict[str, Any], db_id: str) -> None:
    databases = state.get("databases", [])
    if not isinstance(databases, list):
        return
    state["databases"] = [
        item for item in databases if not (isinstance(item, dict) and str(item.get("id", "")) == db_id)
    ]
    if state.get("active_database_id") == db_id:
        state["active_database_id"] = None


def _update_agent_in_state(state: dict[str, Any], payload: dict[str, Any]) -> None:
    target_id = str(payload.get("id", "")).strip()
    if not target_id:
        raise ValueError("Agent id is required.")
    agents = state.get("agents", [])
    if not isinstance(agents, list):
        agents = []
        state["agents"] = agents

    for idx, item in enumerate(agents):
        if isinstance(item, dict) and str(item.get("id", "")) == target_id:
            agents[idx] = payload
            return
    agents.append(payload)


def _delete_agent_in_state(state: dict[str, Any], agent_id: str) -> None:
    agents = state.get("agents", [])
    if not isinstance(agents, list):
        return
    state["agents"] = [
        item for item in agents if not (isinstance(item, dict) and str(item.get("id", "")) == agent_id)
    ]


def render_configuration_tab() -> None:
    state = st.session_state.runner_state
    st.subheader("Configuration")
    st.caption("Configure local LLM, manager limits and database connections.")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("Reload from config_page.py", use_container_width=True):
            st.session_state.runner_state = _state_from_config_page()
            _refresh_selection_ids()
            st.success("Reloaded from config_page.py")
            st.rerun()
    with c2:
        if st.button("Save runtime config JSON", use_container_width=True):
            try:
                _validate_runtime_state(state)
                _save_runtime_state(state)
                st.success(f"Saved to {RUNTIME_STATE_PATH}")
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
    with c3:
        if st.button("Validate current config", use_container_width=True):
            try:
                _validate_runtime_state(state)
                st.success("Configuration is valid.")
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

    with st.expander("LLM", expanded=True):
        llm = state.setdefault("llm", {})
        provider = st.selectbox(
            "Provider",
            options=["ollama", "http"],
            index=0 if str(llm.get("provider", "ollama")) == "ollama" else 1,
        )
        llm["provider"] = provider
        llm["model"] = st.text_input("Model", value=str(llm.get("model", "llama3.1")))
        llm["base_url"] = st.text_input(
            "Base URL",
            value=str(llm.get("base_url", "http://localhost:11434")),
            help="Ollama default: http://localhost:11434",
        )
        llm["endpoint"] = st.text_input(
            "HTTP endpoint (optional)",
            value=str(llm.get("endpoint", "") or ""),
            help="Used mostly for provider=http.",
        ) or None
        llm["api_key"] = st.text_input(
            "API key (optional)",
            value=str(llm.get("api_key", "") or ""),
            type="password",
        ) or None
        llm["timeout_seconds"] = st.number_input(
            "Timeout (seconds)",
            min_value=1,
            max_value=300,
            value=int(llm.get("timeout_seconds", 60)),
        )
        llm["temperature"] = st.number_input(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=float(llm.get("temperature", 0.1)),
            step=0.1,
        )
        llm["system_prompt"] = st.text_area(
            "Global system prompt",
            value=str(llm.get("system_prompt", "")),
            height=110,
        )
        headers_text = st.text_area(
            "Custom headers (JSON object)",
            value=_json_text(llm.get("headers", {})),
            height=110,
        )
        try:
            parsed_headers = _parse_json_text(headers_text, "LLM headers")
            if not isinstance(parsed_headers, dict):
                raise ValueError("LLM headers must be a JSON object.")
            llm["headers"] = parsed_headers
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))

        if st.button("Test LLM connection"):
            try:
                runtime = _validate_runtime_state(state)
                result = LLMClient(runtime.llm).test_connection()
                st.success(
                    f"{result.get('message')} | provider={result.get('provider')} "
                    f"| model_count={result.get('model_count')}"
                )
                models = result.get("models", [])
                if isinstance(models, list) and models:
                    st.code("\n".join(str(item) for item in models), language="text")
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

    with st.expander("Manager", expanded=True):
        manager = state.setdefault("manager", {})
        manager["max_steps"] = st.number_input(
            "Max orchestration steps",
            min_value=1,
            max_value=30,
            value=int(manager.get("max_steps", 6)),
        )
        manager["max_agent_calls"] = st.number_input(
            "Max agent calls",
            min_value=1,
            max_value=100,
            value=int(manager.get("max_agent_calls", 12)),
        )

    with st.expander("Databases", expanded=True):
        databases = state.setdefault("databases", [])
        if not isinstance(databases, list):
            databases = []
            state["databases"] = databases

        db_options = [str(item.get("id", "")) for item in databases if isinstance(item, dict)]
        db_options_display = db_options if db_options else ["(none)"]

        selected_db_id = st.selectbox(
            "Select database profile",
            options=db_options_display,
            index=(db_options_display.index(st.session_state.selected_db_id) if st.session_state.selected_db_id in db_options_display else 0),
            key="db_selector_streamlit",
        )
        st.session_state.selected_db_id = selected_db_id if selected_db_id != "(none)" else ""

        selected_db = _get_db_by_id(state, st.session_state.selected_db_id) if st.session_state.selected_db_id else None
        if selected_db is None:
            selected_db = _new_db_profile()

        c_new, c_del, c_active = st.columns(3)
        with c_new:
            if st.button("New profile"):
                new_payload = _new_db_profile()
                _update_db_in_state(state, new_payload)
                st.session_state.selected_db_id = str(new_payload["id"])
                st.rerun()
        with c_del:
            if st.button("Delete selected", disabled=not st.session_state.selected_db_id):
                _delete_db_in_state(state, st.session_state.selected_db_id)
                st.session_state.selected_db_id = ""
                st.success("Database profile deleted.")
                st.rerun()
        with c_active:
            if st.button("Set active", disabled=not st.session_state.selected_db_id):
                state["active_database_id"] = st.session_state.selected_db_id
                st.success(f"Active database set to {st.session_state.selected_db_id}")

        with st.form("database_form", clear_on_submit=False):
            db_id = st.text_input("id", value=str(selected_db.get("id", "")))
            db_name = st.text_input("name", value=str(selected_db.get("name", "")))
            db_engine = st.selectbox(
                "engine",
                options=SUPPORTED_ENGINES,
                index=(SUPPORTED_ENGINES.index(str(selected_db.get("engine", "clickhouse"))) if str(selected_db.get("engine", "clickhouse")) in SUPPORTED_ENGINES else 0),
            )
            db_host = st.text_input("host", value=str(selected_db.get("host", "") or ""))
            db_port = st.number_input(
                "port",
                min_value=1,
                max_value=65535,
                value=int(selected_db.get("port", 8123) or 8123),
            )
            db_database = st.text_input("database", value=str(selected_db.get("database", "") or ""))
            db_username = st.text_input("username", value=str(selected_db.get("username", "") or ""))
            db_password = st.text_input("password", value=str(selected_db.get("password", "") or ""), type="password")
            db_dsn = st.text_input("dsn (optional)", value=str(selected_db.get("dsn", "") or ""))
            db_secure = st.checkbox("secure", value=bool(selected_db.get("secure", False)))
            db_options_text = st.text_area(
                "options (JSON object)",
                value=_json_text(selected_db.get("options", {})),
                height=110,
            )
            submitted = st.form_submit_button("Save database profile")

        if submitted:
            try:
                options_payload = _parse_json_text(db_options_text, "database options")
                if not isinstance(options_payload, dict):
                    raise ValueError("database options must be a JSON object.")
                payload = {
                    "id": db_id.strip(),
                    "name": db_name.strip(),
                    "engine": db_engine,
                    "host": db_host.strip() or None,
                    "port": int(db_port),
                    "database": db_database.strip() or None,
                    "username": db_username.strip() or None,
                    "password": db_password if db_password else None,
                    "dsn": db_dsn.strip() or None,
                    "secure": bool(db_secure),
                    "options": options_payload,
                }
                DatabaseProfile.model_validate(payload)
                _update_db_in_state(state, payload)
                st.session_state.selected_db_id = payload["id"]
                st.success("Database profile saved.")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

        c_test_one, c_test_all = st.columns(2)
        with c_test_one:
            if st.button("Test selected DB", disabled=not st.session_state.selected_db_id):
                try:
                    runtime = _validate_runtime_state(state)
                    target = next(
                        db for db in runtime.databases if db.id == st.session_state.selected_db_id
                    )
                    result = connector_for(target).test_connection()
                    st.success(f"[{target.id}] {result}")
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
        with c_test_all:
            if st.button("Test all DB connections"):
                try:
                    runtime = _validate_runtime_state(state)
                    if not runtime.databases:
                        st.info("No databases configured.")
                    for db in runtime.databases:
                        try:
                            result = connector_for(db).test_connection()
                            st.success(f"[{db.id}] OK: {result}")
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"[{db.id}] FAILED: {exc}")
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

        st.caption(
            f"Active database id: {state.get('active_database_id') or '(none)'}"
        )


def render_agents_tab() -> None:
    state = st.session_state.runner_state
    st.subheader("Agents")
    st.caption("Create, edit and validate agent definitions used by single run and manager.")

    agents = state.setdefault("agents", [])
    if not isinstance(agents, list):
        agents = []
        state["agents"] = agents

    templates = list_agent_templates()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Create from template**")
        template_ids = [tpl.id for tpl in templates]
        selected_tpl = st.selectbox("Template", options=template_ids, key="new_tpl_id")
        new_agent_id = st.text_input("New agent id", value=f"{selected_tpl}_new")
        new_agent_name = st.text_input("New agent name", value=f"{selected_tpl} new")
        if st.button("Add agent from template"):
            try:
                payload = _new_agent_from_template(
                    template_id=selected_tpl,
                    agent_id=new_agent_id.strip(),
                    name=new_agent_name.strip(),
                )
                AgentConfig.model_validate(payload)
                _update_agent_in_state(state, payload)
                st.session_state.selected_agent_id = payload["id"]
                st.success("Agent created.")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

    with c2:
        enabled_count = len(
            [item for item in agents if isinstance(item, dict) and bool(item.get("enabled", True))]
        )
        st.metric("Total agents", len(agents))
        st.metric("Enabled agents", enabled_count)
        if st.button("Validate all agents"):
            try:
                for item in agents:
                    AgentConfig.model_validate(item)
                st.success("All agents are valid.")
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

    agent_ids = [str(item.get("id", "")) for item in agents if isinstance(item, dict)]
    if not agent_ids:
        st.info("No agents configured yet.")
        return

    selected_agent_id = st.selectbox(
        "Select agent",
        options=agent_ids,
        index=(agent_ids.index(st.session_state.selected_agent_id) if st.session_state.selected_agent_id in agent_ids else 0),
        key="agent_selector_streamlit",
    )
    st.session_state.selected_agent_id = selected_agent_id

    selected_agent = _get_agent_by_id(state, selected_agent_id)
    if not selected_agent:
        st.warning("Selected agent not found.")
        return

    c_save, c_delete = st.columns(2)
    with c_save:
        st.markdown("**Edit selected agent**")
    with c_delete:
        if st.button("Delete selected agent"):
            _delete_agent_in_state(state, selected_agent_id)
            st.session_state.selected_agent_id = ""
            st.success("Agent deleted.")
            st.rerun()

    with st.form("agent_form", clear_on_submit=False):
        agent_id = st.text_input("id", value=str(selected_agent.get("id", "")))
        agent_name = st.text_input("name", value=str(selected_agent.get("name", "")))
        agent_type_value = str(selected_agent.get("agent_type", "sql_analyst"))
        agent_type = st.selectbox(
            "agent_type",
            options=SUPPORTED_AGENT_TYPES,
            index=(SUPPORTED_AGENT_TYPES.index(agent_type_value) if agent_type_value in SUPPORTED_AGENT_TYPES else 0),
        )
        enabled = st.checkbox("enabled", value=bool(selected_agent.get("enabled", True)))
        description = st.text_area(
            "description",
            value=str(selected_agent.get("description", "")),
            height=80,
        )
        max_rows = st.number_input(
            "max_rows",
            min_value=1,
            max_value=5000,
            value=int(selected_agent.get("max_rows", 200)),
        )
        allowed_tables_text = st.text_input(
            "allowed_tables (comma-separated)",
            value=", ".join(
                [
                    str(item)
                    for item in selected_agent.get("allowed_tables", [])
                    if str(item).strip()
                ]
            ),
        )
        system_prompt = st.text_area(
            "system_prompt",
            value=str(selected_agent.get("system_prompt", "")),
            height=110,
        )
        sql_prompt_template = st.text_area(
            "sql_prompt_template",
            value=str(selected_agent.get("sql_prompt_template", "")),
            height=120,
        )
        answer_prompt_template = st.text_area(
            "answer_prompt_template",
            value=str(selected_agent.get("answer_prompt_template", "")),
            height=120,
        )
        template_config_text = st.text_area(
            "template_config (JSON object)",
            value=_json_text(selected_agent.get("template_config", {})),
            height=220,
        )
        submitted = st.form_submit_button("Save agent")

    if submitted:
        try:
            template_config_payload = _parse_json_text(template_config_text, "template_config")
            if not isinstance(template_config_payload, dict):
                raise ValueError("template_config must be a JSON object.")
            payload = {
                "id": agent_id.strip(),
                "name": agent_name.strip(),
                "agent_type": agent_type,
                "description": description,
                "system_prompt": system_prompt,
                "sql_prompt_template": sql_prompt_template,
                "answer_prompt_template": answer_prompt_template,
                "allowed_tables": [
                    item.strip()
                    for item in allowed_tables_text.split(",")
                    if item.strip()
                ],
                "max_rows": int(max_rows),
                "template_config": template_config_payload,
                "enabled": bool(enabled),
            }
            AgentConfig.model_validate(payload)
            _update_agent_in_state(state, payload)
            st.session_state.selected_agent_id = payload["id"]
            st.success("Agent saved.")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))

    with st.expander("Current agents (raw JSON)", expanded=False):
        st.code(_json_text(agents), language="json")


def _run_single_agent(runtime: RuntimeConfig, agent_id: str, question: str) -> dict[str, Any]:
    agent = next((item for item in runtime.agents if item.id == agent_id), None)
    if not agent:
        raise ValueError(f"Agent '{agent_id}' not found.")
    if not agent.enabled:
        raise ValueError(f"Agent '{agent_id}' is disabled.")

    database = resolve_database_for_agent(
        agent=agent,
        databases=runtime.databases,
        active_database_id=runtime.active_database_id,
        requested_database_id=None,
        required=agent.agent_type in {"sql_analyst", "clickhouse_table_manager"},
    )
    return AgentExecutor(runtime.llm).execute(agent=agent, question=question, database=database)


def render_playground_tab() -> None:
    state = st.session_state.runner_state
    st.subheader("Playground")
    st.caption("Run a single agent or run the multi-agent manager orchestration.")

    try:
        runtime = _validate_runtime_state(state)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Configuration invalid: {exc}")
        return

    mode = st.radio("Mode", options=["manager", "single"], horizontal=True)
    question = st.text_area(
        "Question",
        value="Explique-moi les 3 informations les plus importantes trouvées dans mes données.",
        height=110,
    )

    use_memory = st.checkbox("Use conversation memory", value=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Clear memory"):
            st.session_state.conversation_history = []
            st.success("Conversation memory cleared.")
    with c2:
        st.caption(
            f"Stored turns: {len(st.session_state.conversation_history)} / {MAX_MEMORY_TURNS}"
        )

    if mode == "single":
        enabled_agents = [agent for agent in runtime.agents if agent.enabled]
        if not enabled_agents:
            st.warning("No enabled agents available.")
            return
        selected_agent_id = st.selectbox(
            "Single agent",
            options=[agent.id for agent in enabled_agents],
        )

        if st.button("Run single agent", type="primary"):
            if not question.strip():
                st.error("Question is empty.")
            else:
                try:
                    output = _run_single_agent(runtime, selected_agent_id, question.strip())
                    st.session_state.single_result = output
                    st.session_state.manager_result = None
                    history = st.session_state.conversation_history if use_memory else []
                    next_history = [
                        *history,
                        ConversationTurn(role="user", content=question.strip()),
                        ConversationTurn(
                            role="assistant",
                            content=str(output.get("answer", "")).strip() or "(empty answer)",
                        ),
                    ]
                    st.session_state.conversation_history = next_history[-MAX_MEMORY_TURNS:]
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

        result = st.session_state.single_result
        if isinstance(result, dict):
            st.markdown("### Single agent result")
            sql = str(result.get("sql", "")).strip()
            if sql:
                st.markdown("**SQL**")
                st.code(sql, language="sql")
            st.markdown("**Answer**")
            st.write(str(result.get("answer", "")))
            st.markdown("**Rows**")
            st.code(_json_text(result.get("rows", [])), language="json")
            st.markdown("**Details**")
            st.code(_json_text(result.get("details", {})), language="json")

    else:
        st.caption(
            f"Manager budget: max_steps={runtime.manager.max_steps}, "
            f"max_agent_calls={runtime.manager.max_agent_calls}"
        )
        if st.button("Run manager", type="primary"):
            if not question.strip():
                st.error("Question is empty.")
            else:
                try:
                    memory = st.session_state.conversation_history if use_memory else []
                    manager = MultiAgentManager(
                        llm_config=runtime.llm,
                        agents=runtime.agents,
                        databases=runtime.databases,
                        active_database_id=runtime.active_database_id,
                        requested_database_id=None,
                        conversation_memory=memory,
                    )
                    request = ManagerRunRequest(
                        question=question.strip(),
                        max_steps=runtime.manager.max_steps,
                        max_agent_calls=runtime.manager.max_agent_calls,
                        conversation_history=memory,
                    )

                    timeline_placeholder = st.empty()
                    timeline: list[dict[str, Any]] = []
                    for event in manager.run_stream(request):
                        timeline.append(event)
                        timeline_placeholder.code(_json_text(timeline), language="json")

                    final = next(
                        (item for item in reversed(timeline) if item.get("type") == "manager_final"),
                        None,
                    )
                    if final is None:
                        raise RuntimeError("manager_final event missing.")

                    st.session_state.manager_timeline = timeline
                    st.session_state.manager_result = final
                    st.session_state.single_result = None

                    next_history = [
                        *memory,
                        ConversationTurn(role="user", content=question.strip()),
                        ConversationTurn(
                            role="assistant",
                            content=str(final.get("answer", "")).strip() or "(empty answer)",
                        ),
                    ]
                    st.session_state.conversation_history = next_history[-MAX_MEMORY_TURNS:]
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

        final = st.session_state.manager_result
        if isinstance(final, dict):
            st.markdown("### Manager final")
            st.markdown(f"**Status:** `{final.get('status')}`")
            st.markdown("**Final answer**")
            st.write(str(final.get("answer", "")))
            st.markdown("**Manager summary**")
            st.code(str(final.get("manager_summary", "")), language="text")
            st.markdown(
                f"**Judge:** verdict=`{final.get('judge_verdict')}` | "
                f"confidence=`{final.get('judge_confidence')}`"
            )
            rationale = str(final.get("judge_rationale", "")).strip()
            if rationale:
                st.caption(rationale)

            missing = str(final.get("missing_information", "")).strip()
            if missing:
                st.warning(f"Missing information: {missing}")

            with st.expander("Timeline (raw)", expanded=True):
                st.code(_json_text(st.session_state.manager_timeline), language="json")

    if st.session_state.conversation_history:
        with st.expander("Conversation memory", expanded=False):
            for turn in st.session_state.conversation_history:
                st.markdown(f"**{turn.role}**: {turn.content}")


def main() -> None:
    st.set_page_config(
        page_title="Local Agent - VSCode UI",
        page_icon="🤖",
        layout="wide",
    )
    _init_session_state()
    _refresh_selection_ids()

    st.title("Local Agent V2 - Streamlit UI")
    st.caption(
        "Simple UI executable from VSCode. Uses the same backend engine for agents, manager, DB and LLM."
    )

    tabs = st.tabs(["Configuration", "Agents", "Playground"])
    with tabs[0]:
        render_configuration_tab()
    with tabs[1]:
        render_agents_tab()
    with tabs[2]:
        render_playground_tab()


if __name__ == "__main__":
    main()
