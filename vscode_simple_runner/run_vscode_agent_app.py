#!/usr/bin/env python3
"""
Simple executable Python runner for Local_Agent, designed for VSCode.

Run:
    python vscode_simple_runner/run_vscode_agent_app.py

What this script provides:
1) Single-agent execution
2) Multi-agent manager orchestration
3) LLM connection test
4) Database connection tests
5) Runtime summary of current configuration

Configuration source:
    vscode_simple_runner/config_page.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Backend imports (reuse current project engine)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.agent_executor import AgentExecutor
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
CONFIG_PAGE_PATH = PROJECT_ROOT / "vscode_simple_runner" / "config_page.py"


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

    import importlib.util

    module_name = "vscode_simple_runner_config_page"
    spec = importlib.util.spec_from_file_location(module_name, CONFIG_PAGE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load config module: {CONFIG_PAGE_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_runtime_config() -> RuntimeConfig:
    module = _load_config_module()

    llm_raw = getattr(module, "LLM_CONFIG", {})
    manager_raw = getattr(module, "MANAGER_CONFIG", {})
    dbs_raw = getattr(module, "DATABASES", [])
    agents_raw = getattr(module, "AGENTS", [])
    active_db_id = getattr(module, "ACTIVE_DATABASE_ID", None)

    llm = LLMConfig.model_validate(llm_raw)
    manager = ManagerConfig.model_validate(manager_raw)
    databases = [DatabaseProfile.model_validate(item) for item in dbs_raw]
    agents = [AgentConfig.model_validate(item) for item in agents_raw]

    if active_db_id:
        known_ids = {db.id for db in databases}
        if active_db_id not in known_ids:
            raise ValueError(
                f"ACTIVE_DATABASE_ID='{active_db_id}' is not in DATABASES ids={sorted(known_ids)}"
            )

    return RuntimeConfig(
        llm=llm,
        manager=manager,
        databases=databases,
        active_database_id=active_db_id,
        agents=agents,
    )


def _input(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return ""


def print_config_summary(cfg: RuntimeConfig) -> None:
    print("\n=== CONFIG SUMMARY ===")
    print(
        f"LLM: provider={cfg.llm.provider}, model={cfg.llm.model}, "
        f"base_url={cfg.llm.base_url}, timeout={cfg.llm.timeout_seconds}s"
    )
    print(
        f"Manager: max_steps={cfg.manager.max_steps}, "
        f"max_agent_calls={cfg.manager.max_agent_calls}"
    )
    print(
        f"Databases: {len(cfg.databases)} total, "
        f"active={cfg.active_database_id or '(none)'}"
    )
    for db in cfg.databases:
        print(
            f"  - id={db.id} | name={db.name} | engine={db.engine} "
            f"| host={db.host or '-'} | db={db.database or '-'}"
        )

    enabled_agents = [a for a in cfg.agents if a.enabled]
    disabled_agents = [a for a in cfg.agents if not a.enabled]
    print(
        f"Agents: {len(cfg.agents)} total, enabled={len(enabled_agents)}, "
        f"disabled={len(disabled_agents)}"
    )
    for agent in cfg.agents:
        state = "enabled" if agent.enabled else "disabled"
        print(f"  - id={agent.id} | name={agent.name} | type={agent.agent_type} | {state}")
    print(f"\nConfig file: {CONFIG_PAGE_PATH}")


def test_llm(cfg: RuntimeConfig) -> None:
    print("\n=== TEST LLM CONNECTION ===")
    client = LLMClient(cfg.llm)
    try:
        result = client.test_connection()
        models = result.get("models", [])
        print(
            f"OK: provider={result.get('provider')} "
            f"| model_count={result.get('model_count')} "
            f"| message={result.get('message')}"
        )
        if models:
            print("Models:")
            for name in models[:20]:
                print(f"  - {name}")
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {exc}")


def test_databases(cfg: RuntimeConfig) -> None:
    print("\n=== TEST DATABASE CONNECTIONS ===")
    if not cfg.databases:
        print("No database profiles configured.")
        return

    for db in cfg.databases:
        print(f"\n[{db.id}] {db.name} ({db.engine})")
        try:
            connector = connector_for(db)
            result = connector.test_connection()
            print(f"  OK: {result}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {exc}")


def choose_agent(cfg: RuntimeConfig) -> AgentConfig | None:
    enabled_agents = [a for a in cfg.agents if a.enabled]
    if not enabled_agents:
        print("No enabled agents available.")
        return None

    print("\nEnabled agents:")
    for idx, agent in enumerate(enabled_agents, start=1):
        print(f"  {idx}. {agent.name} ({agent.agent_type}) [id={agent.id}]")

    raw = _input("Select agent number: ").strip()
    if not raw.isdigit():
        print("Invalid selection.")
        return None

    index = int(raw)
    if index < 1 or index > len(enabled_agents):
        print("Invalid selection.")
        return None

    return enabled_agents[index - 1]


def _render_json_preview(rows: list[dict[str, Any]], max_items: int = 5) -> str:
    preview = rows[:max_items]
    return json.dumps(preview, ensure_ascii=False, indent=2)


def run_single_agent(
    cfg: RuntimeConfig,
    conversation_history: list[ConversationTurn],
) -> list[ConversationTurn]:
    print("\n=== RUN SINGLE AGENT ===")
    agent = choose_agent(cfg)
    if not agent:
        return conversation_history

    question = _input("Your question: ").strip()
    if not question:
        print("Question is empty.")
        return conversation_history

    executor = AgentExecutor(cfg.llm)
    try:
        database = resolve_database_for_agent(
            agent=agent,
            databases=cfg.databases,
            active_database_id=cfg.active_database_id,
            requested_database_id=None,
            required=agent.agent_type in {"sql_analyst", "clickhouse_table_manager"},
        )
        output = executor.execute(agent=agent, question=question, database=database)
    except Exception as exc:  # noqa: BLE001
        print(f"Execution failed: {exc}")
        return conversation_history

    sql = str(output.get("sql", "")).strip()
    rows = output.get("rows", [])
    answer = str(output.get("answer", "")).strip()
    details = output.get("details", {})

    print("\n--- RESULT ---")
    if sql:
        print("SQL:")
        print(sql)
    else:
        print("SQL: (none)")
    print("\nAnswer:")
    print(answer or "(empty)")
    print(f"\nRows count: {len(rows) if isinstance(rows, list) else 0}")
    if isinstance(rows, list) and rows:
        print("\nRows preview:")
        print(_render_json_preview(rows))
    print("\nDetails:")
    print(json.dumps(details if isinstance(details, dict) else {}, ensure_ascii=False, indent=2))

    next_history = [
        *conversation_history,
        ConversationTurn(role="user", content=question),
        ConversationTurn(role="assistant", content=answer or "(empty answer)"),
    ]
    return next_history[-MAX_MEMORY_TURNS:]


def _print_manager_event(event: dict[str, Any]) -> None:
    event_type = str(event.get("type", "unknown"))
    ts = str(event.get("ts", ""))

    if event_type == "manager_start":
        print(
            f"[{ts}] manager_start | steps={event.get('max_steps')} "
            f"| calls={event.get('max_agent_calls')}"
        )
        return
    if event_type == "manager_decision":
        calls = event.get("calls", [])
        call_count = len(calls) if isinstance(calls, list) else 0
        print(
            f"[{ts}] manager_decision | step={event.get('step')} "
            f"| status={event.get('status')} | proposed_calls={call_count}"
        )
        return
    if event_type == "agent_call_started":
        print(
            f"[{ts}] agent_call_started | step={event.get('step')} "
            f"| agent={event.get('agent_name')} | question={event.get('question')}"
        )
        return
    if event_type == "agent_call_completed":
        print(
            f"[{ts}] agent_call_completed | step={event.get('step')} "
            f"| agent={event.get('agent_name')} | rows={event.get('row_count')}"
        )
        return
    if event_type == "agent_call_failed":
        print(
            f"[{ts}] agent_call_failed | step={event.get('step')} "
            f"| agent={event.get('agent_name')} | error={event.get('error')}"
        )
        return
    if event_type == "agent_marked_unavailable":
        print(
            f"[{ts}] agent_marked_unavailable | agent={event.get('agent_name')} "
            f"| reason={event.get('reason')}"
        )
        return
    if event_type == "manager_warning":
        print(f"[{ts}] manager_warning | {event.get('message')}")
        return
    if event_type == "manager_final":
        print(
            f"[{ts}] manager_final | status={event.get('status')} "
            f"| steps={event.get('steps')} | calls={event.get('agent_calls')}"
        )
        return
    print(f"[{ts}] {event_type}")


def run_manager(
    cfg: RuntimeConfig,
    conversation_history: list[ConversationTurn],
) -> list[ConversationTurn]:
    print("\n=== RUN AGENT MANAGER ===")
    question = _input("Your question: ").strip()
    if not question:
        print("Question is empty.")
        return conversation_history

    manager = MultiAgentManager(
        llm_config=cfg.llm,
        agents=cfg.agents,
        databases=cfg.databases,
        active_database_id=cfg.active_database_id,
        requested_database_id=None,
        conversation_memory=conversation_history,
    )
    request = ManagerRunRequest(
        question=question,
        max_steps=cfg.manager.max_steps,
        max_agent_calls=cfg.manager.max_agent_calls,
        conversation_history=conversation_history,
    )

    timeline: list[dict[str, Any]] = []
    try:
        for event in manager.run_stream(request):
            timeline.append(event)
            _print_manager_event(event)
    except Exception as exc:  # noqa: BLE001
        print(f"Manager execution failed: {exc}")
        return conversation_history

    final_event = next(
        (item for item in reversed(timeline) if item.get("type") == "manager_final"),
        None,
    )
    if not final_event:
        print("No manager_final event received.")
        return conversation_history

    final_answer = str(final_event.get("answer", "")).strip()
    manager_summary = str(final_event.get("manager_summary", "")).strip()
    judge_verdict = final_event.get("judge_verdict")
    judge_confidence = final_event.get("judge_confidence")
    judge_rationale = final_event.get("judge_rationale")
    missing_information = final_event.get("missing_information")

    print("\n--- MANAGER FINAL ANSWER ---")
    print(final_answer or "(empty)")

    print("\n--- MANAGER SUMMARY ---")
    print(manager_summary or "(summary unavailable)")

    print("\n--- SANITY JUDGE ---")
    print(
        f"verdict={judge_verdict} | confidence={judge_confidence} | "
        f"rationale={judge_rationale}"
    )
    if missing_information:
        print(f"missing_information={missing_information}")

    next_history = [
        *conversation_history,
        ConversationTurn(role="user", content=question),
        ConversationTurn(role="assistant", content=final_answer or "(empty answer)"),
    ]
    return next_history[-MAX_MEMORY_TURNS:]


def print_menu() -> None:
    print("\n==============================")
    print("Local Agent VSCode Runner")
    print("==============================")
    print("1) Show configuration summary")
    print("2) Test LLM connection")
    print("3) Test database connections")
    print("4) Run single agent")
    print("5) Run manager (multi-agent)")
    print("6) Reload configuration")
    print("7) Exit")


def main() -> int:
    print(f"Using configuration page: {CONFIG_PAGE_PATH}")

    try:
        cfg = load_runtime_config()
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load configuration: {exc}")
        return 1

    conversation_history: list[ConversationTurn] = []

    while True:
        print_menu()
        choice = _input("Choose an option: ").strip()

        if choice == "1":
            print_config_summary(cfg)
        elif choice == "2":
            test_llm(cfg)
        elif choice == "3":
            test_databases(cfg)
        elif choice == "4":
            conversation_history = run_single_agent(cfg, conversation_history)
        elif choice == "5":
            conversation_history = run_manager(cfg, conversation_history)
        elif choice == "6":
            try:
                cfg = load_runtime_config()
                print("Configuration reloaded.")
            except Exception as exc:  # noqa: BLE001
                print(f"Reload failed: {exc}")
        elif choice == "7":
            print("Bye.")
            return 0
        else:
            print("Unknown option.")


if __name__ == "__main__":
    raise SystemExit(main())
