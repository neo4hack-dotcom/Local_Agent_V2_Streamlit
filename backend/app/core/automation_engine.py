from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .agent_executor import AgentExecutor
from .database_routing import resolve_database_for_agent
from .models import (
    AgentCatalog,
    AppSettings,
    AutomationRule,
    AutomationRunLog,
    AutomationRunStep,
    agent_requires_database,
)
from .storage import JSONRepository


class AutomationEngine:
    def __init__(
        self,
        settings_repo: JSONRepository[AppSettings],
        agents_repo: JSONRepository[AgentCatalog],
    ) -> None:
        self.settings_repo = settings_repo
        self.agents_repo = agents_repo
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._last_scan_at: dict[str, float] = {}
        self._known_files: dict[str, dict[str, int]] = {}
        self._last_error_at: dict[str, float] = {}

        self._runs: deque[AutomationRunLog] = deque(maxlen=400)
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="automation-engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def reset_rule_state(self, rule_id: str) -> None:
        with self._lock:
            self._last_scan_at.pop(rule_id, None)
            self._known_files.pop(rule_id, None)
            self._last_error_at.pop(rule_id, None)

    def list_runs(self, automation_id: str | None = None, limit: int = 50) -> list[AutomationRunLog]:
        with self._lock:
            logs = list(self._runs)
        logs.reverse()
        if automation_id:
            logs = [item for item in logs if item.automation_id == automation_id]
        return logs[: max(1, min(limit, 200))]

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                settings = self.settings_repo.load()
                catalog = self.agents_repo.load()
            except Exception:  # noqa: BLE001
                self._stop_event.wait(2.0)
                continue

            now = time.time()
            active_ids = {rule.id for rule in settings.automations}
            with self._lock:
                for stale_id in list(self._last_scan_at):
                    if stale_id not in active_ids:
                        self._last_scan_at.pop(stale_id, None)
                        self._known_files.pop(stale_id, None)
                        self._last_error_at.pop(stale_id, None)

            for rule in settings.automations:
                if not rule.enabled:
                    continue

                with self._lock:
                    last_scan = self._last_scan_at.get(rule.id, 0.0)
                if now - last_scan < rule.poll_interval_seconds:
                    continue

                with self._lock:
                    self._last_scan_at[rule.id] = now

                self._scan_rule(rule, settings=settings, catalog=catalog, now=now)

            self._stop_event.wait(1.0)

    def _scan_rule(self, rule: AutomationRule, settings: AppSettings, catalog: AgentCatalog, now: float) -> None:
        try:
            current_files = self._snapshot_files(rule)
        except Exception as exc:  # noqa: BLE001
            self._record_rule_error_once(rule, str(exc), now)
            return

        with self._lock:
            previous_files = self._known_files.get(rule.id)
            if previous_files is None:
                self._known_files[rule.id] = current_files
                return

        new_paths = [path for path in current_files if path not in previous_files]
        with self._lock:
            self._known_files[rule.id] = current_files

        if not new_paths:
            return

        for file_path in sorted(new_paths)[: rule.max_events_per_scan]:
            self._run_rule_for_file(rule=rule, file_path=file_path, settings=settings, catalog=catalog)

    def _snapshot_files(self, rule: AutomationRule) -> dict[str, int]:
        root = Path(rule.watch_path).expanduser()
        if not root.exists() or not root.is_dir():
            raise ValueError(f"Watch path is missing or not a directory: {root}")

        normalized_extensions = self._normalize_extensions(rule.file_extensions)
        include_all = not normalized_extensions or "*" in normalized_extensions

        iterator = root.rglob("*") if rule.recursive else root.glob("*")
        files: dict[str, int] = {}
        for path in iterator:
            if not path.is_file():
                continue
            if not include_all and path.suffix.lower() not in normalized_extensions:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            files[str(path)] = int(stat.st_mtime_ns)
        return files

    @staticmethod
    def _normalize_extensions(values: list[str]) -> set[str]:
        normalized: set[str] = set()
        for value in values:
            raw = str(value).strip().lower()
            if not raw:
                continue
            if raw == "*":
                normalized.add("*")
                continue
            if not raw.startswith("."):
                raw = f".{raw}"
            normalized.add(raw)
        return normalized

    def _run_rule_for_file(
        self,
        rule: AutomationRule,
        file_path: str,
        settings: AppSettings,
        catalog: AgentCatalog,
    ) -> None:
        started = self._now_iso()
        prompt = self._render_prompt(rule.prompt_template, file_path=file_path, event_type=rule.event_type)
        steps: list[AutomationRunStep] = []
        current_question = prompt
        error: str | None = None

        enabled_agents = {agent.id: agent for agent in catalog.agents if agent.enabled}
        if not rule.agent_chain:
            error = "No agent is configured in this automation chain."
        else:
            executor = AgentExecutor(settings.llm)
            for agent_id in rule.agent_chain:
                agent = enabled_agents.get(agent_id)
                if not agent:
                    steps.append(
                        AutomationRunStep(
                            agent_id=agent_id,
                            agent_name=agent_id,
                            status="failed",
                            error="Agent not found or disabled.",
                        )
                    )
                    error = f"Agent '{agent_id}' not found or disabled."
                    break

                try:
                    database = resolve_database_for_agent(
                        agent=agent,
                        databases=settings.databases,
                        active_database_id=settings.active_database_id,
                        requested_database_id=None,
                        required=agent_requires_database(agent.agent_type),
                    )
                    output = executor.execute(agent=agent, question=current_question, database=database)
                    answer = str(output.get("answer", "")).strip()
                    steps.append(
                        AutomationRunStep(
                            agent_id=agent.id,
                            agent_name=agent.name,
                            status="success",
                            answer_preview=answer[:280] if answer else None,
                            row_count=len(output.get("rows", [])),
                            details=output.get("details", {}),
                        )
                    )
                    if answer:
                        current_question = answer
                except Exception as exc:  # noqa: BLE001
                    steps.append(
                        AutomationRunStep(
                            agent_id=agent.id,
                            agent_name=agent.name,
                            status="failed",
                            error=str(exc),
                        )
                    )
                    error = str(exc)
                    break

        success_count = sum(1 for step in steps if step.status == "success")
        failed_count = sum(1 for step in steps if step.status == "failed")
        if failed_count == 0 and success_count > 0:
            status = "success"
        elif success_count > 0:
            status = "partial"
        else:
            status = "failed"

        run_log = AutomationRunLog(
            id=uuid4().hex,
            automation_id=rule.id,
            automation_name=rule.name,
            event_type=rule.event_type,
            event_file_path=file_path,
            started_at=started,
            finished_at=self._now_iso(),
            status=status,
            prompt=prompt,
            steps=steps,
            final_answer=current_question if success_count > 0 else None,
            error=error,
        )
        with self._lock:
            self._runs.append(run_log)

    def _record_rule_error_once(self, rule: AutomationRule, error: str, now: float) -> None:
        cooldown_seconds = 60.0
        with self._lock:
            last_logged_at = self._last_error_at.get(rule.id, 0.0)
            if now - last_logged_at < cooldown_seconds:
                return
            self._last_error_at[rule.id] = now
            self._runs.append(
                AutomationRunLog(
                    id=uuid4().hex,
                    automation_id=rule.id,
                    automation_name=rule.name,
                    event_type=rule.event_type,
                    event_file_path=rule.watch_path,
                    started_at=self._now_iso(),
                    finished_at=self._now_iso(),
                    status="failed",
                    prompt=f"Watch path check failed: {rule.watch_path}",
                    steps=[],
                    final_answer=None,
                    error=error,
                )
            )

    @staticmethod
    def _render_prompt(template: str, *, file_path: str, event_type: str) -> str:
        payload = {
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "event_type": event_type,
        }
        try:
            return template.format(**payload)
        except Exception:  # noqa: BLE001
            return (
                f"{template}\n\n"
                f"File path: {payload['file_path']}\n"
                f"File name: {payload['file_name']}\n"
                f"Event type: {payload['event_type']}"
            )

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
