from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from .agent_executor import AgentExecutor
from .database_routing import resolve_database_for_agent
from .llm_client import LLMClient
from .models import (
    AgentConfig,
    ConversationTurn,
    DatabaseProfile,
    LLMConfig,
    ManagerRunRequest,
    agent_requires_database,
)
from .web_navigation_agent import web_navigator_runtime_status

_JSON_FENCE = re.compile(r"```(?:json)?\\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL)
_CODE_FENCE = re.compile(r"```(?:[a-z0-9_-]+)?\\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL)


class ManagerCall(BaseModel):
    agent_id: str = Field(min_length=1)
    question: str = Field(min_length=1)


class ManagerDecision(BaseModel):
    status: Literal["continue", "done", "blocked"]
    rationale: str = ""
    calls: list[ManagerCall] = Field(default_factory=list)
    final_answer: str | None = None
    missing_information: str | None = None


class ManagerJudgeReport(BaseModel):
    verdict: Literal["pass", "partial", "fail"] = "partial"
    confidence: int = Field(default=50, ge=0, le=100)
    rationale: str = ""
    checks_passed: list[str] = Field(default_factory=list)
    checks_failed: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class MultiAgentManager:
    def __init__(
        self,
        llm_config: LLMConfig,
        agents: list[AgentConfig],
        databases: list[DatabaseProfile],
        active_database_id: str | None,
        requested_database_id: str | None = None,
        conversation_memory: list[ConversationTurn] | None = None,
    ) -> None:
        self.llm_config = llm_config
        self.databases = databases
        self.active_database_id = active_database_id
        self.requested_database_id = requested_database_id
        self.manager_llm = LLMClient(llm_config)
        self.agent_executor = AgentExecutor(llm_config)
        self.enabled_agents = {agent.id: agent for agent in agents if agent.enabled}
        self.unavailable_agents: dict[str, str] = {}
        self.history: list[dict[str, Any]] = []
        self.conversation_memory = conversation_memory[:] if conversation_memory else []
        self._initialize_runtime_unavailable_agents()

    def run_stream(self, request: ManagerRunRequest):
        agent_calls = 0
        completed_steps = 0

        yield self._event(
            "manager_start",
            question=request.question,
            database_id=self.requested_database_id,
            max_steps=request.max_steps,
            max_agent_calls=request.max_agent_calls,
            memory_turns=len(self.conversation_memory),
            enabled_agents=[
                {
                    "agent_id": agent.id,
                    "name": agent.name,
                    "agent_type": agent.agent_type,
                    "description": agent.description,
                }
                for agent in self.enabled_agents.values()
            ],
            unavailable_agents=[
                {
                    "agent_id": agent_id,
                    "name": self.enabled_agents[agent_id].name
                    if agent_id in self.enabled_agents
                    else agent_id,
                    "agent_type": self.enabled_agents[agent_id].agent_type
                    if agent_id in self.enabled_agents
                    else "unknown",
                    "reason": reason,
                }
                for agent_id, reason in self.unavailable_agents.items()
            ],
        )

        if not self.enabled_agents:
            yield self._final_event(
                status="blocked",
                answer="I could not start because no enabled agents are available.",
                missing_information="Enable at least one agent in the Agents tab.",
                steps=0,
                agent_calls=0,
            )
            return

        if not self._plannable_agents():
            yield self._final_event(
                status="blocked",
                answer="I could not start because all enabled agents are currently unavailable at runtime.",
                missing_information=self._infer_missing_information(),
                steps=0,
                agent_calls=0,
            )
            return

        last_decision: ManagerDecision | None = None

        while completed_steps < request.max_steps and agent_calls < request.max_agent_calls:
            if not self._plannable_agents():
                yield self._final_event(
                    status="blocked",
                    answer="I could not continue because no runnable agents remain available.",
                    missing_information=self._infer_missing_information(),
                    steps=completed_steps,
                    agent_calls=agent_calls,
                )
                return

            completed_steps += 1
            decision = self._decide(
                question=request.question,
                step=completed_steps,
                max_steps=request.max_steps,
                used_calls=agent_calls,
                max_calls=request.max_agent_calls,
            )
            if decision.status == "continue":
                decision.calls = self._dedupe_proposed_calls(decision.calls)
            decision = self._adapt_decision_for_data_analysis(
                decision=decision,
                question=request.question,
                step=completed_steps,
                max_steps=request.max_steps,
                used_calls=agent_calls,
                max_calls=request.max_agent_calls,
            )
            last_decision = decision

            decision_payload: dict[str, Any] = {
                "step": completed_steps,
                "status": decision.status,
                "rationale": decision.rationale,
                "calls": [call.model_dump() for call in decision.calls],
            }
            if self._is_data_analysis_request(request.question):
                decision_payload["data_analysis_state"] = self._data_analysis_state()
            yield self._event(
                "manager_decision",
                **decision_payload,
            )

            if decision.status == "done":
                answer = self._normalize_final_answer(decision.final_answer)
                if not answer:
                    answer = self._fallback_answer()
                yield self._final_event(
                    status="done",
                    answer=answer,
                    missing_information=decision.missing_information,
                    steps=completed_steps,
                    agent_calls=agent_calls,
                )
                return

            if decision.status == "blocked":
                answer = self._normalize_final_answer(decision.final_answer)
                if not answer:
                    answer = "I could not complete the full request."
                missing = (
                    self._normalize_optional_text(decision.missing_information)
                    or self._infer_missing_information()
                )
                yield self._final_event(
                    status="blocked",
                    answer=answer,
                    missing_information=missing,
                    steps=completed_steps,
                    agent_calls=agent_calls,
                )
                return

            if not decision.calls:
                if self._has_successful_answer():
                    yield self._final_event(
                        status="done",
                        answer=self._fallback_answer(),
                        missing_information=None,
                        steps=completed_steps,
                        agent_calls=agent_calls,
                    )
                else:
                    yield self._final_event(
                        status="blocked",
                        answer="The manager could not identify the next useful action.",
                        missing_information="No valid next agent call was proposed.",
                        steps=completed_steps,
                        agent_calls=agent_calls,
                    )
                return

            had_executed_call_in_step = False
            had_success_in_step = False
            latest_success_observation: dict[str, Any] | None = None
            for index, call in enumerate(decision.calls, start=1):
                if agent_calls >= request.max_agent_calls:
                    break

                agent = self.enabled_agents.get(call.agent_id)
                if not agent:
                    self.history.append(
                        {
                            "step": completed_steps,
                            "agent_id": call.agent_id,
                            "question": call.question,
                            "status": "invalid_agent",
                            "error": "Agent ID not found or disabled.",
                        }
                    )
                    yield self._event(
                        "manager_warning",
                        step=completed_steps,
                        message=f"Agent '{call.agent_id}' is not available.",
                    )
                    continue

                unavailable_reason = self.unavailable_agents.get(agent.id)
                if unavailable_reason:
                    self.history.append(
                        {
                            "step": completed_steps,
                            "agent_id": agent.id,
                            "agent_name": agent.name,
                            "agent_type": agent.agent_type,
                            "question": call.question,
                            "status": "unavailable",
                            "error": unavailable_reason,
                        }
                    )
                    yield self._event(
                        "manager_warning",
                        step=completed_steps,
                        message=(
                            f"Agent '{agent.name}' is temporarily unavailable: {unavailable_reason}"
                        ),
                    )
                    continue

                if self._is_redundant_call(call.agent_id, call.question):
                    self.history.append(
                        {
                            "step": completed_steps,
                            "agent_id": agent.id,
                            "agent_name": agent.name,
                            "agent_type": agent.agent_type,
                            "question": call.question,
                            "status": "skipped_redundant",
                            "error": "Equivalent successful call already exists in history.",
                        }
                    )
                    yield self._event(
                        "manager_warning",
                        step=completed_steps,
                        message=(
                            f"Skipping redundant call for '{agent.name}' because an equivalent "
                            "successful answer is already available."
                        ),
                    )
                    continue

                try:
                    selected_database = resolve_database_for_agent(
                        agent=agent,
                        databases=self.databases,
                        active_database_id=self.active_database_id,
                        requested_database_id=self.requested_database_id,
                        required=agent_requires_database(agent.agent_type),
                    )

                    yield self._event(
                        "agent_call_started",
                        step=completed_steps,
                        call_index=index,
                        agent_id=agent.id,
                        agent_name=agent.name,
                        agent_type=agent.agent_type,
                        question=call.question,
                        database_id=selected_database.id if selected_database else None,
                        database_name=selected_database.name if selected_database else None,
                    )
                    had_executed_call_in_step = True

                    execution_question = self._build_execution_question_for_agent(
                        user_question=request.question,
                        planned_question=call.question,
                        agent=agent,
                    )
                    output = self.agent_executor.execute(
                        agent=agent,
                        question=execution_question,
                        database=selected_database,
                    )
                    agent_calls += 1
                    output_rows = output.get("rows", [])
                    if not isinstance(output_rows, list):
                        output_rows = []
                    row_count = len(output_rows)
                    answer = output.get("answer", "")
                    sql = output.get("sql", "")
                    details = output.get("details", {})
                    rows_preview = self._rows_preview(output_rows)
                    table_names = self._extract_sql_tables(sql)
                    cross_table_query = self._is_cross_table_sql(sql)
                    where_preview = self._extract_where_preview(sql)

                    observation = {
                        "step": completed_steps,
                        "agent_id": agent.id,
                        "agent_name": agent.name,
                        "agent_type": agent.agent_type,
                        "question": call.question,
                        "status": "success",
                        "sql": sql,
                        "answer": answer,
                        "row_count": row_count,
                        "database_id": selected_database.id if selected_database else None,
                        "database_name": selected_database.name if selected_database else None,
                        "details": details,
                        "rows_preview": rows_preview,
                        "table_names": table_names,
                        "cross_table_query": cross_table_query,
                        "where_preview": where_preview,
                    }
                    self.history.append(observation)
                    had_success_in_step = True
                    latest_success_observation = observation

                    yield self._event(
                        "agent_call_completed",
                        step=completed_steps,
                        call_index=index,
                        agent_id=agent.id,
                        agent_name=agent.name,
                        agent_type=agent.agent_type,
                        sql=sql,
                        answer=answer,
                        row_count=row_count,
                        database_id=selected_database.id if selected_database else None,
                        database_name=selected_database.name if selected_database else None,
                        details=details,
                        rows_preview=rows_preview,
                        table_names=table_names,
                        cross_table_query=cross_table_query,
                        where_preview=where_preview,
                    )

                    remaining_calls_in_step = max(0, len(decision.calls) - index)
                    if remaining_calls_in_step > 0:
                        finalized_answer = self._maybe_finalize_after_success(
                            question=request.question,
                            step=completed_steps,
                            max_steps=request.max_steps,
                            used_calls=agent_calls,
                            max_calls=request.max_agent_calls,
                            latest_observation=observation,
                            remaining_calls_in_step=remaining_calls_in_step,
                        )
                        if finalized_answer:
                            yield self._final_event(
                                status="done",
                                answer=finalized_answer,
                                missing_information=None,
                                steps=completed_steps,
                                agent_calls=agent_calls,
                            )
                            return
                except Exception as exc:  # noqa: BLE001
                    had_executed_call_in_step = True
                    agent_calls += 1
                    error_message = str(exc)
                    self.history.append(
                        {
                            "step": completed_steps,
                            "agent_id": agent.id,
                            "agent_name": agent.name,
                            "agent_type": agent.agent_type,
                            "question": call.question,
                            "status": "failed",
                            "error": error_message,
                        }
                    )
                    yield self._event(
                        "agent_call_failed",
                        step=completed_steps,
                        call_index=index,
                        agent_id=agent.id,
                        agent_name=agent.name,
                        agent_type=agent.agent_type,
                        error=error_message,
                    )
                    if self._should_mark_unavailable(agent=agent, error_message=error_message):
                        self.unavailable_agents[agent.id] = error_message
                        yield self._event(
                            "agent_marked_unavailable",
                            step=completed_steps,
                            agent_id=agent.id,
                            agent_name=agent.name,
                            agent_type=agent.agent_type,
                            reason=error_message,
                        )

            if not had_executed_call_in_step:
                if self._has_successful_answer():
                    yield self._final_event(
                        status="done",
                        answer=self._fallback_answer(),
                        missing_information=None,
                        steps=completed_steps,
                        agent_calls=agent_calls,
                    )
                else:
                    yield self._final_event(
                        status="blocked",
                        answer="The manager could not execute any useful call for this step.",
                        missing_information="All proposed calls were invalid, unavailable, or redundant.",
                        steps=completed_steps,
                        agent_calls=agent_calls,
                    )
                return

            if had_success_in_step and latest_success_observation is not None:
                finalized_answer = self._maybe_finalize_after_success(
                    question=request.question,
                    step=completed_steps,
                    max_steps=request.max_steps,
                    used_calls=agent_calls,
                    max_calls=request.max_agent_calls,
                    latest_observation=latest_success_observation,
                    remaining_calls_in_step=0,
                )
                if finalized_answer:
                    yield self._final_event(
                        status="done",
                        answer=finalized_answer,
                        missing_information=None,
                        steps=completed_steps,
                        agent_calls=agent_calls,
                    )
                    return

        missing = self._infer_missing_information()
        exhausted_answer = (
            "I could not fully complete the request before exhausting "
            "the available orchestration paths."
        )

        if last_decision:
            candidate = self._normalize_final_answer(last_decision.final_answer)
            if candidate:
                exhausted_answer = candidate

        yield self._final_event(
            status="exhausted",
            answer=exhausted_answer,
            missing_information=missing,
            steps=completed_steps,
            agent_calls=agent_calls,
        )

    def _decide(
        self,
        question: str,
        step: int,
        max_steps: int,
        used_calls: int,
        max_calls: int,
    ) -> ManagerDecision:
        prompt = self._manager_prompt(
            question=question,
            step=step,
            max_steps=max_steps,
            used_calls=used_calls,
            max_calls=max_calls,
        )
        raw = self.manager_llm.generate(
            prompt,
            system_prompt=(
                "You are a multi-agent orchestration manager. "
                "Return strict JSON only."
            ),
        )
        return self._parse_decision(raw)

    def _manager_prompt(
        self,
        question: str,
        step: int,
        max_steps: int,
        used_calls: int,
        max_calls: int,
    ) -> str:
        plannable_agents = list(self._plannable_agents().values())
        available_agents_text = "\\n".join(
            [
                (
                    f"- id={agent.id} | type={agent.agent_type} | name={agent.name} "
                    f"| description={agent.description} | runtime_hint={self._agent_runtime_hint(agent)}"
                )
                for agent in plannable_agents
            ]
        )

        if not available_agents_text:
            available_agents_text = "- none"

        unavailable_agents_text = "\\n".join(
            [
                (
                    f"- id={agent_id} | type={self.enabled_agents[agent_id].agent_type} "
                    f"| name={self.enabled_agents[agent_id].name} | reason={reason}"
                )
                for agent_id, reason in self.unavailable_agents.items()
                if agent_id in self.enabled_agents
            ]
        )
        if not unavailable_agents_text:
            unavailable_agents_text = "- none"

        history_text = self._history_text()
        memory_text = self._conversation_memory_text()
        sql_chain_guidance = self._sql_chain_guidance_text(plannable_agents)
        data_memory_text = self._data_analysis_memory_text()
        remaining_steps = max_steps - step + 1
        remaining_calls = max_calls - used_calls

        return (
            "User request:\\n"
            f"{question}\\n\\n"
            "Conversation memory:\\n"
            f"{memory_text}\\n\\n"
            "Available runnable agents:\\n"
            f"{available_agents_text}\\n\\n"
            "Temporarily unavailable agents:\\n"
            f"{unavailable_agents_text}\\n\\n"
            "Execution history:\\n"
            f"{history_text}\\n\\n"
            "Data-analysis chaining guidance:\\n"
            f"{sql_chain_guidance}\\n\\n"
            "Data-analysis memory:\\n"
            f"{data_memory_text}\\n\\n"
            "Limits:\\n"
            f"- remaining_steps={remaining_steps}\\n"
            f"- remaining_agent_calls={remaining_calls}\\n\\n"
            "Return only one JSON object with this schema:\\n"
            "{\\n"
            '  "status": "continue" | "done" | "blocked",\\n'
            '  "rationale": "short reason",\\n'
            '  "calls": [{"agent_id": "...", "question": "..."}],\\n'
            '  "final_answer": "... or null",\\n'
            '  "missing_information": "... or null"\\n'
            "}\\n\\n"
            "Rules:\\n"
            "- Use only listed agent_id values.\\n"
            "- Do not call agents listed as temporarily unavailable.\\n"
            "- If execution history already contains a successful answer that satisfies the user request, return status=done immediately.\\n"
            "- If solved, return status=done with final_answer.\\n"
            "- If impossible with current capabilities/data, return status=blocked and missing_information.\\n"
            "- If more work is needed, return status=continue with 1 to 3 calls.\\n"
            "- For data-analysis requests requiring schema exploration, scoped filtering, or cross-table logic, prefer progressive chaining across SQL-capable agents before status=done unless complete evidence is already available.\\n"
            "- Typical SQL progression: discover tables -> inspect columns -> build scoped query -> reuse scope on related tables -> synthesize final answer.\\n"
            "- If cross-table analysis is requested, gather evidence from at least two relevant tables and at least one cross-table SQL relation (JOIN/subquery) before status=done.\\n"
            "- web_scraper uses HTTP scraping and does not require Playwright.\\n"
            "- For web_scraper calls, include concrete target URL(s) or domain names in the call question when available.\\n"
            "- web_navigator requires Playwright browser runtime.\\n"
            "- If web_navigator is unavailable, prefer web_scraper or wikipedia_retriever for web info tasks.\\n"
            "- Avoid repeating the exact same call, especially if it already succeeded.\\n"
            "- When re-calling a previously used agent, vary the question only if new dependency information was produced."
        )

    def _conversation_memory_text(self) -> str:
        if not self.conversation_memory:
            return "- none"
        lines: list[str] = []
        for turn in self.conversation_memory[-12:]:
            role = str(turn.role).strip() or "user"
            content = self._truncate_text(turn.content, 280)
            lines.append(f"- {role}: {content}")
        return "\\n".join(lines)

    def _history_text(self) -> str:
        if not self.history:
            return "- none"

        lines: list[str] = []
        for item in self.history[-20:]:
            step = item.get("step")
            agent_name = item.get("agent_name") or item.get("agent_id")
            agent_type = str(item.get("agent_type") or "unknown")
            status = item.get("status")
            question = str(item.get("question", ""))[:180]
            if status == "success":
                answer = str(item.get("answer", ""))[:220]
                row_count = item.get("row_count")
                sql_preview = self._truncate_text(item.get("sql", ""), 180)
                rows_preview_raw = item.get("rows_preview")
                table_names_raw = item.get("table_names")
                where_preview = self._truncate_text(item.get("where_preview", ""), 120)
                cross_table_query = bool(item.get("cross_table_query"))
                rows_preview = ""
                if isinstance(rows_preview_raw, list) and rows_preview_raw:
                    rows_preview = self._truncate_text(
                        json.dumps(rows_preview_raw, ensure_ascii=False),
                        220,
                    )
                table_names = ""
                if isinstance(table_names_raw, list) and table_names_raw:
                    normalized: list[str] = []
                    for name in table_names_raw:
                        cleaned = self._sanitize_table_name(name)
                        if cleaned:
                            normalized.append(cleaned)
                    if normalized:
                        table_names = ", ".join(normalized[:6])

                line = (
                    f"- step={step} agent={agent_name} type={agent_type} "
                    f"status=success rows={row_count} question={question}"
                )
                if sql_preview:
                    line += f" sql={sql_preview}"
                if rows_preview:
                    line += f" rows_preview={rows_preview}"
                if table_names:
                    line += f" tables={table_names}"
                if where_preview:
                    line += f" where={where_preview}"
                if cross_table_query:
                    line += " cross_table=true"
                if answer:
                    line += f" answer={answer}"
                lines.append(line)
            else:
                error = str(item.get("error", ""))[:220]
                lines.append(
                    f"- step={step} agent={agent_name} type={agent_type} status={status} question={question} error={error}"
                )
        return "\\n".join(lines)

    def _parse_decision(self, raw_text: str) -> ManagerDecision:
        cleaned = raw_text.strip()
        fence_match = _JSON_FENCE.search(cleaned)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        payload: Any
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            payload = self._json_object_from_text(cleaned)

        if payload is None:
            return ManagerDecision(
                status="blocked",
                rationale="Manager output could not be parsed as JSON.",
                final_answer="I could not produce a reliable orchestration plan.",
                missing_information="A valid manager decision JSON payload is missing.",
            )

        try:
            decision = ManagerDecision.model_validate(payload)
        except ValidationError as exc:
            return ManagerDecision(
                status="blocked",
                rationale="Manager output failed schema validation.",
                final_answer="I could not continue because the orchestration decision format was invalid.",
                missing_information=str(exc),
            )

        if decision.status != "continue":
            decision.calls = []
        decision.rationale = self._truncate_text(decision.rationale, 500)
        decision.final_answer = (
            self._normalize_final_answer(decision.final_answer) or None
        )
        decision.missing_information = (
            self._normalize_optional_text(decision.missing_information) or None
        )

        return decision

    @staticmethod
    def _normalize_question(question: str) -> str:
        normalized = re.sub(r"\s+", " ", str(question or "").strip().lower())
        return normalized

    def _call_signature(self, agent_id: str, question: str) -> str:
        return f"{str(agent_id or '').strip()}::{self._normalize_question(question)}"

    def _dedupe_proposed_calls(self, calls: list[ManagerCall]) -> list[ManagerCall]:
        deduped: list[ManagerCall] = []
        seen_signatures: set[str] = set()
        for call in calls:
            signature = self._call_signature(call.agent_id, call.question)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            deduped.append(call)
        return deduped

    def _adapt_decision_for_data_analysis(
        self,
        *,
        decision: ManagerDecision,
        question: str,
        step: int,
        max_steps: int,
        used_calls: int,
        max_calls: int,
    ) -> ManagerDecision:
        if not self._is_data_analysis_request(question):
            return decision
        if max_steps - step + 1 <= 0 or max_calls - used_calls <= 0:
            return decision

        sql_agents = self._sql_plannable_agents()
        if not sql_agents:
            return decision

        state = self._data_analysis_state()
        has_enough_progress = self._can_finalize_data_analysis(question)

        if decision.status in {"done", "blocked"} and not has_enough_progress:
            bootstrap_call = self._next_sql_bootstrap_call(
                question=question,
                state=state,
                sql_agents=sql_agents,
            )
            if bootstrap_call:
                return ManagerDecision(
                    status="continue",
                    rationale=(
                        "Manager decision was refined to continue SQL orchestration "
                        "because data evidence is still incomplete."
                    ),
                    calls=[bootstrap_call],
                    final_answer=None,
                    missing_information=None,
                )
            return decision

        if decision.status != "continue":
            return decision

        has_sql_call = any(
            self._is_sql_capable_agent_type(
                str(self.enabled_agents.get(call.agent_id).agent_type)
            )
            for call in decision.calls
            if self.enabled_agents.get(call.agent_id)
        )
        if has_sql_call or has_enough_progress:
            return decision

        bootstrap_call = self._next_sql_bootstrap_call(
            question=question,
            state=state,
            sql_agents=sql_agents,
        )
        if not bootstrap_call:
            return decision

        new_calls = [*decision.calls, bootstrap_call]
        if len(new_calls) > 3:
            new_calls = new_calls[:3]

        return ManagerDecision(
            status="continue",
            rationale=(
                "Added one SQL progression call to improve multi-table evidence before finalization."
            ),
            calls=self._dedupe_proposed_calls(new_calls),
            final_answer=decision.final_answer,
            missing_information=decision.missing_information,
        )

    def _sql_plannable_agents(self) -> list[AgentConfig]:
        return [
            agent
            for agent in self._plannable_agents().values()
            if self._is_sql_capable_agent_type(str(agent.agent_type))
        ]

    def _next_sql_bootstrap_call(
        self,
        *,
        question: str,
        state: dict[str, Any],
        sql_agents: list[AgentConfig],
    ) -> ManagerCall | None:
        if not sql_agents:
            return None

        preferred = next(
            (agent for agent in sql_agents if str(agent.agent_type) == "sql_analyst"),
            sql_agents[0],
        )
        if state.get("successful_sql_calls", 0) <= 0:
            subtask = (
                "Discover candidate tables relevant to the user objective and identify key columns "
                "(IDs, dates, foreign keys, business keys). Return concrete table/column evidence."
            )
        elif self._needs_cross_table_analysis(question) and int(state.get("tables_count", 0)) < 2:
            subtask = (
                "Find at least one additional related table and infer relationship keys with the current "
                "scope so cross-table analysis becomes possible."
            )
        elif not bool(state.get("has_where_clause")):
            subtask = (
                "Build a scoped SQL subset using WHERE filters aligned with the user objective, "
                "and return the filtered evidence."
            )
        elif self._needs_cross_table_analysis(question) and int(state.get("cross_table_queries", 0)) < 1:
            subtask = (
                "Create a cross-table SQL (JOIN or subquery) that reuses scoped results from prior calls "
                "to progress toward the final answer."
            )
        else:
            subtask = (
                "Validate and refine the current SQL analysis with one targeted query "
                "(aggregation/check/consistency) to support a reliable final conclusion."
            )

        return ManagerCall(agent_id=preferred.id, question=subtask)

    def _is_redundant_call(self, agent_id: str, question: str) -> bool:
        signature = self._call_signature(agent_id, question)
        for item in reversed(self.history[-50:]):
            if item.get("status") != "success":
                continue
            previous_signature = self._call_signature(
                str(item.get("agent_id", "")),
                str(item.get("question", "")),
            )
            if previous_signature != signature:
                continue
            if self._is_potentially_final_answer(
                answer=item.get("answer", ""),
                details=item.get("details"),
            ):
                return True
        return False

    def _has_successful_answer(self) -> bool:
        return self._latest_success_observation() is not None

    def _latest_success_observation(self) -> dict[str, Any] | None:
        for item in reversed(self.history):
            if item.get("status") != "success":
                continue
            if self._is_potentially_final_answer(
                answer=item.get("answer", ""),
                details=item.get("details"),
            ):
                return item
        return None

    @staticmethod
    def _is_potentially_final_answer(answer: Any, details: Any) -> bool:
        text = str(answer or "").strip()
        if len(text) < 24:
            return False

        lowered = text.lower()
        uncertain_markers = (
            "missing information",
            "please provide",
            "need more",
            "not enough",
            "cannot",
            "can't",
            "unable",
            "failed",
            "error",
            "blocked",
            "no relevant content found",
        )
        if any(marker in lowered for marker in uncertain_markers):
            return False

        if isinstance(details, dict):
            for key in ("missing_information", "error", "errors", "blocked", "needs_input"):
                value = details.get(key)
                if value:
                    return False

        return True

    def _maybe_finalize_after_success(
        self,
        *,
        question: str,
        step: int,
        max_steps: int,
        used_calls: int,
        max_calls: int,
        latest_observation: dict[str, Any],
        remaining_calls_in_step: int,
    ) -> str | None:
        answer = str(latest_observation.get("answer", "")).strip()
        details = latest_observation.get("details")
        if not self._is_potentially_final_answer(answer=answer, details=details):
            return None

        if self._is_data_analysis_request(question) and not self._can_finalize_data_analysis(question):
            return None

        latest_snapshot = {
            "agent_id": latest_observation.get("agent_id"),
            "agent_name": latest_observation.get("agent_name"),
            "agent_type": latest_observation.get("agent_type"),
            "question": self._truncate_text(latest_observation.get("question", ""), 220),
            "answer": self._truncate_text(answer, 600),
            "row_count": latest_observation.get("row_count"),
            "rows_preview": latest_observation.get("rows_preview", []),
            "details": latest_observation.get("details", {}),
        }
        prompt = (
            "Decide if orchestration can stop now.\n"
            "Return strict JSON only.\n\n"
            "Schema:\n"
            "{\n"
            '  "status": "done" | "continue",\n'
            '  "rationale": "short reason",\n'
            '  "final_answer": "... or null"\n'
            "}\n\n"
            f"User request:\n{question}\n\n"
            "Conversation memory:\n"
            f"{self._conversation_memory_text()}\n\n"
            f"Latest successful result:\n{json.dumps(latest_snapshot, ensure_ascii=False)}\n\n"
            "Recent history:\n"
            f"{self._history_text()}\n\n"
            "Limits:\n"
            f"- step={step}/{max_steps}\n"
            f"- used_agent_calls={used_calls}/{max_calls}\n"
            f"- remaining_calls_in_current_step={remaining_calls_in_step}\n\n"
            "Rules:\n"
            "- If the user request is already satisfied, return status=done with a concise final_answer.\n"
            "- For progressive data-analysis requests, ensure enough evidence was collected before status=done.\n"
            "- If an important gap remains, return status=continue.\n"
            "- Do not invent missing data."
        )

        try:
            raw = self.manager_llm.generate(
                prompt,
                system_prompt=(
                    "You are a strict completion gate for multi-agent orchestration. "
                    "Return strict JSON only."
                ),
            )
            payload = self._json_object_from_text(raw)
            if not isinstance(payload, dict):
                return None
            status = str(payload.get("status") or "").strip().lower()
            if status != "done":
                return None
            final_answer = self._normalize_final_answer(payload.get("final_answer"))
            normalized_source_answer = self._normalize_final_answer(answer)
            return final_answer or normalized_source_answer
        except Exception:  # noqa: BLE001
            return None

    def _sql_chain_guidance_text(self, agents: list[AgentConfig]) -> str:
        sql_agents = [
            agent
            for agent in agents
            if self._is_sql_capable_agent_type(str(agent.agent_type))
        ]
        if not sql_agents:
            return "- no SQL-capable agent available."

        lines = [
            "- Use SQL-capable agents iteratively when the request needs progressive discovery.",
            "- Prefer a chain such as: list tables -> inspect columns -> build scoped query -> reuse scope on related tables.",
            "- Reuse results from previous successful calls (SQL and rows_preview) to design the next sub-task.",
            "- Use specialized SQL agent descriptions to distribute tasks naturally.",
        ]
        for agent in sql_agents[:8]:
            lines.append(
                f"- SQL agent available: id={agent.id} | name={agent.name} | description={agent.description}"
            )
        return "\\n".join(lines)

    def _build_execution_question_for_agent(
        self,
        *,
        user_question: str,
        planned_question: str,
        agent: AgentConfig,
    ) -> str:
        if not self._is_sql_capable_agent_type(str(agent.agent_type)):
            return planned_question

        dependencies_text = self._recent_data_dependencies_text(limit=6)
        if not dependencies_text:
            return planned_question
        data_memory_text = self._data_analysis_memory_text()
        next_hint = self._next_data_action_hint(user_question)

        return (
            "Global analysis objective:\n"
            f"{user_question}\n\n"
            "Current sub-task:\n"
            f"{planned_question}\n\n"
            "Upstream SQL/data evidence from previous calls:\n"
            f"{dependencies_text}\n\n"
            "Current data-analysis memory:\n"
            f"{data_memory_text}\n\n"
            "Suggested next-step focus:\n"
            f"{next_hint}\n\n"
            "Instruction: continue the analysis using this evidence, keep the scope aligned with the current sub-task."
        )

    def _recent_data_dependencies_text(self, limit: int = 6) -> str:
        lines: list[str] = []
        count = 0
        for item in reversed(self.history):
            if item.get("status") != "success":
                continue
            if not self._is_sql_capable_agent_type(str(item.get("agent_type") or "")):
                continue

            count += 1
            sql_text = self._truncate_text(item.get("sql", ""), 220)
            answer_text = self._truncate_text(item.get("answer", ""), 180)
            rows_preview_raw = item.get("rows_preview")
            rows_preview = ""
            if isinstance(rows_preview_raw, list) and rows_preview_raw:
                rows_preview = self._truncate_text(
                    json.dumps(rows_preview_raw, ensure_ascii=False),
                    220,
                )
            table_names_raw = item.get("table_names")
            table_names = ""
            if isinstance(table_names_raw, list) and table_names_raw:
                normalized_tables: list[str] = []
                for name in table_names_raw:
                    cleaned = self._sanitize_table_name(name)
                    if cleaned:
                        normalized_tables.append(cleaned)
                if normalized_tables:
                    table_names = ", ".join(normalized_tables[:6])

            base = (
                f"- step={item.get('step')} agent={item.get('agent_name') or item.get('agent_id')} "
                f"question={self._truncate_text(item.get('question', ''), 140)}"
            )
            if sql_text:
                base += f" sql={sql_text}"
            if rows_preview:
                base += f" rows_preview={rows_preview}"
            if table_names:
                base += f" tables={table_names}"
            if answer_text:
                base += f" answer={answer_text}"
            lines.append(base)

            if count >= limit:
                break

        if not lines:
            return "- none"
        lines.reverse()
        return "\\n".join(lines)

    def _rows_preview(self, rows: list[Any], max_rows: int = 3) -> list[dict[str, Any]]:
        preview: list[dict[str, Any]] = []
        for raw_row in rows[:max_rows]:
            if not isinstance(raw_row, dict):
                preview.append({"value": self._truncate_text(raw_row, 180)})
                continue
            normalized: dict[str, Any] = {}
            for key, value in list(raw_row.items())[:20]:
                normalized[str(key)] = self._truncate_text(value, 180)
            preview.append(normalized)
        return preview

    def _data_analysis_state(self) -> dict[str, Any]:
        successful_sql_calls = 0
        tables_touched: list[str] = []
        seen_tables: set[str] = set()
        columns_seen: list[str] = []
        seen_columns: set[str] = set()
        where_clauses: list[str] = []
        cross_table_queries = 0
        has_where_clause = False

        for item in self.history:
            if item.get("status") != "success":
                continue
            if not self._is_sql_capable_agent_type(str(item.get("agent_type") or "")):
                continue

            successful_sql_calls += 1
            sql_text = str(item.get("sql") or "")
            table_names_raw = item.get("table_names")
            table_candidates: list[str] = []
            if isinstance(table_names_raw, list):
                table_candidates = [str(name) for name in table_names_raw]
            if not table_candidates:
                table_candidates = self._extract_sql_tables(sql_text)
            for table in table_candidates:
                normalized_table = self._sanitize_table_name(table)
                if not normalized_table:
                    continue
                if normalized_table in seen_tables:
                    continue
                seen_tables.add(normalized_table)
                tables_touched.append(normalized_table)

            rows_preview = item.get("rows_preview")
            if isinstance(rows_preview, list):
                for row in rows_preview:
                    if not isinstance(row, dict):
                        continue
                    for key in row.keys():
                        column_name = str(key).strip()
                        if not column_name:
                            continue
                        lowered = column_name.lower()
                        if lowered in seen_columns:
                            continue
                        seen_columns.add(lowered)
                        columns_seen.append(column_name)

            where_preview = str(
                item.get("where_preview") or self._extract_where_preview(sql_text)
            ).strip()
            if where_preview:
                has_where_clause = True
                if where_preview not in where_clauses:
                    where_clauses.append(where_preview)
            if self._is_cross_table_sql(sql_text):
                cross_table_queries += 1

        return {
            "successful_sql_calls": successful_sql_calls,
            "tables_touched": tables_touched[:20],
            "tables_count": len(tables_touched),
            "columns_seen": columns_seen[:30],
            "cross_table_queries": cross_table_queries,
            "has_where_clause": has_where_clause,
            "where_clauses": where_clauses[:8],
        }

    def _data_analysis_memory_text(self) -> str:
        state = self._data_analysis_state()
        successful_calls = int(state.get("successful_sql_calls", 0))
        if successful_calls <= 0:
            return "- no SQL evidence yet."

        tables = state.get("tables_touched", [])
        if not isinstance(tables, list):
            tables = []
        columns = state.get("columns_seen", [])
        if not isinstance(columns, list):
            columns = []
        where_clauses = state.get("where_clauses", [])
        if not isinstance(where_clauses, list):
            where_clauses = []

        lines = [
            f"- successful_sql_calls={successful_calls}",
            (
                f"- tables_touched({len(tables)}): "
                + (", ".join([self._truncate_text(item, 40) for item in tables[:12]]) or "none")
            ),
            f"- cross_table_queries={int(state.get('cross_table_queries', 0))}",
            (
                "- columns_seen: "
                + (", ".join([self._truncate_text(item, 30) for item in columns[:12]]) or "none")
            ),
        ]
        if where_clauses:
            lines.append(
                "- where_scopes: "
                + " | ".join([self._truncate_text(item, 90) for item in where_clauses[:4]])
            )
        else:
            lines.append("- where_scopes: none")
        return "\\n".join(lines)

    def _next_data_action_hint(self, question: str) -> str:
        state = self._data_analysis_state()
        successful_sql_calls = int(state.get("successful_sql_calls", 0))
        tables_count = int(state.get("tables_count", 0))
        cross_table_queries = int(state.get("cross_table_queries", 0))
        has_where_clause = bool(state.get("has_where_clause"))

        if successful_sql_calls <= 0:
            return (
                "Start with schema discovery: identify relevant tables and key columns before deep analysis."
            )
        if self._needs_cross_table_analysis(question) and tables_count < 2:
            return (
                "Find at least one additional related table and relationship key(s) "
                "to enable cross-table reasoning."
            )
        if not has_where_clause:
            return (
                "Create a scoped subset with explicit WHERE conditions aligned to the business intent."
            )
        if self._needs_cross_table_analysis(question) and cross_table_queries < 1:
            return (
                "Execute one JOIN/subquery that reuses the scoped subset to produce cross-table evidence."
            )
        return (
            "Refine and validate with one targeted SQL check (aggregation or consistency control) "
            "then synthesize the final answer."
        )

    def _can_finalize_data_analysis(self, question: str) -> bool:
        state = self._data_analysis_state()
        if int(state.get("successful_sql_calls", 0)) < 2:
            return False
        if int(state.get("tables_count", 0)) < 1:
            return False

        if self._needs_cross_table_analysis(question):
            if int(state.get("tables_count", 0)) < 2:
                return False
            if int(state.get("cross_table_queries", 0)) < 1:
                return False

        normalized = str(question or "").lower()
        scoped_tokens = ("filter", "where", "scope", "segment", "subset", "périmètre")
        if any(token in normalized for token in scoped_tokens) and not bool(
            state.get("has_where_clause")
        ):
            return False

        return True

    @staticmethod
    def _is_sql_capable_agent_type(agent_type: str) -> bool:
        return agent_type in {"sql_analyst", "clickhouse_table_manager"}

    @staticmethod
    def _is_data_analysis_request(question: str) -> bool:
        normalized = str(question or "").lower()
        hints = (
            "table",
            "tables",
            "schema",
            "column",
            "columns",
            "sql",
            "clickhouse",
            "oracle",
            "join",
            "sub scope",
            "subscope",
            "filter",
            "where",
            "analysis",
            "analyt",
            "champ",
            "champs",
            "requete",
            "query",
            "kpi",
            "metric",
            "measure",
            "trend",
        )
        return any(token in normalized for token in hints)

    @staticmethod
    def _needs_cross_table_analysis(question: str) -> bool:
        normalized = str(question or "").lower()
        hints = (
            "join",
            "cross-table",
            "cross table",
            "across tables",
            "multiple tables",
            "multi table",
            "plusieurs tables",
            "croiser",
            "relation",
            "related table",
            "table a",
            "table b",
        )
        return any(token in normalized for token in hints)

    def _extract_sql_tables(self, sql: str) -> list[str]:
        text = str(sql or "")
        if not text:
            return []

        patterns = (
            r"(?i)\bfrom\s+([`\"\[]?[a-zA-Z0-9_.]+[`\"\]]?)",
            r"(?i)\bjoin\s+([`\"\[]?[a-zA-Z0-9_.]+[`\"\]]?)",
            r"(?i)\binto\s+([`\"\[]?[a-zA-Z0-9_.]+[`\"\]]?)",
            r"(?i)\bupdate\s+([`\"\[]?[a-zA-Z0-9_.]+[`\"\]]?)",
            r"(?i)\btable\s+([`\"\[]?[a-zA-Z0-9_.]+[`\"\]]?)",
        )
        blocked = {
            "select",
            "where",
            "group",
            "order",
            "limit",
            "inner",
            "left",
            "right",
            "full",
            "on",
            "as",
            "values",
            "set",
        }
        discovered: list[str] = []
        seen: set[str] = set()
        for pattern in patterns:
            for raw in re.findall(pattern, text):
                candidate = self._sanitize_table_name(raw)
                if not candidate:
                    continue
                if candidate.lower() in blocked:
                    continue
                if candidate.lower().startswith("system."):
                    continue
                if candidate.lower() in seen:
                    continue
                seen.add(candidate.lower())
                discovered.append(candidate)
        return discovered

    @staticmethod
    def _sanitize_table_name(raw: Any) -> str:
        value = str(raw or "").strip()
        if not value:
            return ""
        value = value.strip(",;")
        value = value.strip("`\"[]")
        if value.startswith("("):
            return ""
        value = re.sub(r"\s+", "", value)
        return value

    def _is_cross_table_sql(self, sql: str) -> bool:
        text = str(sql or "").lower()
        if not text:
            return False
        tables = self._extract_sql_tables(sql)
        if len({table.lower() for table in tables}) >= 2:
            return True
        if " join " in f" {text} ":
            return True
        if re.search(r"\b(?:in|exists)\s*\(\s*select\b", text):
            return True
        return False

    def _extract_where_preview(self, sql: str) -> str:
        text = str(sql or "")
        if not text:
            return ""
        match = re.search(
            r"(?is)\bwhere\b\s+(.*?)(?:\bgroup\s+by\b|\border\s+by\b|\blimit\b|\bhaving\b|$)",
            text,
        )
        if not match:
            return ""
        where_clause = re.sub(r"\s+", " ", match.group(1)).strip().strip(";")
        return self._truncate_text(where_clause, 180)

    @staticmethod
    def _json_object_from_text(text: str) -> Any | None:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _normalize_optional_text(value: Any) -> str:
        return str(value or "").strip()

    def _normalize_final_answer(self, value: Any) -> str:
        text = self._normalize_optional_text(value)
        if not text:
            return ""

        collapsed = re.sub(r"\s+", " ", text).strip().lower().strip(".")
        placeholder_values = {
            "none",
            "null",
            "n/a",
            "na",
            "empty",
            "(empty)",
            "undefined",
        }
        if collapsed in placeholder_values:
            return ""
        return text

    def _default_answer_for_status(
        self, status: Literal["done", "blocked", "exhausted"]
    ) -> str:
        if status == "done":
            return self._fallback_answer()
        if status == "blocked":
            return "The manager stopped before producing an explicit final answer."
        return "The manager exhausted its budget before producing an explicit final answer."

    def _fallback_answer(self) -> str:
        latest_success = self._latest_success_observation()
        if latest_success:
            latest_answer = self._normalize_final_answer(latest_success.get("answer"))
            if latest_answer:
                return latest_answer

        for item in reversed(self.history):
            if item.get("status") != "success":
                continue
            answer = self._normalize_final_answer(item.get("answer"))
            if answer:
                return answer
        return "Done. The manager completed execution, but no explicit final answer was produced."

    def _build_manager_summary(
        self,
        *,
        status: Literal["done", "blocked", "exhausted"],
        answer: str,
        missing_information: str | None,
        steps: int,
        agent_calls: int,
    ) -> str:
        snapshot = self._summary_snapshot(
            status=status,
            answer=answer,
            missing_information=missing_information,
            steps=steps,
            agent_calls=agent_calls,
        )
        prompt = (
            "You summarize multi-agent execution traces for an operator dashboard.\n"
            "Return plain text only (no JSON), maximum 14 lines.\n"
            "Use exactly this structure:\n"
            "What worked:\n"
            "- ...\n"
            "What did not work:\n"
            "- ...\n"
            "Agents called and purpose:\n"
            "- AgentName (agent_type): purpose -> outcome\n"
            "If a section is empty, write '- none'.\n\n"
            f"Execution snapshot JSON:\n{json.dumps(snapshot, ensure_ascii=False)}"
        )
        try:
            raw = self.manager_llm.generate(
                prompt,
                system_prompt=(
                    "You are concise, factual and operational. "
                    "Do not invent details beyond the provided execution snapshot."
                ),
            )
            normalized = self._normalize_summary_text(raw)
            if normalized:
                return normalized
        except Exception:  # noqa: BLE001
            pass
        return self._fallback_manager_summary(snapshot)

    def _summary_snapshot(
        self,
        *,
        status: Literal["done", "blocked", "exhausted"],
        answer: str,
        missing_information: str | None,
        steps: int,
        agent_calls: int,
    ) -> dict[str, Any]:
        calls: list[dict[str, Any]] = []
        for item in self.history[-60:]:
            calls.append(
                {
                    "step": item.get("step"),
                    "agent_id": item.get("agent_id"),
                    "agent_name": item.get("agent_name"),
                    "agent_type": item.get("agent_type"),
                    "question": self._truncate_text(item.get("question", ""), 180),
                    "status": item.get("status"),
                    "row_count": item.get("row_count"),
                    "answer_preview": self._truncate_text(item.get("answer", ""), 220),
                    "error": self._truncate_text(item.get("error", ""), 220),
                }
            )
        return {
            "status": status,
            "final_answer": self._truncate_text(answer, 500),
            "missing_information": self._truncate_text(missing_information or "", 300),
            "steps": steps,
            "agent_calls": agent_calls,
            "calls": calls,
        }

    def _fallback_manager_summary(self, snapshot: dict[str, Any]) -> str:
        calls = snapshot.get("calls", [])
        if not isinstance(calls, list):
            calls = []

        worked: list[str] = []
        failed: list[str] = []
        called: list[str] = []

        for call in calls:
            if not isinstance(call, dict):
                continue
            agent_name = str(call.get("agent_name") or call.get("agent_id") or "Agent").strip()
            agent_type = str(call.get("agent_type") or "unknown").strip()
            question = self._truncate_text(call.get("question", ""), 120)
            call_status = str(call.get("status") or "unknown").strip()
            row_count = call.get("row_count")
            answer_preview = self._truncate_text(call.get("answer_preview", ""), 100)
            error_text = self._truncate_text(call.get("error", ""), 120)
            purpose = question or "no explicit question"
            outcome = call_status
            if call_status == "success":
                outcome = f"success ({row_count if row_count is not None else 0} rows)"
                if answer_preview:
                    outcome += f", {answer_preview}"
                worked.append(f"- {agent_name}: {outcome}")
            else:
                if error_text:
                    outcome = f"{call_status}, {error_text}"
                failed.append(f"- {agent_name}: {outcome}")

            called.append(
                f"- {agent_name} ({agent_type}): {purpose} -> {outcome}"
            )

        if not worked:
            worked = ["- none"]
        if not failed:
            failed = ["- none"]
        if not called:
            called = ["- none"]

        worked = worked[:6]
        failed = failed[:6]
        called = called[:8]

        return "\n".join(
            [
                "What worked:",
                *worked,
                "What did not work:",
                *failed,
                "Agents called and purpose:",
                *called,
            ]
        )

    @staticmethod
    def _truncate_text(value: Any, limit: int) -> str:
        text = str(value or "").strip()
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."

    @staticmethod
    def _normalize_summary_text(raw: str) -> str:
        text = raw.strip()
        match = _CODE_FENCE.search(text)
        if match:
            text = match.group(1).strip()
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        if not lines:
            return ""
        normalized = "\n".join(lines)
        # Keep response compact for UI readability.
        if len(normalized) > 2200:
            normalized = normalized[:2197].rstrip() + "..."
        return normalized

    def _run_sanity_judge(
        self,
        *,
        status: Literal["done", "blocked", "exhausted"],
        answer: str,
        missing_information: str | None,
        steps: int,
        agent_calls: int,
    ) -> ManagerJudgeReport:
        snapshot = self._summary_snapshot(
            status=status,
            answer=answer,
            missing_information=missing_information,
            steps=steps,
            agent_calls=agent_calls,
        )
        prompt = (
            "You are an independent sanity checker (LLM-as-a-Judge) for multi-agent execution.\n"
            "Evaluate chain quality, consistency, and result reliability.\n"
            "Return strict JSON only with this schema:\n"
            "{\n"
            '  "verdict": "pass" | "partial" | "fail",\n'
            '  "confidence": 0..100,\n'
            '  "rationale": "short factual rationale",\n'
            '  "checks_passed": ["..."],\n'
            '  "checks_failed": ["..."],\n'
            '  "recommendations": ["..."]\n'
            "}\n\n"
            "Rules:\n"
            "- confidence must be an integer between 0 and 100.\n"
            "- be strict: if critical gaps remain, use verdict=partial or fail.\n"
            "- do not invent facts outside the snapshot.\n\n"
            f"Execution snapshot JSON:\n{json.dumps(snapshot, ensure_ascii=False)}"
        )
        try:
            raw = self.manager_llm.generate(
                prompt,
                system_prompt=(
                    "You are a strict reviewer for AI-agent workflows. "
                    "Return strict JSON only."
                ),
            )
            report = self._parse_judge_report(raw)
            if report:
                return report
        except Exception:  # noqa: BLE001
            pass
        return self._fallback_judge_report(
            status=status,
            answer=answer,
            missing_information=missing_information,
        )

    def _parse_judge_report(self, raw_text: str) -> ManagerJudgeReport | None:
        cleaned = raw_text.strip()
        fence_match = _JSON_FENCE.search(cleaned)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        payload: Any
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            payload = self._json_object_from_text(cleaned)

        if payload is None:
            return None

        try:
            report = ManagerJudgeReport.model_validate(payload)
        except ValidationError:
            return None

        report.confidence = max(0, min(100, int(report.confidence)))
        report.rationale = self._truncate_text(report.rationale, 500)
        report.checks_passed = [self._truncate_text(item, 160) for item in report.checks_passed[:6]]
        report.checks_failed = [self._truncate_text(item, 160) for item in report.checks_failed[:6]]
        report.recommendations = [self._truncate_text(item, 160) for item in report.recommendations[:6]]
        return report

    def _fallback_judge_report(
        self,
        *,
        status: Literal["done", "blocked", "exhausted"],
        answer: str,
        missing_information: str | None,
    ) -> ManagerJudgeReport:
        success_calls = len([item for item in self.history if item.get("status") == "success"])
        failed_calls = len(
            [
                item
                for item in self.history
                if item.get("status") in {"failed", "unavailable", "invalid_agent"}
            ]
        )
        has_answer = bool(str(answer or "").strip())
        has_missing = bool(str(missing_information or "").strip())

        verdict: Literal["pass", "partial", "fail"] = "partial"
        confidence = 55
        if status == "done" and has_answer and failed_calls == 0 and success_calls > 0 and not has_missing:
            verdict = "pass"
            confidence = 88
        elif status == "done" and has_answer and success_calls > 0:
            verdict = "partial"
            confidence = 68
        elif status in {"blocked", "exhausted"}:
            verdict = "fail"
            confidence = 30

        checks_passed: list[str] = []
        checks_failed: list[str] = []
        recommendations: list[str] = []

        if has_answer:
            checks_passed.append("A final answer was produced.")
        else:
            checks_failed.append("No final answer was produced.")

        if success_calls > 0:
            checks_passed.append(f"{success_calls} successful agent call(s) were recorded.")
        else:
            checks_failed.append("No successful agent calls were recorded.")

        if failed_calls > 0:
            checks_failed.append(f"{failed_calls} call(s) failed or were unavailable.")

        if has_missing:
            checks_failed.append("Missing information is still present.")
            recommendations.append("Provide the missing information and rerun the workflow.")
        if status in {"blocked", "exhausted"}:
            recommendations.append("Increase manager limits or enable additional capable agents.")
        if not recommendations:
            recommendations.append("Result appears consistent with current execution evidence.")

        rationale = (
            f"Fallback judge used heuristic scoring (status={status}, "
            f"success_calls={success_calls}, failed_calls={failed_calls})."
        )
        return ManagerJudgeReport(
            verdict=verdict,
            confidence=confidence,
            rationale=rationale,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            recommendations=recommendations,
        )

    def _infer_missing_information(self) -> str:
        if self.unavailable_agents:
            unavailable_messages: list[str] = []
            for agent_id, reason in list(self.unavailable_agents.items())[:3]:
                name = self.enabled_agents[agent_id].name if agent_id in self.enabled_agents else agent_id
                unavailable_messages.append(f"{name}: {reason}")
            return "Some agents are unavailable at runtime: " + " | ".join(unavailable_messages)

        errors = [str(item.get("error")) for item in self.history if item.get("error")]
        if errors:
            return (
                "The workflow encountered blocking errors: "
                + " | ".join(errors[:3])
            )
        return (
            "Additional data access, broader agent capabilities, or a higher "
            "execution budget may be required."
        )

    def _plannable_agents(self) -> dict[str, AgentConfig]:
        return {
            agent_id: agent
            for agent_id, agent in self.enabled_agents.items()
            if agent_id not in self.unavailable_agents
        }

    def _initialize_runtime_unavailable_agents(self) -> None:
        has_web_navigator = any(
            agent.agent_type == "web_navigator" for agent in self.enabled_agents.values()
        )
        if not has_web_navigator:
            return

        navigator_ready, navigator_reason = web_navigator_runtime_status()
        if navigator_ready:
            return

        reason = navigator_reason or (
            "Web navigator runtime is unavailable. Install Playwright Chromium binaries."
        )
        for agent_id, agent in self.enabled_agents.items():
            if agent.agent_type == "web_navigator":
                self.unavailable_agents[agent_id] = reason

    @staticmethod
    def _agent_runtime_hint(agent: AgentConfig) -> str:
        if agent.agent_type == "sql_analyst":
            cfg = agent.template_config if isinstance(agent.template_config, dict) else {}
            mode = str(cfg.get("sql_use_case_mode", "llm_sql")).strip().lower()
            sql_template = str(cfg.get("sql_query_template", "")).strip()
            if mode in {"parameterized_template", "templated_sql", "use_case"} and sql_template:
                raw_params = cfg.get("sql_parameters", [])
                param_names: list[str] = []
                if isinstance(raw_params, list):
                    for item in raw_params:
                        if not isinstance(item, dict):
                            continue
                        name = str(item.get("name", "")).strip()
                        if name and name not in param_names:
                            param_names.append(name)
                if param_names:
                    return (
                        "Parameterized SQL use case. "
                        "Provide values for: " + ", ".join(param_names[:6])
                    )
                return "Parameterized SQL use case with template placeholders."
            return "General SQL generation on ClickHouse/Oracle."
        if agent.agent_type == "clickhouse_table_manager":
            cfg = agent.template_config if isinstance(agent.template_config, dict) else {}
            protected = self._to_bool(cfg.get("protect_existing_tables"), default=True)
            updates = self._to_bool(cfg.get("allow_row_updates"), default=True)
            deletes = self._to_bool(cfg.get("allow_row_deletes"), default=False)
            return (
                "ClickHouse DDL/DML manager. "
                f"protect_existing_tables={protected}, allow_row_updates={updates}, "
                f"allow_row_deletes={deletes}."
            )
        if agent.agent_type == "web_scraper":
            return "HTTP scraping, no browser automation required."
        if agent.agent_type == "internet_search":
            return "Web search over HTTP APIs."
        if agent.agent_type == "rss_news":
            return "RSS/Atom feed aggregation and filtered news briefing."
        if agent.agent_type == "wikipedia_retriever":
            return "Wikipedia API retrieval over HTTP."
        if agent.agent_type == "web_navigator":
            return "Interactive browser automation (requires Playwright)."
        if agent.agent_type == "word_manager":
            return "Word document operations (.docx) in configured folders."
        return "Standard runtime."

    @staticmethod
    def _should_mark_unavailable(agent: AgentConfig, error_message: str) -> bool:
        normalized = error_message.lower()
        if agent.agent_type == "web_navigator" and "playwright" in normalized:
            return True
        return False

    @staticmethod
    def _to_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off"}:
                return False
        return default

    @staticmethod
    def _event(event_type: str, **payload: Any) -> dict[str, Any]:
        return {
            "type": event_type,
            "ts": datetime.now(timezone.utc).isoformat(),
            **payload,
        }

    def _final_event(
        self,
        status: Literal["done", "blocked", "exhausted"],
        answer: str,
        missing_information: str | None,
        steps: int,
        agent_calls: int,
    ) -> dict[str, Any]:
        normalized_answer = self._normalize_final_answer(answer)
        if not normalized_answer:
            normalized_answer = self._default_answer_for_status(status)
        normalized_missing = self._normalize_optional_text(missing_information) or None
        manager_summary = self._build_manager_summary(
            status=status,
            answer=normalized_answer,
            missing_information=normalized_missing,
            steps=steps,
            agent_calls=agent_calls,
        )
        judge = self._run_sanity_judge(
            status=status,
            answer=normalized_answer,
            missing_information=normalized_missing,
            steps=steps,
            agent_calls=agent_calls,
        )
        return {
            "type": "manager_final",
            "ts": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "answer": normalized_answer,
            "manager_summary": manager_summary,
            "judge_verdict": judge.verdict,
            "judge_confidence": judge.confidence,
            "judge_rationale": judge.rationale,
            "judge_checks_passed": judge.checks_passed,
            "judge_checks_failed": judge.checks_failed,
            "judge_recommendations": judge.recommendations,
            "missing_information": normalized_missing,
            "steps": steps,
            "agent_calls": agent_calls,
        }
