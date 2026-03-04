from __future__ import annotations

import json
import re
from datetime import date, datetime, time
from typing import Any
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

from .db_connectors import connector_for
from .llm_client import LLMClient
from .models import AgentConfig, DatabaseProfile, LLMConfig

_SQL_BLOCK = re.compile(r"```sql\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL)


def _json_default(value: Any) -> str:
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    return str(value)


class AgentState(TypedDict, total=False):
    question: str
    agent: AgentConfig
    database: DatabaseProfile
    llm: LLMConfig
    schema: str
    sql: str
    rows: list[dict[str, Any]]
    answer: str


def _extract_sql(raw_text: str) -> str:
    match = _SQL_BLOCK.search(raw_text)
    if match:
        return match.group(1).strip().rstrip(";")
    return raw_text.strip().rstrip(";")


def _schema_node(state: AgentState) -> AgentState:
    connector = connector_for(state["database"])
    schema = connector.schema_snapshot(state["agent"].allowed_tables)
    return {"schema": schema}


def _sql_node(state: AgentState) -> AgentState:
    allowed_tables = state["agent"].allowed_tables or ["ALL"]
    prompt = state["agent"].sql_prompt_template.format(
        question=state["question"],
        schema=state.get("schema", "No schema available"),
        allowed_tables=", ".join(allowed_tables),
    )
    llm = LLMClient(state["llm"])
    raw_sql = llm.generate(prompt, system_prompt=state["agent"].system_prompt)
    return {"sql": _extract_sql(raw_sql)}


def _query_node(state: AgentState) -> AgentState:
    connector = connector_for(state["database"])
    rows = connector.run_query(state["sql"], limit=state["agent"].max_rows)
    return {"rows": rows}


def _answer_node(state: AgentState) -> AgentState:
    llm = LLMClient(state["llm"])
    rows_payload = json.dumps(
        state.get("rows", []), ensure_ascii=False, default=_json_default
    )
    prompt = state["agent"].answer_prompt_template.format(
        question=state["question"],
        sql=state["sql"],
        rows=rows_payload,
    )
    answer = llm.generate(prompt, system_prompt=state["agent"].system_prompt)
    return {"answer": answer}


class LangGraphAgentRunner:
    def __init__(self) -> None:
        graph = StateGraph(AgentState)
        graph.add_node("schema", _schema_node)
        graph.add_node("sql", _sql_node)
        graph.add_node("query", _query_node)
        graph.add_node("answer", _answer_node)

        graph.add_edge(START, "schema")
        graph.add_edge("schema", "sql")
        graph.add_edge("sql", "query")
        graph.add_edge("query", "answer")
        graph.add_edge("answer", END)

        self.compiled = graph.compile()

    def run(
        self,
        question: str,
        agent: AgentConfig,
        database: DatabaseProfile,
        llm: LLMConfig,
    ) -> dict[str, Any]:
        initial_state: AgentState = {
            "question": question,
            "agent": agent,
            "database": database,
            "llm": llm,
        }
        result = self.compiled.invoke(initial_state)
        return {
            "sql": result.get("sql", ""),
            "rows": result.get("rows", []),
            "answer": result.get("answer", ""),
        }
