from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

DEFAULT_SQL_TEMPLATE = (
    "You must produce a valid SQL query that answers the question.\\n"
    "Constraint: return SQL only (no explanation).\\n"
    "User question: {question}\\n"
    "Allowed tables: {allowed_tables}\\n"
    "Available schema:\\n{schema}\\n"
)

DEFAULT_ANSWER_TEMPLATE = (
    "Question: {question}\\n"
    "Executed SQL: {sql}\\n"
    "Raw result (JSON): {rows}\\n"
    "Produce a concise, business-oriented answer in English."
)

AgentType = Literal[
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


class LLMConfig(BaseModel):
    provider: Literal["ollama", "http"] = "ollama"
    model: str = "llama3.1"
    system_prompt: str = ""
    base_url: str = "http://localhost:11434"
    endpoint: str | None = None
    api_key: str | None = None
    timeout_seconds: int = Field(default=60, ge=1, le=300)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    headers: dict[str, str] = Field(default_factory=dict)


class WebhookConfig(BaseModel):
    enabled: bool = False
    replace_playground: bool = False
    url: str = ""
    auth_token: str | None = None
    timeout_seconds: int = Field(default=8, ge=1, le=60)
    verify_ssl: bool = True
    include_timeline_on_final: bool = True
    headers: dict[str, str] = Field(default_factory=dict)


class DatabaseProfileBase(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    engine: Literal["clickhouse", "oracle", "elasticsearch"]
    host: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    database: str | None = None
    username: str | None = None
    password: str | None = None
    dsn: str | None = None
    secure: bool = False
    options: dict[str, Any] = Field(default_factory=dict)


class DatabaseProfile(DatabaseProfileBase):
    id: str


class DatabaseProfileCreate(DatabaseProfileBase):
    pass


class DatabaseProfileUpdate(DatabaseProfileBase):
    pass


class DatabaseConfigExport(BaseModel):
    export_version: Literal["1.0"] = "1.0"
    exported_at: str
    source: str = "local_agent_studio"
    active_database_id: str | None = None
    databases: list[DatabaseProfile] = Field(default_factory=list)


class DatabaseConfigImportRequest(BaseModel):
    payload: DatabaseConfigExport
    mode: Literal["replace", "merge"] = "replace"


class DatabaseConfigImportResponse(BaseModel):
    mode: Literal["replace", "merge"]
    imported_databases: int
    created_databases: int
    updated_databases: int
    replaced_databases: int
    active_database_id: str | None = None


class AgentConfigBase(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    agent_type: AgentType = "sql_analyst"
    description: str = ""
    system_prompt: str = "You are a reliable SQL expert and data analyst."
    sql_prompt_template: str = DEFAULT_SQL_TEMPLATE
    answer_prompt_template: str = DEFAULT_ANSWER_TEMPLATE
    allowed_tables: list[str] = Field(default_factory=list)
    max_rows: int = Field(default=200, ge=1, le=5000)
    template_config: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class AgentConfig(AgentConfigBase):
    id: str


class AgentConfigCreate(AgentConfigBase):
    pass


class AgentConfigUpdate(AgentConfigBase):
    pass


class AgentAuditEntry(BaseModel):
    version_id: str
    created_at: str
    reason: Literal["update", "restore"]
    snapshot: AgentConfig


class AgentCatalog(BaseModel):
    agents: list[AgentConfig] = Field(default_factory=list)
    audit_history: dict[str, list[AgentAuditEntry]] = Field(default_factory=dict)


class AgentConfigExport(BaseModel):
    export_version: Literal["1.0"] = "1.0"
    exported_at: str
    source: str = "local_agent_studio"
    agents: list[AgentConfig] = Field(default_factory=list)
    audit_history: dict[str, list[AgentAuditEntry]] = Field(default_factory=dict)


class AgentConfigImportRequest(BaseModel):
    payload: AgentConfigExport
    mode: Literal["replace", "merge"] = "replace"
    preserve_audit_history: bool = True


class AgentConfigImportResponse(BaseModel):
    mode: Literal["replace", "merge"]
    imported_agents: int
    created_agents: int
    updated_agents: int
    replaced_agents: int
    preserved_audit_entries: int


class ManagerConfig(BaseModel):
    max_steps: int = Field(default=6, ge=1, le=30)
    max_agent_calls: int = Field(default=12, ge=1, le=100)


class AutomationRuleBase(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    enabled: bool = True
    event_type: Literal["new_file"] = "new_file"
    watch_path: str = Field(min_length=1)
    recursive: bool = True
    file_extensions: list[str] = Field(default_factory=lambda: [".txt", ".md", ".json", ".csv"])
    poll_interval_seconds: int = Field(default=10, ge=2, le=3600)
    max_events_per_scan: int = Field(default=3, ge=1, le=20)
    prompt_template: str = (
        "A new file has been detected. Analyze it and execute the workflow.\n"
        "File path: {file_path}\n"
        "File name: {file_name}\n"
        "Event type: {event_type}"
    )
    agent_chain: list[str] = Field(default_factory=list)


class AutomationRule(AutomationRuleBase):
    id: str


class AutomationRuleCreate(AutomationRuleBase):
    pass


class AutomationRuleUpdate(AutomationRuleBase):
    pass


class AutomationRunStep(BaseModel):
    agent_id: str
    agent_name: str
    status: Literal["success", "failed", "skipped"]
    answer_preview: str | None = None
    row_count: int | None = None
    error: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class AutomationRunLog(BaseModel):
    id: str
    automation_id: str
    automation_name: str
    event_type: str
    event_file_path: str
    started_at: str
    finished_at: str | None = None
    status: Literal["success", "failed", "partial"]
    prompt: str
    steps: list[AutomationRunStep] = Field(default_factory=list)
    final_answer: str | None = None
    error: str | None = None


class AppSettings(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    webhook: WebhookConfig = Field(default_factory=WebhookConfig)
    manager: ManagerConfig = Field(default_factory=ManagerConfig)
    automations: list[AutomationRule] = Field(default_factory=list)
    databases: list[DatabaseProfile] = Field(default_factory=list)
    active_database_id: str | None = None


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=6000)


class AgentRunRequest(BaseModel):
    question: str = Field(min_length=1)
    database_id: str | None = None
    conversation_history: list[ConversationTurn] = Field(default_factory=list)


class AgentRunResponse(BaseModel):
    agent_id: str
    database_id: str | None = None
    sql: str
    rows: list[dict[str, Any]]
    answer: str
    details: dict[str, Any] = Field(default_factory=dict)


class ManagerRunRequest(BaseModel):
    question: str = Field(min_length=1)
    database_id: str | None = None
    max_steps: int = Field(default=6, ge=1, le=30)
    max_agent_calls: int = Field(default=12, ge=1, le=100)
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    export_intermediate_results_to_excel: bool = False


class ManagerRunResponse(BaseModel):
    status: Literal["done", "blocked", "exhausted"]
    answer: str
    manager_summary: str | None = None
    judge_verdict: Literal["pass", "partial", "fail"] | None = None
    judge_confidence: int | None = Field(default=None, ge=0, le=100)
    judge_rationale: str | None = None
    judge_checks_passed: list[str] = Field(default_factory=list)
    judge_checks_failed: list[str] = Field(default_factory=list)
    judge_recommendations: list[str] = Field(default_factory=list)
    missing_information: str | None = None
    intermediate_results_excel_path: str | None = None
    intermediate_results_excel_error: str | None = None
    steps: int
    agent_calls: int
    timeline: list[dict[str, Any]] = Field(default_factory=list)


class AgentTemplate(BaseModel):
    id: str
    name: str
    description: str
    defaults: AgentConfigCreate


def agent_requires_database(agent_type: AgentType) -> bool:
    return agent_type in {"sql_analyst", "clickhouse_table_manager"}
