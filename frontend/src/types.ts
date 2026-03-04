export type LLMProvider = "ollama" | "http";
export type DatabaseEngine = "clickhouse" | "oracle" | "elasticsearch";
export type AgentType =
  | "sql_analyst"
  | "knowledge_data_dictionary"
  | "clickhouse_table_manager"
  | "unstructured_to_structured"
  | "email_cleaner"
  | "file_assistant"
  | "text_file_manager"
  | "excel_manager"
  | "word_manager"
  | "elasticsearch_retriever"
  | "internet_search"
  | "rss_news"
  | "web_scraper"
  | "web_navigator"
  | "wikipedia_retriever"
  | "rag_context";

export interface LLMConfig {
  provider: LLMProvider;
  model: string;
  system_prompt: string;
  base_url: string;
  endpoint: string | null;
  api_key: string | null;
  timeout_seconds: number;
  temperature: number;
  headers: Record<string, string>;
}

export interface InternetAccessCheck {
  name: string;
  target: string;
  ok: boolean;
  detail: string;
  status_code: number | null;
  latency_ms: number | null;
}

export interface InternetAccessTestResponse {
  status: "ok" | "partial" | "blocked";
  message: string;
  checked_at: string;
  successful_checks: number;
  total_checks: number;
  checks: InternetAccessCheck[];
}

export interface ManagerConfig {
  max_steps: number;
  max_agent_calls: number;
}

export interface WebhookConfig {
  enabled: boolean;
  replace_playground: boolean;
  url: string;
  auth_token: string | null;
  timeout_seconds: number;
  verify_ssl: boolean;
  include_timeline_on_final: boolean;
  headers: Record<string, string>;
}

export interface DatabaseProfile {
  id: string;
  name: string;
  engine: DatabaseEngine;
  host: string | null;
  port: number | null;
  database: string | null;
  username: string | null;
  password: string | null;
  dsn: string | null;
  secure: boolean;
  options: Record<string, unknown>;
}

export interface DatabaseConfigExportPayload {
  export_version: "1.0";
  exported_at: string;
  source: string;
  active_database_id: string | null;
  databases: DatabaseProfile[];
}

export interface DatabaseConfigImportResponse {
  mode: "replace" | "merge";
  imported_databases: number;
  created_databases: number;
  updated_databases: number;
  replaced_databases: number;
  active_database_id: string | null;
}

export interface AgentConfig {
  id: string;
  name: string;
  agent_type: AgentType;
  description: string;
  system_prompt: string;
  sql_prompt_template: string;
  answer_prompt_template: string;
  allowed_tables: string[];
  max_rows: number;
  template_config: Record<string, unknown>;
  enabled: boolean;
}

export interface AgentTemplate {
  id: string;
  name: string;
  description: string;
  defaults: Omit<AgentConfig, "id">;
}

export interface AgentAuditEntry {
  version_id: string;
  created_at: string;
  reason: "update" | "restore";
  snapshot: AgentConfig;
}

export interface AgentConfigExportPayload {
  export_version: "1.0";
  exported_at: string;
  source: string;
  agents: AgentConfig[];
  audit_history: Record<string, AgentAuditEntry[]>;
}

export interface AgentConfigImportResponse {
  mode: "replace" | "merge";
  imported_agents: number;
  created_agents: number;
  updated_agents: number;
  replaced_agents: number;
  preserved_audit_entries: number;
}

export interface AppConfig {
  llm: LLMConfig;
  webhook: WebhookConfig;
  manager: ManagerConfig;
  automations: AutomationRule[];
  databases: DatabaseProfile[];
  active_database_id: string | null;
}

export interface AgentRunResponse {
  agent_id: string;
  database_id: string | null;
  sql: string;
  rows: Array<Record<string, unknown>>;
  answer: string;
  details: Record<string, unknown>;
}

export interface ConversationTurn {
  role: "user" | "assistant";
  content: string;
}

export type ManagerEventType =
  | "manager_start"
  | "manager_decision"
  | "manager_warning"
  | "agent_call_started"
  | "agent_call_completed"
  | "agent_call_failed"
  | "agent_marked_unavailable"
  | "manager_final";

export interface ManagerEvent {
  type: ManagerEventType;
  ts: string;
  [key: string]: unknown;
}

export interface ManagerFinalEvent extends ManagerEvent {
  type: "manager_final";
  status: "done" | "blocked" | "exhausted";
  answer: string;
  manager_summary?: string | null;
  judge_verdict?: "pass" | "partial" | "fail" | null;
  judge_confidence?: number | null;
  judge_rationale?: string | null;
  judge_checks_passed?: string[] | null;
  judge_checks_failed?: string[] | null;
  judge_recommendations?: string[] | null;
  missing_information?: string | null;
  intermediate_results_excel_path?: string | null;
  intermediate_results_excel_error?: string | null;
  steps: number;
  agent_calls: number;
}

export interface AutomationRule {
  id: string;
  name: string;
  enabled: boolean;
  event_type: "new_file";
  watch_path: string;
  recursive: boolean;
  file_extensions: string[];
  poll_interval_seconds: number;
  max_events_per_scan: number;
  prompt_template: string;
  agent_chain: string[];
}

export interface AutomationRunStep {
  agent_id: string;
  agent_name: string;
  status: "success" | "failed" | "skipped";
  answer_preview: string | null;
  row_count: number | null;
  error: string | null;
  details: Record<string, unknown>;
}

export interface AutomationRunLog {
  id: string;
  automation_id: string;
  automation_name: string;
  event_type: string;
  event_file_path: string;
  started_at: string;
  finished_at: string | null;
  status: "success" | "failed" | "partial";
  prompt: string;
  steps: AutomationRunStep[];
  final_answer: string | null;
  error: string | null;
}
