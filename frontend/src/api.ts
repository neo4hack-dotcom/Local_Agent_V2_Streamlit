import {
  AgentConfigExportPayload,
  AgentConfigImportResponse,
  AgentAuditEntry,
  AgentConfig,
  AgentTemplate,
  AgentRunResponse,
  ConversationTurn,
  AutomationRule,
  AutomationRunLog,
  AppConfig,
  DatabaseConfigExportPayload,
  DatabaseConfigImportResponse,
  DatabaseProfile,
  InternetAccessTestResponse,
  LLMConfig,
  ManagerConfig,
  ManagerEvent,
  ManagerFinalEvent,
  WebhookConfig
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    },
    ...init
  });

  if (!response.ok) {
    const contentType = response.headers.get("content-type") ?? "";
    if (contentType.includes("application/json")) {
      const errorBody = (await response.json()) as { detail?: string };
      throw new Error(errorBody.detail ?? `API error: ${response.status}`);
    }
    throw new Error(`API error: ${response.status}`);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

export function fetchConfig(): Promise<AppConfig> {
  return request<AppConfig>("/config");
}

export function saveLLMConfig(payload: LLMConfig): Promise<LLMConfig> {
  return request<LLMConfig>("/config/llm", {
    method: "PUT",
    body: JSON.stringify(payload)
  });
}

export function saveManagerConfig(payload: ManagerConfig): Promise<ManagerConfig> {
  return request<ManagerConfig>("/config/manager", {
    method: "PUT",
    body: JSON.stringify(payload)
  });
}

export function saveWebhookConfig(payload: WebhookConfig): Promise<WebhookConfig> {
  return request<WebhookConfig>("/config/webhook", {
    method: "PUT",
    body: JSON.stringify(payload)
  });
}

export function testWebhookConfig(payload: WebhookConfig): Promise<{
  status: string;
  message: string;
  url: string;
  status_code: number | null;
  latency_ms: number | null;
  response_preview: string | null;
  tested_at: string;
}> {
  return request("/config/webhook/test", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function testLLMConnection(): Promise<{
  status: string;
  provider: string;
  message: string;
  model_count: number;
  models: string[];
}> {
  return request("/config/llm/test", {
    method: "POST"
  });
}

export function fetchLLMModels(): Promise<{ provider: string; models: string[] }> {
  return request("/config/llm/models");
}

export function testInternetAccess(): Promise<InternetAccessTestResponse> {
  return request<InternetAccessTestResponse>("/config/network/test", {
    method: "POST"
  });
}

export function createDatabase(
  payload: Omit<DatabaseProfile, "id">
): Promise<DatabaseProfile> {
  return request<DatabaseProfile>("/databases", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function updateDatabase(
  id: string,
  payload: Omit<DatabaseProfile, "id">
): Promise<DatabaseProfile> {
  return request<DatabaseProfile>(`/databases/${id}`, {
    method: "PUT",
    body: JSON.stringify(payload)
  });
}

export function deleteDatabase(id: string): Promise<void> {
  return request<void>(`/databases/${id}`, {
    method: "DELETE"
  });
}

export function setActiveDatabase(id: string): Promise<AppConfig> {
  return request<AppConfig>(`/databases/active/${id}`, {
    method: "PUT"
  });
}

export function testDatabase(id: string): Promise<{ message: string; status: string }> {
  return request<{ message: string; status: string }>(`/databases/${id}/test`, {
    method: "POST"
  });
}

export function exportDatabaseConnections(): Promise<DatabaseConfigExportPayload> {
  return request<DatabaseConfigExportPayload>("/databases/export");
}

export function importDatabaseConnections(
  payload: DatabaseConfigExportPayload,
  mode: "replace" | "merge" = "replace"
): Promise<DatabaseConfigImportResponse> {
  return request<DatabaseConfigImportResponse>("/databases/import", {
    method: "POST",
    body: JSON.stringify({
      payload,
      mode
    })
  });
}

export function fetchAgents(): Promise<AgentConfig[]> {
  return request<AgentConfig[]>("/agents");
}

export function fetchAgentTemplates(): Promise<AgentTemplate[]> {
  return request<AgentTemplate[]>("/agents/templates");
}

export function createAgent(payload: Omit<AgentConfig, "id">): Promise<AgentConfig> {
  return request<AgentConfig>("/agents", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function updateAgent(
  id: string,
  payload: Omit<AgentConfig, "id">
): Promise<AgentConfig> {
  return request<AgentConfig>(`/agents/${id}`, {
    method: "PUT",
    body: JSON.stringify(payload)
  });
}

export function deleteAgent(id: string): Promise<void> {
  return request<void>(`/agents/${id}`, {
    method: "DELETE"
  });
}

export function fetchAgentAudit(id: string): Promise<AgentAuditEntry[]> {
  return request<AgentAuditEntry[]>(`/agents/${id}/audit`);
}

export function exportAgentsConfig(): Promise<AgentConfigExportPayload> {
  return request<AgentConfigExportPayload>("/agents/export");
}

export function importAgentsConfig(
  payload: AgentConfigExportPayload,
  mode: "replace" | "merge" = "replace"
): Promise<AgentConfigImportResponse> {
  return request<AgentConfigImportResponse>("/agents/import", {
    method: "POST",
    body: JSON.stringify({
      payload,
      mode,
      preserve_audit_history: true
    })
  });
}

export function restoreAgentAuditVersion(
  id: string,
  versionId: string
): Promise<AgentConfig> {
  return request<AgentConfig>(`/agents/${id}/audit/${versionId}/restore`, {
    method: "POST"
  });
}

export function fetchAutomations(): Promise<AutomationRule[]> {
  return request<AutomationRule[]>("/automations");
}

export function createAutomation(
  payload: Omit<AutomationRule, "id">
): Promise<AutomationRule> {
  return request<AutomationRule>("/automations", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function updateAutomation(
  id: string,
  payload: Omit<AutomationRule, "id">
): Promise<AutomationRule> {
  return request<AutomationRule>(`/automations/${id}`, {
    method: "PUT",
    body: JSON.stringify(payload)
  });
}

export function deleteAutomation(id: string): Promise<void> {
  return request<void>(`/automations/${id}`, {
    method: "DELETE"
  });
}

export function fetchAutomationRuns(
  automationId: string,
  limit = 30
): Promise<AutomationRunLog[]> {
  return request<AutomationRunLog[]>(
    `/automations/${automationId}/runs?limit=${String(limit)}`
  );
}

export function runAgent(
  id: string,
  payload: {
    question: string;
    database_id?: string;
    conversation_history?: ConversationTurn[];
  }
): Promise<AgentRunResponse> {
  return request<AgentRunResponse>(`/agents/${id}/run`, {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export async function runManagerStream(
  payload: {
    question: string;
    database_id?: string;
    max_steps?: number;
    max_agent_calls?: number;
    conversation_history?: ConversationTurn[];
  },
  onEvent: (event: ManagerEvent) => void
): Promise<ManagerFinalEvent> {
  const response = await fetch(`${API_BASE}/agents/manager/run/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    const contentType = response.headers.get("content-type") ?? "";
    if (contentType.includes("application/json")) {
      const errorBody = (await response.json()) as { detail?: string };
      throw new Error(errorBody.detail ?? `API error: ${response.status}`);
    }
    throw new Error(`API error: ${response.status}`);
  }

  if (!response.body) {
    throw new Error("Streaming response is not available.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalEvent: ManagerFinalEvent | null = null;

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });

    let newlineIndex = buffer.indexOf("\n");
    while (newlineIndex >= 0) {
      const line = buffer.slice(0, newlineIndex).trim();
      buffer = buffer.slice(newlineIndex + 1);
      if (line) {
        const event = JSON.parse(line) as ManagerEvent;
        onEvent(event);
        if (event.type === "manager_final") {
          finalEvent = event as ManagerFinalEvent;
        }
      }
      newlineIndex = buffer.indexOf("\n");
    }

    if (done) {
      break;
    }
  }

  if (!finalEvent) {
    throw new Error("Manager stream ended without a final event.");
  }

  return finalEvent;
}
