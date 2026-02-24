import { useEffect, useMemo, useRef, useState, type ChangeEvent } from "react";

import {
  createAutomation,
  createDatabase,
  deleteAutomation,
  deleteDatabase,
  exportDatabaseConnections,
  fetchAutomationRuns,
  fetchLLMModels,
  importDatabaseConnections,
  saveLLMConfig,
  saveManagerConfig,
  saveWebhookConfig,
  setActiveDatabase,
  testInternetAccess,
  testDatabase,
  testLLMConnection,
  testWebhookConfig,
  updateAutomation,
  updateDatabase
} from "../api";
import {
  AgentConfig,
  AppConfig,
  AutomationRunLog,
  AutomationRule,
  DatabaseConfigExportPayload,
  DatabaseProfile,
  InternetAccessTestResponse,
  LLMConfig,
  ManagerConfig,
  WebhookConfig
} from "../types";

interface ConfigurationTabProps {
  config: AppConfig;
  agents: AgentConfig[];
  onRefresh: () => Promise<void>;
  onNotify: (message: string, tone?: "success" | "error") => void;
}

interface DatabaseDraft {
  name: string;
  engine: "clickhouse" | "oracle" | "elasticsearch";
  host: string;
  port: number;
  database: string;
  username: string;
  password: string;
  dsn: string;
  secure: boolean;
  options: Record<string, unknown>;
}

interface AutomationDraft {
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

const EMPTY_DATABASE: DatabaseDraft = {
  name: "",
  engine: "clickhouse",
  host: "localhost",
  port: 8123,
  database: "default",
  username: "",
  password: "",
  dsn: "",
  secure: false,
  options: {}
};

const DEFAULT_AUTOMATION_PROMPT = `A new file has been detected. Analyze it and execute the workflow.
File path: {file_path}
File name: {file_name}
Event type: {event_type}`;

const EMPTY_AUTOMATION: AutomationDraft = {
  name: "",
  enabled: true,
  event_type: "new_file",
  watch_path: "",
  recursive: true,
  file_extensions: [".txt", ".md", ".json", ".csv"],
  poll_interval_seconds: 10,
  max_events_per_scan: 3,
  prompt_template: DEFAULT_AUTOMATION_PROMPT,
  agent_chain: []
};

function defaultPortForEngine(engine: DatabaseDraft["engine"]): number {
  if (engine === "oracle") {
    return 1521;
  }
  if (engine === "elasticsearch") {
    return 9200;
  }
  return 8123;
}

function toEditableDatabase(profile: DatabaseProfile): DatabaseDraft {
  return {
    name: profile.name,
    engine: profile.engine,
    host: profile.host ?? "",
    port: profile.port ?? defaultPortForEngine(profile.engine),
    database: profile.database ?? "",
    username: profile.username ?? "",
    password: profile.password ?? "",
    dsn: profile.dsn ?? "",
    secure: profile.secure,
    options: profile.options
  };
}

function toEditableAutomation(rule: AutomationRule): AutomationDraft {
  return {
    name: rule.name,
    enabled: rule.enabled,
    event_type: rule.event_type,
    watch_path: rule.watch_path,
    recursive: rule.recursive,
    file_extensions: rule.file_extensions,
    poll_interval_seconds: rule.poll_interval_seconds,
    max_events_per_scan: rule.max_events_per_scan,
    prompt_template: rule.prompt_template,
    agent_chain: rule.agent_chain
  };
}

function parseJsonObject(value: string, fieldName: string): Record<string, unknown> {
  if (!value.trim()) {
    return {};
  }
  try {
    const parsed = JSON.parse(value) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new Error(`${fieldName} must be a JSON object.`);
    }
    return parsed as Record<string, unknown>;
  } catch (error) {
    throw new Error(
      error instanceof Error ? error.message : `${fieldName} contains invalid JSON.`
    );
  }
}

function parseHeaderObject(value: string): Record<string, string> {
  const parsed = parseJsonObject(value, "Headers");
  const output: Record<string, string> = {};
  for (const [key, item] of Object.entries(parsed)) {
    output[key] = String(item);
  }
  return output;
}

function parseExtensions(value: string): string[] {
  if (!value.trim()) {
    return [];
  }
  return value
    .split(",")
    .map((item) => item.trim().toLowerCase())
    .filter(Boolean)
    .map((item) => {
      if (item === "*") {
        return "*";
      }
      return item.startsWith(".") ? item : `.${item}`;
    });
}

function normalizeDatabaseImportPayload(raw: unknown): DatabaseConfigExportPayload | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return null;
  }

  const candidate = raw as Record<string, unknown>;
  if (!Array.isArray(candidate.databases)) {
    return null;
  }

  const exportVersion =
    typeof candidate.export_version === "string" && candidate.export_version
      ? candidate.export_version
      : "1.0";
  const exportedAt =
    typeof candidate.exported_at === "string" && candidate.exported_at
      ? candidate.exported_at
      : new Date().toISOString();
  const source =
    typeof candidate.source === "string" && candidate.source
      ? candidate.source
      : "manual_import";
  const activeDatabaseId =
    typeof candidate.active_database_id === "string" && candidate.active_database_id
      ? candidate.active_database_id
      : null;

  return {
    export_version: exportVersion === "1.0" ? "1.0" : "1.0",
    exported_at: exportedAt,
    source,
    active_database_id: activeDatabaseId,
    databases: candidate.databases as DatabaseProfile[]
  };
}

export function ConfigurationTab({
  config,
  agents,
  onRefresh,
  onNotify
}: ConfigurationTabProps) {
  const [llmDraft, setLlmDraft] = useState<LLMConfig>(config.llm);
  const [webhookDraft, setWebhookDraft] = useState<WebhookConfig>(config.webhook);
  const [managerDraft, setManagerDraft] = useState<ManagerConfig>(config.manager);
  const [llmHeadersText, setLlmHeadersText] = useState(
    JSON.stringify(config.llm.headers ?? {}, null, 2)
  );
  const [webhookHeadersText, setWebhookHeadersText] = useState(
    JSON.stringify(config.webhook.headers ?? {}, null, 2)
  );
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [isRefreshingLlm, setIsRefreshingLlm] = useState(false);
  const [lastLlmRefresh, setLastLlmRefresh] = useState<string | null>(null);
  const [isTestingInternet, setIsTestingInternet] = useState(false);
  const [internetTestResult, setInternetTestResult] =
    useState<InternetAccessTestResponse | null>(null);
  const [isTestingWebhook, setIsTestingWebhook] = useState(false);
  const [lastWebhookTestAt, setLastWebhookTestAt] = useState<string | null>(null);

  const [selectedDatabaseId, setSelectedDatabaseId] = useState<string>(
    config.databases[0]?.id ?? "new"
  );
  const [databaseDraft, setDatabaseDraft] = useState<DatabaseDraft>({
    ...EMPTY_DATABASE
  });
  const [databaseOptionsText, setDatabaseOptionsText] = useState("{}");
  const [isTransferringDatabaseConfig, setIsTransferringDatabaseConfig] = useState(false);
  const databaseImportInputRef = useRef<HTMLInputElement | null>(null);
  const [selectedAutomationId, setSelectedAutomationId] = useState<string>(
    config.automations[0]?.id ?? "new"
  );
  const [automationDraft, setAutomationDraft] = useState<AutomationDraft>({
    ...EMPTY_AUTOMATION
  });
  const [automationExtensionsText, setAutomationExtensionsText] = useState(
    EMPTY_AUTOMATION.file_extensions.join(", ")
  );
  const [agentToAppend, setAgentToAppend] = useState<string>(agents[0]?.id ?? "");
  const [automationRuns, setAutomationRuns] = useState<AutomationRunLog[]>([]);
  const [isLoadingAutomationRuns, setIsLoadingAutomationRuns] = useState(false);

  const selectedDatabase = useMemo(
    () => config.databases.find((db) => db.id === selectedDatabaseId) ?? null,
    [config.databases, selectedDatabaseId]
  );
  const activeDatabaseProfile = useMemo(
    () => config.databases.find((db) => db.id === config.active_database_id) ?? null,
    [config.databases, config.active_database_id]
  );
  const selectedAutomation = useMemo(
    () => config.automations.find((item) => item.id === selectedAutomationId) ?? null,
    [config.automations, selectedAutomationId]
  );
  const agentNameById = useMemo(
    () => new Map<string, string>(agents.map((agent) => [agent.id, agent.name])),
    [agents]
  );

  const modelOptions = useMemo(() => {
    const values = new Set(availableModels);
    const current = llmDraft.model.trim();
    if (current) {
      values.add(current);
    }
    return Array.from(values);
  }, [availableModels, llmDraft.model]);

  useEffect(() => {
    setLlmDraft(config.llm);
    setLlmHeadersText(JSON.stringify(config.llm.headers ?? {}, null, 2));
  }, [config.llm]);

  useEffect(() => {
    setWebhookDraft(config.webhook);
    setWebhookHeadersText(JSON.stringify(config.webhook.headers ?? {}, null, 2));
  }, [config.webhook]);

  useEffect(() => {
    setManagerDraft(config.manager);
  }, [config.manager]);

  useEffect(() => {
    void (async () => {
      try {
        const result = await fetchLLMModels();
        setAvailableModels(result.models ?? []);
      } catch {
        setAvailableModels([]);
      }
    })();
  }, []);

  useEffect(() => {
    if (selectedDatabase) {
      const editable = toEditableDatabase(selectedDatabase);
      setDatabaseDraft(editable);
      setDatabaseOptionsText(JSON.stringify(editable.options, null, 2));
      return;
    }
    setDatabaseDraft({ ...EMPTY_DATABASE });
    setDatabaseOptionsText("{}");
  }, [selectedDatabase]);

  useEffect(() => {
    if (selectedDatabaseId === "new") {
      return;
    }
    if (!config.databases.some((item) => item.id === selectedDatabaseId)) {
      setSelectedDatabaseId(config.databases[0]?.id ?? "new");
    }
  }, [config.databases, selectedDatabaseId]);

  useEffect(() => {
    if (selectedAutomationId === "new") {
      return;
    }
    if (!config.automations.some((item) => item.id === selectedAutomationId)) {
      setSelectedAutomationId(config.automations[0]?.id ?? "new");
    }
  }, [config.automations, selectedAutomationId]);

  useEffect(() => {
    if (!selectedAutomation) {
      setAutomationDraft({ ...EMPTY_AUTOMATION });
      setAutomationExtensionsText(EMPTY_AUTOMATION.file_extensions.join(", "));
      return;
    }

    const editable = toEditableAutomation(selectedAutomation);
    setAutomationDraft(editable);
    setAutomationExtensionsText((editable.file_extensions ?? []).join(", "));
  }, [selectedAutomation]);

  useEffect(() => {
    if (!agents.some((agent) => agent.id === agentToAppend)) {
      setAgentToAppend(agents[0]?.id ?? "");
    }
  }, [agents, agentToAppend]);

  const loadAutomationRuns = async (automationId: string, notifyOnError = false) => {
    if (!automationId || automationId === "new") {
      setAutomationRuns([]);
      return;
    }

    try {
      setIsLoadingAutomationRuns(true);
      const runs = await fetchAutomationRuns(automationId, 30);
      setAutomationRuns(runs);
    } catch (error) {
      setAutomationRuns([]);
      if (notifyOnError) {
        onNotify(
          error instanceof Error ? error.message : "Unable to load automation runs.",
          "error"
        );
      }
    } finally {
      setIsLoadingAutomationRuns(false);
    }
  };

  useEffect(() => {
    void loadAutomationRuns(selectedAutomationId);
  }, [selectedAutomationId]);

  const buildLlmPayload = (): LLMConfig => ({
    ...llmDraft,
    system_prompt: llmDraft.system_prompt?.trim()
      ? llmDraft.system_prompt
      : "",
    endpoint: llmDraft.endpoint?.trim() ? llmDraft.endpoint : null,
    api_key: llmDraft.api_key?.trim() ? llmDraft.api_key : null,
    headers: parseHeaderObject(llmHeadersText)
  });

  const buildWebhookPayload = (): WebhookConfig => ({
    ...webhookDraft,
    url: webhookDraft.url.trim(),
    auth_token: webhookDraft.auth_token?.trim() ? webhookDraft.auth_token : null,
    headers: parseHeaderObject(webhookHeadersText)
  });

  const saveLlm = async () => {
    try {
      await saveLLMConfig(buildLlmPayload());
      await onRefresh();
      onNotify("LLM settings saved.");
    } catch (error) {
      onNotify(error instanceof Error ? error.message : "LLM update failed.", "error");
    }
  };

  const refreshLlm = async () => {
    try {
      setIsRefreshingLlm(true);
      await saveLLMConfig(buildLlmPayload());
      const [testResult, modelsResult] = await Promise.all([
        testLLMConnection(),
        fetchLLMModels()
      ]);
      setAvailableModels(modelsResult.models ?? []);
      setLastLlmRefresh(new Date().toLocaleTimeString());
      await onRefresh();
      onNotify(
        `${testResult.message} ${modelsResult.models.length} model(s) available.`
      );
    } catch (error) {
      onNotify(
        error instanceof Error ? error.message : "Unable to refresh LLM status.",
        "error"
      );
    } finally {
      setIsRefreshingLlm(false);
    }
  };

  const runInternetTest = async () => {
    try {
      setIsTestingInternet(true);
      const result = await testInternetAccess();
      setInternetTestResult(result);
      onNotify(
        `${result.message} (${String(result.successful_checks)}/${String(result.total_checks)} checks successful).`,
        result.status === "blocked" ? "error" : "success"
      );
    } catch (error) {
      onNotify(
        error instanceof Error ? error.message : "Internet access test failed.",
        "error"
      );
    } finally {
      setIsTestingInternet(false);
    }
  };

  const saveManager = async () => {
    try {
      await saveManagerConfig(managerDraft);
      await onRefresh();
      onNotify("Manager settings saved.");
    } catch (error) {
      onNotify(error instanceof Error ? error.message : "Manager update failed.", "error");
    }
  };

  const saveWebhook = async () => {
    try {
      const payload = buildWebhookPayload();
      await saveWebhookConfig(payload);
      await onRefresh();
      onNotify("Webhook settings saved.");
    } catch (error) {
      onNotify(error instanceof Error ? error.message : "Webhook update failed.", "error");
    }
  };

  const runWebhookTest = async () => {
    try {
      setIsTestingWebhook(true);
      const payload = buildWebhookPayload();
      const result = await testWebhookConfig(payload);
      setLastWebhookTestAt(new Date(result.tested_at).toLocaleTimeString());
      onNotify(
        `${result.message}${result.latency_ms !== null ? ` (${String(result.latency_ms)} ms)` : ""}`
      );
    } catch (error) {
      onNotify(error instanceof Error ? error.message : "Webhook test failed.", "error");
    } finally {
      setIsTestingWebhook(false);
    }
  };

  const saveDatabase = async () => {
    try {
      const payload = {
        ...databaseDraft,
        host: databaseDraft.host?.trim() ? databaseDraft.host : null,
        database: databaseDraft.database?.trim() ? databaseDraft.database : null,
        username: databaseDraft.username?.trim() ? databaseDraft.username : null,
        password: databaseDraft.password?.trim() ? databaseDraft.password : null,
        dsn: databaseDraft.dsn?.trim() ? databaseDraft.dsn : null,
        options: parseJsonObject(databaseOptionsText, "Options")
      };

      if (selectedDatabaseId === "new") {
        const created = await createDatabase(payload);
        setSelectedDatabaseId(created.id);
        onNotify("Database profile created.");
      } else {
        await updateDatabase(selectedDatabaseId, payload);
        onNotify("Database profile updated.");
      }

      await onRefresh();
    } catch (error) {
      onNotify(
        error instanceof Error ? error.message : "Database update failed.",
        "error"
      );
    }
  };

  const removeDatabase = async () => {
    if (selectedDatabaseId === "new") {
      return;
    }
    try {
      await deleteDatabase(selectedDatabaseId);
      setSelectedDatabaseId("new");
      await onRefresh();
      onNotify("Database profile deleted.");
    } catch (error) {
      onNotify(error instanceof Error ? error.message : "Delete failed.", "error");
    }
  };

  const runDatabaseTest = async () => {
    if (selectedDatabaseId === "new") {
      onNotify("Save this profile before testing the connection.", "error");
      return;
    }

    try {
      const result = await testDatabase(selectedDatabaseId);
      onNotify(`Connection test successful: ${result.message}`);
    } catch (error) {
      onNotify(error instanceof Error ? error.message : "Connection test failed.", "error");
    }
  };

  const activateDatabase = async () => {
    if (selectedDatabaseId === "new") {
      onNotify("Select an existing profile first.", "error");
      return;
    }
    try {
      await setActiveDatabase(selectedDatabaseId);
      await onRefresh();
      onNotify("Active database updated.");
    } catch (error) {
      onNotify(
        error instanceof Error ? error.message : "Could not set active database.",
        "error"
      );
    }
  };

  const startNewDatabase = () => {
    setSelectedDatabaseId("new");
    setDatabaseDraft({ ...EMPTY_DATABASE });
    setDatabaseOptionsText("{}");
  };

  const exportDatabaseConfig = async () => {
    try {
      setIsTransferringDatabaseConfig(true);
      const payload = await exportDatabaseConnections();
      const blob = new Blob([JSON.stringify(payload, null, 2)], {
        type: "application/json"
      });
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      const stamp = new Date().toISOString().replace(/[:.]/g, "-");
      link.download = `database-connections-${stamp}.json`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(link.href);
      onNotify("Database connections exported.");
    } catch (error) {
      onNotify(
        error instanceof Error ? error.message : "Database export failed.",
        "error"
      );
    } finally {
      setIsTransferringDatabaseConfig(false);
    }
  };

  const triggerDatabaseImport = () => {
    if (isTransferringDatabaseConfig) {
      return;
    }
    databaseImportInputRef.current?.click();
  };

  const importDatabaseConfig = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) {
      return;
    }

    try {
      setIsTransferringDatabaseConfig(true);
      const text = await file.text();
      const parsed = JSON.parse(text) as unknown;
      const payload = normalizeDatabaseImportPayload(parsed);
      if (!payload) {
        throw new Error(
          "Invalid import file format. Expected a database connections export JSON payload."
        );
      }

      const result = await importDatabaseConnections(payload, "replace");
      await onRefresh();

      const nextId =
        result.active_database_id ??
        payload.active_database_id ??
        payload.databases[0]?.id ??
        "new";
      setSelectedDatabaseId(nextId);
      onNotify(
        `Database connections imported. Profiles: ${String(result.imported_databases)}.`
      );
    } catch (error) {
      onNotify(
        error instanceof Error ? error.message : "Database import failed.",
        "error"
      );
    } finally {
      setIsTransferringDatabaseConfig(false);
    }
  };

  const startNewAutomation = () => {
    setSelectedAutomationId("new");
    setAutomationDraft({ ...EMPTY_AUTOMATION });
    setAutomationExtensionsText(EMPTY_AUTOMATION.file_extensions.join(", "));
    setAutomationRuns([]);
  };

  const addAgentToChain = () => {
    if (!agentToAppend) {
      return;
    }
    setAutomationDraft((previous) => ({
      ...previous,
      agent_chain: [...previous.agent_chain, agentToAppend]
    }));
  };

  const moveAgentInChain = (index: number, direction: -1 | 1) => {
    setAutomationDraft((previous) => {
      const nextIndex = index + direction;
      if (nextIndex < 0 || nextIndex >= previous.agent_chain.length) {
        return previous;
      }
      const chain = [...previous.agent_chain];
      const [target] = chain.splice(index, 1);
      chain.splice(nextIndex, 0, target);
      return {
        ...previous,
        agent_chain: chain
      };
    });
  };

  const removeAgentFromChain = (index: number) => {
    setAutomationDraft((previous) => ({
      ...previous,
      agent_chain: previous.agent_chain.filter((_, itemIndex) => itemIndex !== index)
    }));
  };

  const saveAutomation = async () => {
    try {
      if (!automationDraft.name.trim()) {
        onNotify("Automation rule name is required.", "error");
        return;
      }
      if (!automationDraft.watch_path.trim()) {
        onNotify("Watch path is required.", "error");
        return;
      }
      if (automationDraft.agent_chain.length === 0) {
        onNotify("Add at least one agent in the chain.", "error");
        return;
      }

      const payload: Omit<AutomationRule, "id"> = {
        ...automationDraft,
        name: automationDraft.name.trim(),
        watch_path: automationDraft.watch_path.trim(),
        file_extensions: parseExtensions(automationExtensionsText)
      };

      let targetAutomationId = selectedAutomationId;
      if (selectedAutomationId === "new") {
        const created = await createAutomation(payload);
        targetAutomationId = created.id;
        setSelectedAutomationId(created.id);
        onNotify("Automation rule created.");
      } else {
        await updateAutomation(selectedAutomationId, payload);
        onNotify("Automation rule updated.");
      }

      await onRefresh();
      await loadAutomationRuns(targetAutomationId);
    } catch (error) {
      onNotify(
        error instanceof Error ? error.message : "Automation update failed.",
        "error"
      );
    }
  };

  const removeAutomation = async () => {
    if (selectedAutomationId === "new") {
      return;
    }
    try {
      await deleteAutomation(selectedAutomationId);
      startNewAutomation();
      await onRefresh();
      onNotify("Automation rule deleted.");
    } catch (error) {
      onNotify(
        error instanceof Error ? error.message : "Automation delete failed.",
        "error"
      );
    }
  };

  return (
    <div className="tab-layout">
      <section className="card">
        <header className="card-header">
          <h2>LLM Settings</h2>
          <div className="button-row">
            <button className="btn-ghost" onClick={refreshLlm} disabled={isRefreshingLlm}>
              {isRefreshingLlm ? "Refreshing..." : "Refresh"}
            </button>
            <button className="btn-ghost" onClick={runInternetTest} disabled={isTestingInternet}>
              {isTestingInternet ? "Testing internet..." : "Test internet access"}
            </button>
            <button className="btn-primary" onClick={saveLlm}>
              Save
            </button>
          </div>
        </header>

        <p className="hint">
          Models detected: <strong>{availableModels.length}</strong>
          {lastLlmRefresh ? ` • Last refresh: ${lastLlmRefresh}` : ""}
        </p>

        {internetTestResult && (
          <section className={`internet-test-card ${internetTestResult.status}`}>
            <p className="hint">
              Internet status: <strong>{internetTestResult.status.toUpperCase()}</strong> •{" "}
              {internetTestResult.message}
            </p>
            <p className="hint">
              Successful checks: <strong>{internetTestResult.successful_checks}</strong> /{" "}
              <strong>{internetTestResult.total_checks}</strong> • Checked at{" "}
              <strong>{new Date(internetTestResult.checked_at).toLocaleTimeString()}</strong>
            </p>
            <div className="internet-check-list">
              {internetTestResult.checks.map((check, index) => (
                <article
                  className={check.ok ? "internet-check-item ok" : "internet-check-item failed"}
                  key={`${check.name}-${check.target}-${String(index)}`}
                >
                  <div className="timeline-header">
                    <span className="timeline-type">{check.name}</span>
                    <span className="timeline-time">{check.ok ? "OK" : "FAILED"}</span>
                  </div>
                  <p className="hint">{check.target}</p>
                  <p className="hint">
                    {check.detail}
                    {check.status_code !== null ? ` • HTTP ${String(check.status_code)}` : ""}
                    {check.latency_ms !== null ? ` • ${String(check.latency_ms)} ms` : ""}
                  </p>
                </article>
              ))}
            </div>
          </section>
        )}

        <div className="grid two-columns">
          <label>
            Provider
            <select
              value={llmDraft.provider}
              onChange={(event) =>
                setLlmDraft((prev) => ({
                  ...prev,
                  provider: event.target.value as "ollama" | "http"
                }))
              }
            >
              <option value="ollama">Ollama</option>
              <option value="http">HTTP</option>
            </select>
          </label>

          <label>
            Available models
            <select
              value={llmDraft.model}
              onChange={(event) =>
                setLlmDraft((prev) => ({ ...prev, model: event.target.value }))
              }
            >
              {modelOptions.length === 0 ? (
                <option value={llmDraft.model || ""}>No model available (refresh)</option>
              ) : (
                modelOptions.map((modelName) => (
                  <option value={modelName} key={modelName}>
                    {modelName}
                  </option>
                ))
              )}
            </select>
          </label>

          <label>
            Model (manual override)
            <input
              value={llmDraft.model}
              onChange={(event) =>
                setLlmDraft((prev) => ({ ...prev, model: event.target.value }))
              }
              placeholder="llama3.1"
            />
          </label>

          <label className="full-width">
            Global system prompt (optional)
            <textarea
              rows={4}
              value={llmDraft.system_prompt}
              onChange={(event) =>
                setLlmDraft((prev) => ({ ...prev, system_prompt: event.target.value }))
              }
              placeholder="Applied to all LLM calls (manager and agents)."
            />
          </label>

          <label>
            Base URL
            <input
              value={llmDraft.base_url}
              onChange={(event) =>
                setLlmDraft((prev) => ({ ...prev, base_url: event.target.value }))
              }
              placeholder="http://localhost:11434"
            />
          </label>

          <label>
            HTTP endpoint (optional)
            <input
              value={llmDraft.endpoint ?? ""}
              onChange={(event) =>
                setLlmDraft((prev) => ({ ...prev, endpoint: event.target.value }))
              }
              placeholder="http://localhost:8001/generate"
            />
          </label>

          <label>
            API key (optional)
            <input
              value={llmDraft.api_key ?? ""}
              onChange={(event) =>
                setLlmDraft((prev) => ({ ...prev, api_key: event.target.value }))
              }
              type="password"
            />
          </label>

          <label>
            Timeout (seconds)
            <input
              type="number"
              min={1}
              max={300}
              value={llmDraft.timeout_seconds}
              onChange={(event) =>
                setLlmDraft((prev) => ({
                  ...prev,
                  timeout_seconds: Number(event.target.value)
                }))
              }
            />
          </label>

          <label>
            Temperature
            <input
              type="number"
              step="0.1"
              min={0}
              max={2}
              value={llmDraft.temperature}
              onChange={(event) =>
                setLlmDraft((prev) => ({
                  ...prev,
                  temperature: Number(event.target.value)
                }))
              }
            />
          </label>

          <label className="full-width">
            Headers JSON (optional)
            <textarea
              rows={5}
              value={llmHeadersText}
              onChange={(event) => setLlmHeadersText(event.target.value)}
            />
          </label>
        </div>
      </section>

      <section className="card">
        <header className="card-header">
          <h2>External Webhook UI</h2>
          <div className="button-row">
            <button className="btn-ghost" onClick={runWebhookTest} disabled={isTestingWebhook}>
              {isTestingWebhook ? "Testing..." : "Test webhook"}
            </button>
            <button className="btn-primary" onClick={saveWebhook}>
              Save
            </button>
          </div>
        </header>

        <p className="hint">
          Send manager events in real time to an external frontend (for example Open WebUI)
          with full step-by-step traces and final answer.
        </p>
        <p className="hint">
          Event stream includes: manager_start, manager_decision, manager_warning,
          agent_call_started/completed/failed, agent_marked_unavailable, manager_final.
          {lastWebhookTestAt ? ` • Last test: ${lastWebhookTestAt}` : ""}
        </p>

        <div className="grid two-columns">
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={webhookDraft.enabled}
              onChange={(event) =>
                setWebhookDraft((previous) => ({
                  ...previous,
                  enabled: event.target.checked
                }))
              }
            />
            Enable webhook forwarding
          </label>

          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={webhookDraft.replace_playground}
              onChange={(event) =>
                setWebhookDraft((previous) => ({
                  ...previous,
                  replace_playground: event.target.checked
                }))
              }
            />
            Replace built-in Playground with external UI
          </label>

          <label className="full-width">
            Webhook URL
            <input
              value={webhookDraft.url}
              onChange={(event) =>
                setWebhookDraft((previous) => ({
                  ...previous,
                  url: event.target.value
                }))
              }
              placeholder="http://localhost:3000/webhook/agent-events"
            />
          </label>

          <label>
            Auth token (optional)
            <input
              type="password"
              value={webhookDraft.auth_token ?? ""}
              onChange={(event) =>
                setWebhookDraft((previous) => ({
                  ...previous,
                  auth_token: event.target.value
                }))
              }
              placeholder="Bearer token (Authorization header)"
            />
          </label>

          <label>
            Timeout (seconds)
            <input
              type="number"
              min={1}
              max={60}
              value={webhookDraft.timeout_seconds}
              onChange={(event) =>
                setWebhookDraft((previous) => ({
                  ...previous,
                  timeout_seconds: Math.max(
                    1,
                    Math.min(60, Number(event.target.value) || 1)
                  )
                }))
              }
            />
          </label>

          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={webhookDraft.verify_ssl}
              onChange={(event) =>
                setWebhookDraft((previous) => ({
                  ...previous,
                  verify_ssl: event.target.checked
                }))
              }
            />
            Verify SSL certificate
          </label>

          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={webhookDraft.include_timeline_on_final}
              onChange={(event) =>
                setWebhookDraft((previous) => ({
                  ...previous,
                  include_timeline_on_final: event.target.checked
                }))
              }
            />
            Include full timeline in manager_final payload
          </label>

          <label className="full-width">
            Additional headers JSON (optional)
            <textarea
              rows={4}
              value={webhookHeadersText}
              onChange={(event) => setWebhookHeadersText(event.target.value)}
              placeholder='{"X-Workspace": "local-agent"}'
            />
          </label>
        </div>
      </section>

      <section className="card">
        <header className="card-header">
          <h2>Manager Settings</h2>
          <button className="btn-primary" onClick={saveManager}>
            Save
          </button>
        </header>

        <p className="hint">
          Define the orchestration budget used by the multi-agent manager in the Playground.
        </p>

        <div className="grid two-columns">
          <label>
            Maximum orchestration steps
            <input
              type="number"
              min={1}
              max={30}
              value={managerDraft.max_steps}
              onChange={(event) =>
                setManagerDraft((previous) => {
                  const parsed = Number(event.target.value);
                  const normalized = Number.isFinite(parsed)
                    ? Math.max(1, Math.min(30, parsed))
                    : 1;
                  return {
                    ...previous,
                    max_steps: normalized
                  };
                })
              }
            />
          </label>

          <label>
            Maximum agent calls
            <input
              type="number"
              min={1}
              max={100}
              value={managerDraft.max_agent_calls}
              onChange={(event) =>
                setManagerDraft((previous) => {
                  const parsed = Number(event.target.value);
                  const normalized = Number.isFinite(parsed)
                    ? Math.max(1, Math.min(100, parsed))
                    : 1;
                  return {
                    ...previous,
                    max_agent_calls: normalized
                  };
                })
              }
            />
          </label>
        </div>
      </section>

      <section className="card">
        <header className="card-header">
          <h2>Automation Scheduling</h2>
          <div className="button-row">
            <button className="btn-ghost" onClick={startNewAutomation}>
              New rule
            </button>
            <button className="btn-primary" onClick={saveAutomation}>
              Save
            </button>
          </div>
        </header>

        <p className="hint">
          Trigger a chain of agents automatically when an event occurs (example: a new
          file appears in a folder).
        </p>

        <div className="toolbar">
          <label>
            Rule
            <select
              value={selectedAutomationId}
              onChange={(event) => setSelectedAutomationId(event.target.value)}
            >
              <option value="new">New</option>
              {config.automations.map((automation) => (
                <option value={automation.id} key={automation.id}>
                  {automation.name}
                </option>
              ))}
            </select>
          </label>

          <button
            className="btn-ghost"
            onClick={() => void loadAutomationRuns(selectedAutomationId, true)}
            disabled={selectedAutomationId === "new" || isLoadingAutomationRuns}
          >
            {isLoadingAutomationRuns ? "Refreshing runs..." : "Refresh runs"}
          </button>
          <button className="btn-danger" onClick={removeAutomation}>
            Delete
          </button>
        </div>

        <div className="grid two-columns">
          <label>
            Rule name
            <input
              value={automationDraft.name}
              onChange={(event) =>
                setAutomationDraft((previous) => ({
                  ...previous,
                  name: event.target.value
                }))
              }
              placeholder="Example: Inbox file triage"
            />
          </label>

          <label>
            Event type
            <select
              value={automationDraft.event_type}
              onChange={(event) =>
                setAutomationDraft((previous) => ({
                  ...previous,
                  event_type: event.target.value as "new_file"
                }))
              }
            >
              <option value="new_file">New file detected</option>
            </select>
          </label>

          <label className="full-width">
            Watch folder path
            <input
              value={automationDraft.watch_path}
              onChange={(event) =>
                setAutomationDraft((previous) => ({
                  ...previous,
                  watch_path: event.target.value
                }))
              }
              placeholder="/Users/your-user/Downloads"
            />
          </label>

          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={automationDraft.recursive}
              onChange={(event) =>
                setAutomationDraft((previous) => ({
                  ...previous,
                  recursive: event.target.checked
                }))
              }
            />
            Include subfolders
          </label>

          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={automationDraft.enabled}
              onChange={(event) =>
                setAutomationDraft((previous) => ({
                  ...previous,
                  enabled: event.target.checked
                }))
              }
            />
            Enabled
          </label>

          <label>
            Poll interval (seconds)
            <input
              type="number"
              min={2}
              max={3600}
              value={automationDraft.poll_interval_seconds}
              onChange={(event) =>
                setAutomationDraft((previous) => ({
                  ...previous,
                  poll_interval_seconds: Math.max(
                    2,
                    Math.min(3600, Number(event.target.value) || 2)
                  )
                }))
              }
            />
          </label>

          <label>
            Max events per scan
            <input
              type="number"
              min={1}
              max={20}
              value={automationDraft.max_events_per_scan}
              onChange={(event) =>
                setAutomationDraft((previous) => ({
                  ...previous,
                  max_events_per_scan: Math.max(
                    1,
                    Math.min(20, Number(event.target.value) || 1)
                  )
                }))
              }
            />
          </label>

          <label className="full-width">
            File extensions (comma-separated)
            <input
              value={automationExtensionsText}
              onChange={(event) => setAutomationExtensionsText(event.target.value)}
              placeholder=".txt, .md, .json, .csv or *"
            />
          </label>

          <label className="full-width">
            Prompt template
            <textarea
              rows={5}
              value={automationDraft.prompt_template}
              onChange={(event) =>
                setAutomationDraft((previous) => ({
                  ...previous,
                  prompt_template: event.target.value
                }))
              }
              placeholder="Use placeholders: {file_path}, {file_name}, {event_type}"
            />
          </label>
        </div>

        <section className="card">
          <div className="card-header">
            <h3>Agent chain</h3>
          </div>

          <div className="toolbar">
            <label>
              Add agent
              <select
                value={agentToAppend}
                onChange={(event) => setAgentToAppend(event.target.value)}
              >
                {agents.map((agent) => (
                  <option value={agent.id} key={agent.id}>
                    {agent.name} ({agent.agent_type})
                  </option>
                ))}
              </select>
            </label>
            <button className="btn-ghost" onClick={addAgentToChain} disabled={!agentToAppend}>
              Add to chain
            </button>
          </div>

          {automationDraft.agent_chain.length === 0 ? (
            <p className="hint">No agent in chain yet.</p>
          ) : (
            <div className="automation-chain">
              {automationDraft.agent_chain.map((agentId, index) => (
                <article className="automation-chain-item" key={`${agentId}-${index}`}>
                  <div>
                    <strong>
                      {index + 1}. {agentNameById.get(agentId) ?? `Unknown agent (${agentId})`}
                    </strong>
                    <p className="hint">{agentId}</p>
                  </div>

                  <div className="button-row">
                    <button
                      className="btn-ghost"
                      onClick={() => moveAgentInChain(index, -1)}
                      disabled={index === 0}
                    >
                      Up
                    </button>
                    <button
                      className="btn-ghost"
                      onClick={() => moveAgentInChain(index, 1)}
                      disabled={index === automationDraft.agent_chain.length - 1}
                    >
                      Down
                    </button>
                    <button
                      className="btn-danger"
                      onClick={() => removeAgentFromChain(index)}
                    >
                      Remove
                    </button>
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>

        <section className="card">
          <div className="card-header">
            <h3>Recent automation runs</h3>
            <button
              className="btn-ghost"
              onClick={() => void loadAutomationRuns(selectedAutomationId, true)}
              disabled={selectedAutomationId === "new" || isLoadingAutomationRuns}
            >
              {isLoadingAutomationRuns ? "Loading..." : "Refresh"}
            </button>
          </div>

          {selectedAutomationId === "new" ? (
            <p className="hint">Save this rule to start receiving execution logs.</p>
          ) : isLoadingAutomationRuns ? (
            <p className="hint">Loading runs...</p>
          ) : automationRuns.length === 0 ? (
            <p className="hint">No runs yet for this rule.</p>
          ) : (
            <div className="timeline">
              {automationRuns.map((run) => (
                <article className="timeline-item" key={run.id}>
                  <div className="timeline-header">
                    <span className="timeline-type">
                      {run.status} • {run.event_type}
                    </span>
                    <span className="timeline-time">
                      {new Date(run.started_at).toLocaleString()}
                    </span>
                  </div>
                  <p className="hint">
                    File: <strong>{run.event_file_path}</strong>
                  </p>
                  {run.error && (
                    <p className="hint">
                      Error: <strong>{run.error}</strong>
                    </p>
                  )}
                  {run.final_answer && <p>{run.final_answer}</p>}
                  <pre>{JSON.stringify(run.steps, null, 2)}</pre>
                </article>
              ))}
            </div>
          )}
        </section>
      </section>

      <section className="card">
        <header className="card-header">
          <h2>Database Settings</h2>
          <div className="button-row">
            <input
              ref={databaseImportInputRef}
              type="file"
              accept=".json,application/json"
              className="visually-hidden"
              onChange={importDatabaseConfig}
            />
            <button className="btn-ghost" onClick={triggerDatabaseImport}>
              Import
            </button>
            <button
              className="btn-ghost"
              onClick={exportDatabaseConfig}
              disabled={isTransferringDatabaseConfig}
            >
              Export
            </button>
            <button className="btn-ghost" onClick={startNewDatabase}>
              New profile
            </button>
            <button className="btn-primary" onClick={saveDatabase}>
              Save
            </button>
          </div>
        </header>

        <div className="toolbar">
          <label>
            Profile
            <select
              value={selectedDatabaseId}
              onChange={(event) => setSelectedDatabaseId(event.target.value)}
            >
              <option value="new">New</option>
              {config.databases.map((db) => (
                <option value={db.id} key={db.id}>
                  {db.name} ({db.engine})
                </option>
              ))}
            </select>
          </label>

          <button className="btn-ghost" onClick={activateDatabase}>
            Set active
          </button>
          <button className="btn-ghost" onClick={runDatabaseTest}>
            Test connection
          </button>
          <button className="btn-danger" onClick={removeDatabase}>
            Delete
          </button>
        </div>

        <p className="hint">
          Current active database:{" "}
          <strong>{activeDatabaseProfile?.name ?? config.active_database_id ?? "None"}</strong>
        </p>
        <p className="hint">
          Export/Import lets you back up all DB connections and restore them later in one
          step.
        </p>

        <div className="database-summary-grid">
          <article className="database-summary-item">
            <span>Total profiles</span>
            <strong>{String(config.databases.length)}</strong>
          </article>
          <article className="database-summary-item">
            <span>ClickHouse</span>
            <strong>
              {String(config.databases.filter((profile) => profile.engine === "clickhouse").length)}
            </strong>
          </article>
          <article className="database-summary-item">
            <span>Oracle</span>
            <strong>
              {String(config.databases.filter((profile) => profile.engine === "oracle").length)}
            </strong>
          </article>
          <article className="database-summary-item">
            <span>Elasticsearch</span>
            <strong>
              {String(
                config.databases.filter((profile) => profile.engine === "elasticsearch").length
              )}
            </strong>
          </article>
        </div>

        {config.databases.length > 0 && (
          <div className="database-profile-list">
            {config.databases.map((profile) => (
              <article
                key={profile.id}
                className={`database-profile-item ${
                  profile.id === selectedDatabaseId ? "selected" : ""
                }`}
              >
                <div className="database-profile-head">
                  <strong>{profile.name}</strong>
                  <div className="button-row">
                    {profile.id === config.active_database_id && (
                      <span className="status-pill done">Active</span>
                    )}
                    <button
                      className="btn-ghost"
                      onClick={() => setSelectedDatabaseId(profile.id)}
                    >
                      Edit
                    </button>
                  </div>
                </div>
                <p className="hint">
                  <strong>{profile.engine}</strong>
                  {profile.host ? ` • ${profile.host}` : ""}
                  {profile.port ? `:${String(profile.port)}` : ""}
                  {profile.database ? ` • ${profile.database}` : ""}
                </p>
                <p className="hint">ID: {profile.id}</p>
              </article>
            ))}
          </div>
        )}

        <div className="grid two-columns">
          <label>
            Name
            <input
              value={databaseDraft.name}
              onChange={(event) =>
                setDatabaseDraft((prev) => ({ ...prev, name: event.target.value }))
              }
            />
          </label>

          <label>
            Engine
            <select
              value={databaseDraft.engine}
              onChange={(event) =>
                setDatabaseDraft((prev) => ({
                  ...prev,
                  engine: event.target.value as DatabaseDraft["engine"],
                  port: defaultPortForEngine(event.target.value as DatabaseDraft["engine"])
                }))
              }
            >
              <option value="clickhouse">ClickHouse</option>
              <option value="oracle">Oracle</option>
              <option value="elasticsearch">Elasticsearch</option>
            </select>
          </label>

          <label>
            Host
            <input
              value={databaseDraft.host}
              onChange={(event) =>
                setDatabaseDraft((prev) => ({ ...prev, host: event.target.value }))
              }
              placeholder={
                databaseDraft.engine === "elasticsearch"
                  ? "localhost or https://my-es-host"
                  : "localhost"
              }
            />
          </label>

          <label>
            Port
            <input
              type="number"
              value={databaseDraft.port}
              onChange={(event) =>
                setDatabaseDraft((prev) => ({
                  ...prev,
                  port: Number(event.target.value)
                }))
              }
            />
          </label>

          <label>
            {databaseDraft.engine === "elasticsearch"
              ? "Default index (optional)"
              : "Database / Service"}
            <input
              value={databaseDraft.database}
              onChange={(event) =>
                setDatabaseDraft((prev) => ({ ...prev, database: event.target.value }))
              }
            />
          </label>

          <label>
            Username
            <input
              value={databaseDraft.username}
              onChange={(event) =>
                setDatabaseDraft((prev) => ({ ...prev, username: event.target.value }))
              }
            />
          </label>

          <label>
            Password
            <input
              type="password"
              value={databaseDraft.password}
              onChange={(event) =>
                setDatabaseDraft((prev) => ({ ...prev, password: event.target.value }))
              }
            />
          </label>

          {databaseDraft.engine === "oracle" && (
            <label>
              DSN (Oracle optional)
              <input
                value={databaseDraft.dsn}
                onChange={(event) =>
                  setDatabaseDraft((prev) => ({ ...prev, dsn: event.target.value }))
                }
                placeholder="host:1521/service_name"
              />
            </label>
          )}

          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={databaseDraft.secure}
              onChange={(event) =>
                setDatabaseDraft((prev) => ({ ...prev, secure: event.target.checked }))
              }
            />
            TLS / Secure
          </label>

          <label className="full-width">
            Options JSON
            <textarea
              rows={5}
              value={databaseOptionsText}
              onChange={(event) => setDatabaseOptionsText(event.target.value)}
            />
            {databaseDraft.engine === "elasticsearch" && (
              <span className="hint">
                Optional keys for Elasticsearch: <code>api_key</code>, <code>verify_ssl</code>, <code>timeout_seconds</code>, <code>headers</code>.
              </span>
            )}
          </label>
        </div>
      </section>
    </div>
  );
}
