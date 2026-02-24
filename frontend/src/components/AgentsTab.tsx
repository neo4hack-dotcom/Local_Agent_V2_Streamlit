import { useEffect, useMemo, useRef, useState, type ChangeEvent } from "react";

import {
  createAgent,
  deleteAgent,
  exportAgentsConfig,
  fetchAgentAudit,
  fetchAgentTemplates,
  importAgentsConfig,
  restoreAgentAuditVersion,
  saveManagerConfig,
  updateAgent
} from "../api";
import {
  AgentAuditEntry,
  AgentConfig,
  AgentConfigExportPayload,
  AgentTemplate,
  ManagerConfig
} from "../types";

interface AgentsTabProps {
  agents: AgentConfig[];
  managerConfig: ManagerConfig;
  onRefresh: () => Promise<void>;
  onNotify: (message: string, tone?: "success" | "error") => void;
}

type AgentDraft = Omit<AgentConfig, "id">;
type CreateFlowStep = "select_type" | "configure";

const DEFAULT_SQL_TEMPLATE = `You must produce a valid SQL query that answers the question.
Constraint: return SQL only (no explanation).
User question: {question}
Allowed tables: {allowed_tables}
Available schema:
{schema}`;

const DEFAULT_ANSWER_TEMPLATE = `Question: {question}
Executed SQL: {sql}
Raw result (JSON): {rows}
Produce a concise, business-oriented answer in English.`;

const EMPTY_AGENT: AgentDraft = {
  name: "",
  agent_type: "sql_analyst",
  description: "",
  system_prompt: "You are a reliable SQL expert and data analyst.",
  sql_prompt_template: DEFAULT_SQL_TEMPLATE,
  answer_prompt_template: DEFAULT_ANSWER_TEMPLATE,
  allowed_tables: [],
  max_rows: 200,
  template_config: {},
  enabled: true
};

function cloneDraft(draft: AgentDraft): AgentDraft {
  return JSON.parse(JSON.stringify(draft)) as AgentDraft;
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

function tryParseJsonObject(value: string): Record<string, unknown> | null {
  if (!value.trim()) {
    return {};
  }

  try {
    const parsed = JSON.parse(value) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return null;
    }
    return parsed as Record<string, unknown>;
  } catch {
    return null;
  }
}

function parseCsvList(value: string): string[] {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function parseExtensionList(value: string): string[] {
  return parseCsvList(value).map((item) => {
    if (item === "*") {
      return "*";
    }
    return item.startsWith(".") ? item.toLowerCase() : `.${item.toLowerCase()}`;
  });
}

type SqlParameterType = "string" | "integer" | "number" | "boolean" | "date";

interface SqlUseCaseParameterDraft {
  name: string;
  description: string;
  type: SqlParameterType;
  required: boolean;
  format_hint: string;
  example: string;
  default_value: string;
}

const SQL_PARAMETER_TYPES: SqlParameterType[] = [
  "string",
  "integer",
  "number",
  "boolean",
  "date"
];

function normalizeSqlParameterName(value: string): string {
  const normalized = value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_]+/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "");
  if (!normalized) {
    return "";
  }
  if (/^[0-9]/.test(normalized)) {
    return `p_${normalized}`;
  }
  return normalized;
}

function sqlUseCaseParametersFromConfig(value: unknown): SqlUseCaseParameterDraft[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const rows: SqlUseCaseParameterDraft[] = [];
  for (const item of value) {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      continue;
    }
    const raw = item as Record<string, unknown>;
    const typeCandidate = String(raw.type ?? "string").trim().toLowerCase();
    const type = SQL_PARAMETER_TYPES.includes(typeCandidate as SqlParameterType)
      ? (typeCandidate as SqlParameterType)
      : "string";
    rows.push({
      name: normalizeSqlParameterName(String(raw.name ?? "")),
      description: String(raw.description ?? ""),
      type,
      required: typeof raw.required === "boolean" ? raw.required : true,
      format_hint: String(raw.format_hint ?? ""),
      example: String(raw.example ?? ""),
      default_value: String(raw.default_value ?? "")
    });
  }
  return rows;
}

function sqlUseCaseParametersToConfig(
  parameters: SqlUseCaseParameterDraft[]
): Array<Record<string, unknown>> {
  return parameters
    .map((item) => ({
      name: normalizeSqlParameterName(item.name),
      description: item.description.trim(),
      type: item.type,
      required: item.required,
      format_hint: item.format_hint.trim(),
      example: item.example.trim(),
      default_value: item.default_value.trim()
    }))
    .filter((item) => item.name);
}

function defaultedTemplateDraft(template: AgentTemplate): AgentDraft {
  const draft = cloneDraft(template.defaults);
  const config: Record<string, unknown> = {
    ...(draft.template_config ?? {})
  };
  const fileLikeTypes = new Set([
    "file_assistant",
    "rag_context",
    "text_file_manager",
    "excel_manager",
    "word_manager"
  ]);

  if (fileLikeTypes.has(draft.agent_type)) {
    const folderPath = String(config.folder_path ?? "").trim();
    if (!folderPath) {
      config.folder_path = "~/Downloads";
    }
  }

  if (draft.agent_type === "web_scraper") {
    if (!Array.isArray(config.start_urls)) {
      config.start_urls = [];
    }
    if (!Array.isArray(config.allowed_domains)) {
      config.allowed_domains = [];
    }
  }

  if (draft.agent_type === "web_navigator") {
    if (typeof config.start_url !== "string") {
      config.start_url = "";
    }
    if (!Array.isArray(config.allowed_domains)) {
      config.allowed_domains = [];
    }
  }

  if (draft.agent_type === "elasticsearch_retriever") {
    const index = String(config.index ?? "").trim();
    if (!index) {
      config.index = "documents";
    }
  }

  if (draft.agent_type === "sql_analyst") {
    if (typeof config.sql_use_case_mode !== "string") {
      config.sql_use_case_mode = "llm_sql";
    }
    if (typeof config.sql_query_template !== "string") {
      config.sql_query_template = "";
    }
    if (!Array.isArray(config.sql_parameters)) {
      config.sql_parameters = [];
    }
  }

  if (draft.agent_type === "clickhouse_table_manager") {
    if (typeof config.protect_existing_tables !== "boolean") {
      config.protect_existing_tables = true;
    }
    if (typeof config.allow_row_inserts !== "boolean") {
      config.allow_row_inserts = true;
    }
    if (typeof config.allow_row_updates !== "boolean") {
      config.allow_row_updates = true;
    }
    if (typeof config.allow_row_deletes !== "boolean") {
      config.allow_row_deletes = false;
    }
    if (
      typeof config.max_statements !== "number" ||
      !Number.isFinite(config.max_statements)
    ) {
      config.max_statements = 8;
    }
    if (
      typeof config.preview_select_rows !== "number" ||
      !Number.isFinite(config.preview_select_rows)
    ) {
      config.preview_select_rows = 100;
    }
    if (typeof config.stop_on_error !== "boolean") {
      config.stop_on_error = true;
    }
  }

  if (draft.agent_type === "rss_news") {
    if (!Array.isArray(config.feed_urls) || config.feed_urls.length === 0) {
      config.feed_urls = [
        "https://www.lemonde.fr/rss/une.xml",
        "https://www.franceinfo.fr/titres.rss",
        "https://www.lefigaro.fr/rss/figaro_actualites.xml",
        "https://www.rfi.fr/fr/rss"
      ];
    }
    if (!Array.isArray(config.interests)) {
      config.interests = ["economie", "ia", "technologie", "geopolitique"];
    }
    if (!Array.isArray(config.exclude_keywords)) {
      config.exclude_keywords = [];
    }
    if (typeof config.language_hint !== "string" || !String(config.language_hint).trim()) {
      config.language_hint = "fr";
    }
  }

  return {
    ...draft,
    template_config: config
  };
}

function labelWithInfo(text: string, info: string) {
  return (
    <span className="label-title">
      <span>{text}</span>
      <span className="info-bubble" title={info} aria-label={info}>
        i
      </span>
    </span>
  );
}

function normalizeImportPayload(
  raw: unknown
): AgentConfigExportPayload | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return null;
  }

  const candidate = raw as Record<string, unknown>;
  if (Array.isArray(candidate.agents)) {
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
    const auditHistory =
      candidate.audit_history &&
      typeof candidate.audit_history === "object" &&
      !Array.isArray(candidate.audit_history)
        ? (candidate.audit_history as Record<string, AgentAuditEntry[]>)
        : {};

    return {
      export_version: exportVersion === "1.0" ? "1.0" : "1.0",
      exported_at: exportedAt,
      source,
      agents: candidate.agents as AgentConfig[],
      audit_history: auditHistory
    };
  }
  return null;
}

export function AgentsTab({ agents, managerConfig, onRefresh, onNotify }: AgentsTabProps) {
  const [templates, setTemplates] = useState<AgentTemplate[]>([]);
  const [selectedTemplateId, setSelectedTemplateId] = useState<string>("sql_analyst");
  const [selectedAgentId, setSelectedAgentId] = useState<string>(agents[0]?.id ?? "new");
  const [createFlowStep, setCreateFlowStep] = useState<CreateFlowStep>("configure");
  const [agentDraft, setAgentDraft] = useState<AgentDraft>(cloneDraft(EMPTY_AGENT));
  const [managerDraft, setManagerDraft] = useState<ManagerConfig>(managerConfig);
  const [allowedTablesText, setAllowedTablesText] = useState<string>("");
  const [templateConfigText, setTemplateConfigText] = useState<string>("{}");
  const [showAdvancedTemplateConfig, setShowAdvancedTemplateConfig] = useState(false);
  const [showAdvancedPrompts, setShowAdvancedPrompts] = useState(false);
  const [agentAuditHistory, setAgentAuditHistory] = useState<AgentAuditEntry[]>([]);
  const [isAuditLoading, setIsAuditLoading] = useState(false);
  const [isTransferringConfig, setIsTransferringConfig] = useState(false);
  const importFileInputRef = useRef<HTMLInputElement | null>(null);
  const isCreating = selectedAgentId === "new";
  const isSelectingType = isCreating && createFlowStep === "select_type";

  const selectedAgent = useMemo(
    () => agents.find((item) => item.id === selectedAgentId) ?? null,
    [agents, selectedAgentId]
  );

  const templateById = useMemo(
    () =>
      new Map<string, AgentTemplate>(
        templates.map((template): [string, AgentTemplate] => [template.id, template])
      ),
    [templates]
  );
  const parsedTemplateConfig = useMemo(
    () => tryParseJsonObject(templateConfigText),
    [templateConfigText]
  );
  const templateConfigMap: Record<string, unknown> = parsedTemplateConfig ?? {};
  const getConfigString = (key: string, fallback = ""): string => {
    const value = templateConfigMap[key];
    return typeof value === "string" ? value : fallback;
  };
  const getConfigNumber = (
    key: string,
    fallback: number,
    min: number,
    max: number
  ): number => {
    const value = Number(templateConfigMap[key]);
    if (!Number.isFinite(value)) {
      return fallback;
    }
    return Math.max(min, Math.min(max, value));
  };
  const getConfigBoolean = (key: string, fallback: boolean): boolean => {
    const value = templateConfigMap[key];
    if (typeof value === "boolean") {
      return value;
    }
    if (typeof value === "string") {
      const normalized = value.trim().toLowerCase();
      if (["1", "true", "yes", "on"].includes(normalized)) {
        return true;
      }
      if (["0", "false", "no", "off"].includes(normalized)) {
        return false;
      }
    }
    return fallback;
  };
  const getConfigStringList = (key: string): string[] => {
    const value = templateConfigMap[key];
    if (Array.isArray(value)) {
      return value.map((item) => String(item).trim()).filter(Boolean);
    }
    if (typeof value === "string") {
      return parseCsvList(value);
    }
    return [];
  };

  const agentType = agentDraft.agent_type;
  const hasAssistedTemplateFields = true;

  const folderPathValue = getConfigString("folder_path", "");
  const databaseIdValue = getConfigString("database_id", "");
  const databaseNameValue = getConfigString("database_name", "");
  const sqlUseCaseModeValue =
    getConfigString("sql_use_case_mode", "llm_sql").trim().toLowerCase() ===
    "parameterized_template"
      ? "parameterized_template"
      : "llm_sql";
  const sqlQueryTemplateValue = getConfigString("sql_query_template", "");
  const sqlUseCaseParameters = sqlUseCaseParametersFromConfig(
    templateConfigMap.sql_parameters
  );
  const fileExtensionsValue = getConfigStringList("file_extensions").join(", ");
  const allowedDomainsValue = getConfigStringList("allowed_domains").join(", ");
  const fieldsValue = getConfigStringList("fields").join(", ");
  const includeSectionsValue = getConfigStringList("include_sections").join(", ");
  const startUrlsValue = getConfigStringList("start_urls").join(", ");
  const feedUrlsValue = getConfigStringList("feed_urls").join(", ");
  const interestsValue = getConfigStringList("interests").join(", ");
  const excludeKeywordsValue = getConfigStringList("exclude_keywords").join(", ");
  const outputSchemaValue =
    typeof templateConfigMap.output_schema === "object" && templateConfigMap.output_schema
      ? JSON.stringify(templateConfigMap.output_schema, null, 2)
      : JSON.stringify(
          {
            summary: "string",
            entities: [{ type: "string", value: "string" }],
            priority: "low|medium|high"
          },
          null,
          2
        );

  const strictJsonValue = getConfigBoolean("strict_json", true);
  const maxBulletsValue = getConfigNumber("max_bullets", 8, 1, 30);
  const maxFilesValue = getConfigNumber("max_files", 40, 1, 5000);
  const maxFileSizeKbValue = getConfigNumber("max_file_size_kb", 400, 50, 200000);
  const topKValue = getConfigNumber("top_k", 5, 1, 50);
  const topKChunksValue = getConfigNumber("top_k_chunks", 6, 1, 50);
  const chunkSizeValue = getConfigNumber("chunk_size", 1200, 200, 30000);
  const chunkOverlapValue = getConfigNumber("chunk_overlap", 150, 0, 10000);
  const maxCharsReadValue = getConfigNumber("max_chars_read", 12000, 200, 500000);
  const maxRowsReadValue = getConfigNumber("max_rows_read", 200, 1, 10000);
  const maxParagraphsReadValue = getConfigNumber("max_paragraphs_read", 80, 1, 5000);
  const verifySslValue = getConfigBoolean("verify_ssl", true);
  const includeUrlsFromQuestionValue = getConfigBoolean("include_urls_from_question", true);
  const searchFallbackValue = getConfigBoolean("search_fallback", true);
  const followLinksValue = getConfigBoolean("follow_links", false);
  const sameDomainOnlyValue = getConfigBoolean("same_domain_only", true);
  const maxPagesValue = getConfigNumber("max_pages", 3, 1, 20);
  const maxLinksPerPageValue = getConfigNumber("max_links_per_page", 10, 1, 30);
  const maxCharsPerPageValue = getConfigNumber("max_chars_per_page", 6000, 500, 30000);
  const timeoutSecondsValue = getConfigNumber("timeout_seconds", 20, 3, 300);
  const timeoutMsValue = getConfigNumber("timeout_ms", 15000, 2000, 120000);
  const captureHtmlCharsValue = getConfigNumber("capture_html_chars", 7000, 500, 30000);
  const maxItemsPerFeedValue = getConfigNumber("max_items_per_feed", 25, 1, 200);
  const hoursLookbackValue = getConfigNumber("hours_lookback", 24, 1, 24 * 14);
  const navigatorMaxStepsValue = getConfigNumber("max_steps", 8, 1, 30);
  const summarySentencesValue = getConfigNumber("summary_sentences", 2, 1, 5);
  const headlessValue = getConfigBoolean("headless", true);
  const includeGeneralIfNoMatchValue = getConfigBoolean(
    "include_general_if_no_match",
    true
  );
  const autoCreateFolderValue = getConfigBoolean("auto_create_folder", true);
  const allowOverwriteValue = getConfigBoolean("allow_overwrite", true);
  const autoCreateWorkbookValue = getConfigBoolean("auto_create_workbook", true);
  const autoCreateDocumentValue = getConfigBoolean("auto_create_document", true);
  const languageHintValue = getConfigString("language_hint", "fr");
  const protectExistingTablesValue = getConfigBoolean("protect_existing_tables", true);
  const allowRowInsertsValue = getConfigBoolean("allow_row_inserts", true);
  const allowRowUpdatesValue = getConfigBoolean("allow_row_updates", true);
  const allowRowDeletesValue = getConfigBoolean("allow_row_deletes", false);
  const maxStatementsValue = getConfigNumber("max_statements", 8, 1, 40);
  const previewSelectRowsValue = getConfigNumber("preview_select_rows", 100, 1, 500);
  const stopOnErrorValue = getConfigBoolean("stop_on_error", true);

  useEffect(() => {
    void (async () => {
      try {
        const response = await fetchAgentTemplates();
        setTemplates(response);
        if (response.length > 0) {
          setSelectedTemplateId((previous) =>
            response.some((item) => item.id === previous) ? previous : response[0].id
          );
        }
      } catch (error) {
        onNotify(
          error instanceof Error
            ? error.message
            : "Unable to load agent templates.",
          "error"
        );
      }
    })();
  }, []);

  useEffect(() => {
    if (selectedAgentId === "new") {
      return;
    }
    if (!agents.some((item) => item.id === selectedAgentId)) {
      setSelectedAgentId(agents[0]?.id ?? "new");
      setCreateFlowStep("configure");
    }
  }, [agents, selectedAgentId]);

  useEffect(() => {
    if (!selectedAgent) {
      setAgentAuditHistory([]);
      return;
    }

    setAgentDraft({
      name: selectedAgent.name,
      agent_type: selectedAgent.agent_type,
      description: selectedAgent.description,
      system_prompt: selectedAgent.system_prompt,
      sql_prompt_template: selectedAgent.sql_prompt_template,
      answer_prompt_template: selectedAgent.answer_prompt_template,
      allowed_tables: selectedAgent.allowed_tables,
      max_rows: selectedAgent.max_rows,
      template_config: selectedAgent.template_config,
      enabled: selectedAgent.enabled
    });
    setAllowedTablesText(selectedAgent.allowed_tables.join(", "));
    setTemplateConfigText(JSON.stringify(selectedAgent.template_config ?? {}, null, 2));
    setSelectedTemplateId(selectedAgent.agent_type);
    setShowAdvancedTemplateConfig(false);
    setShowAdvancedPrompts(false);
    setCreateFlowStep("configure");
  }, [selectedAgent]);

  useEffect(() => {
    if (!selectedAgent) {
      setAgentAuditHistory([]);
      return;
    }

    void (async () => {
      try {
        setIsAuditLoading(true);
        const history = await fetchAgentAudit(selectedAgent.id);
        setAgentAuditHistory(history);
      } catch (error) {
        onNotify(
          error instanceof Error ? error.message : "Unable to load agent audit history.",
          "error"
        );
        setAgentAuditHistory([]);
      } finally {
        setIsAuditLoading(false);
      }
    })();
  }, [selectedAgent]);

  useEffect(() => {
    setManagerDraft(managerConfig);
  }, [managerConfig]);

  const applyTemplate = (
    templateId: string,
    options?: { preserveName?: boolean; preserveDescription?: boolean }
  ) => {
    const template = templateById.get(templateId);
    if (!template) {
      onNotify("Selected template is unavailable.", "error");
      return;
    }

    const defaults = defaultedTemplateDraft(template);
    setAgentDraft((previous) => ({
      ...defaults,
      name: options?.preserveName && previous.name.trim() ? previous.name : defaults.name,
      description:
        options?.preserveDescription && previous.description.trim()
          ? previous.description
          : defaults.description
    }));
    setAllowedTablesText((defaults.allowed_tables ?? []).join(", "));
    setTemplateConfigText(JSON.stringify(defaults.template_config ?? {}, null, 2));
    setShowAdvancedTemplateConfig(false);
    setShowAdvancedPrompts(false);
  };

  const chooseTemplateForCreate = (templateId: string) => {
    setSelectedTemplateId(templateId);
    applyTemplate(templateId);
    setCreateFlowStep("configure");
    onNotify("Agent type selected. Complete the simple fields and save.");
  };

  const startCreate = () => {
    setSelectedAgentId("new");
    setShowAdvancedTemplateConfig(false);
    setShowAdvancedPrompts(false);
    if (templates.length > 0) {
      const preferred = templateById.has(selectedTemplateId)
        ? selectedTemplateId
        : templates[0].id;
      setSelectedTemplateId(preferred);
      setCreateFlowStep("select_type");
      return;
    }

    setAgentDraft(cloneDraft(EMPTY_AGENT));
    setAllowedTablesText("");
    setTemplateConfigText("{}");
    setCreateFlowStep("configure");
  };

  const updateTemplateConfigField = (field: string, value: unknown) => {
    const base = tryParseJsonObject(templateConfigText) ?? {};
    setTemplateConfigText(
      JSON.stringify(
        {
          ...base,
          [field]: value
        },
        null,
        2
      )
    );
  };

  const updateSqlUseCaseParameters = (next: SqlUseCaseParameterDraft[]) => {
    updateTemplateConfigField("sql_parameters", sqlUseCaseParametersToConfig(next));
  };

  const updateSqlUseCaseParameter = (
    index: number,
    field: keyof SqlUseCaseParameterDraft,
    value: string | boolean
  ) => {
    const next = sqlUseCaseParameters.map((item, itemIndex) => {
      if (itemIndex !== index) {
        return item;
      }
      if (field === "required" && typeof value === "boolean") {
        return { ...item, required: value };
      }
      if (field === "type" && typeof value === "string") {
        const normalizedType = SQL_PARAMETER_TYPES.includes(value as SqlParameterType)
          ? (value as SqlParameterType)
          : "string";
        return { ...item, type: normalizedType };
      }
      if (typeof value === "string") {
        if (field === "name") {
          return { ...item, name: normalizeSqlParameterName(value) };
        }
        return { ...item, [field]: value };
      }
      return item;
    });
    updateSqlUseCaseParameters(next);
  };

  const addSqlUseCaseParameter = () => {
    updateSqlUseCaseParameters([
      ...sqlUseCaseParameters,
      {
        name: "",
        description: "",
        type: "string",
        required: true,
        format_hint: "",
        example: "",
        default_value: ""
      }
    ]);
  };

  const removeSqlUseCaseParameter = (index: number) => {
    updateSqlUseCaseParameters(
      sqlUseCaseParameters.filter((_, itemIndex) => itemIndex !== index)
    );
  };

  const save = async () => {
    try {
      if (!agentDraft.name.trim()) {
        onNotify("Agent name is required.", "error");
        return;
      }

      const payload: AgentDraft = {
        ...agentDraft,
        name: agentDraft.name.trim(),
        description: agentDraft.description.trim(),
        allowed_tables: allowedTablesText
          .split(",")
          .map((table) => table.trim())
          .filter(Boolean),
        template_config: parseJsonObject(templateConfigText, "Template config")
      };

      if (selectedAgentId === "new") {
        const created = await createAgent(payload);
        setSelectedAgentId(created.id);
        setCreateFlowStep("configure");
        onNotify("Agent created.");
      } else {
        await updateAgent(selectedAgentId, payload);
        onNotify("Agent updated.");
      }

      await onRefresh();
    } catch (error) {
      onNotify(error instanceof Error ? error.message : "Agent update failed.", "error");
    }
  };

  const remove = async () => {
    if (!selectedAgent) {
      return;
    }

    try {
      await deleteAgent(selectedAgent.id);
      setSelectedAgentId("new");
      await onRefresh();
      onNotify("Agent deleted.");
    } catch (error) {
      onNotify(error instanceof Error ? error.message : "Delete failed.", "error");
    }
  };

  const saveManager = async () => {
    try {
      const payload: ManagerConfig = {
        max_steps: Math.max(1, Math.min(30, Number(managerDraft.max_steps) || 1)),
        max_agent_calls: Math.max(
          1,
          Math.min(100, Number(managerDraft.max_agent_calls) || 1)
        )
      };
      await saveManagerConfig(payload);
      await onRefresh();
      onNotify("Manager agent settings updated.");
    } catch (error) {
      onNotify(error instanceof Error ? error.message : "Manager update failed.", "error");
    }
  };

  const exportConfig = async () => {
    try {
      setIsTransferringConfig(true);
      const payload = await exportAgentsConfig();
      const filenameSafeDate = new Date()
        .toISOString()
        .replace(/:/g, "-")
        .replace(/\./g, "-");
      const blob = new Blob([JSON.stringify(payload, null, 2)], {
        type: "application/json"
      });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `agents-config-${filenameSafeDate}.json`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      onNotify("Agent configuration exported.");
    } catch (error) {
      onNotify(
        error instanceof Error ? error.message : "Agent export failed.",
        "error"
      );
    } finally {
      setIsTransferringConfig(false);
    }
  };

  const openImportDialog = () => {
    importFileInputRef.current?.click();
  };

  const importConfig = async (event: ChangeEvent<HTMLInputElement>) => {
    const input = event.currentTarget;
    const file = input.files?.[0];
    if (!file) {
      return;
    }

    try {
      setIsTransferringConfig(true);
      const content = await file.text();
      const parsed = JSON.parse(content) as unknown;
      const payload = normalizeImportPayload(parsed);
      if (!payload) {
        throw new Error(
          "Invalid import file format. Expected an agents export JSON payload."
        );
      }

      const confirmed = window.confirm(
        "This will replace current agents and their audit history. Continue?"
      );
      if (!confirmed) {
        return;
      }

      const result = await importAgentsConfig(payload, "replace");
      await onRefresh();
      if (payload.agents.length > 0) {
        setSelectedAgentId(payload.agents[0].id);
      } else {
        setSelectedAgentId("new");
      }
      setCreateFlowStep("configure");
      onNotify(
        `Agent configuration imported. Agents: ${String(result.imported_agents)}.`
      );
    } catch (error) {
      onNotify(
        error instanceof Error ? error.message : "Agent import failed.",
        "error"
      );
    } finally {
      input.value = "";
      setIsTransferringConfig(false);
    }
  };

  const restoreVersion = async (versionId: string) => {
    if (!selectedAgent) {
      return;
    }

    try {
      const restored = await restoreAgentAuditVersion(selectedAgent.id, versionId);
      await onRefresh();
      setSelectedAgentId(restored.id);
      onNotify("Agent configuration restored.");
    } catch (error) {
      onNotify(error instanceof Error ? error.message : "Restore failed.", "error");
    }
  };

  return (
    <div className="tab-layout">
      <section className="card">
        <header className="card-header">
          <h2>Agent Configuration</h2>
          <div className="button-row">
            <input
              ref={importFileInputRef}
              type="file"
              accept="application/json,.json"
              className="visually-hidden"
              onChange={importConfig}
            />
            <button
              className="btn-ghost"
              onClick={exportConfig}
              disabled={isTransferringConfig}
            >
              Export agents
            </button>
            <button
              className="btn-ghost"
              onClick={openImportDialog}
              disabled={isTransferringConfig}
            >
              Import agents
            </button>
            <button className="btn-primary" onClick={startCreate}>
              Create a new agent
            </button>
          </div>
        </header>
        <p className="hint">
          Export/import lets you back up and restore all agent configurations quickly.
        </p>

        <section className="card">
          <div className="card-header">
            <h3>Manager Agent</h3>
            <button className="btn-primary" onClick={saveManager}>
              Save manager
            </button>
          </div>
          <p className="hint">
            Configure orchestration parameters used by the manager in Playground mode.
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
                  setManagerDraft((previous) => ({
                    ...previous,
                    max_steps: Math.max(1, Math.min(30, Number(event.target.value) || 1))
                  }))
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
                  setManagerDraft((previous) => ({
                    ...previous,
                    max_agent_calls: Math.max(
                      1,
                      Math.min(100, Number(event.target.value) || 1)
                    )
                  }))
                }
              />
            </label>
          </div>
        </section>

        <div className="agent-editor-layout">
          <aside className="agent-list-pane">
            <h3>Agents</h3>
            {agents.length === 0 ? (
              <p className="hint">No agents yet. Create one to get started.</p>
            ) : (
              <div className="agent-list">
                {agents.map((agent) => (
                  <button
                    key={agent.id}
                    className={
                      selectedAgentId === agent.id
                        ? "agent-list-item active"
                        : "agent-list-item"
                    }
                    onClick={() => {
                      setSelectedAgentId(agent.id);
                      setCreateFlowStep("configure");
                    }}
                  >
                    <span>{agent.name}</span>
                    <span className="agent-list-meta">
                      {agent.agent_type} • {agent.enabled ? "Enabled" : "Disabled"}
                    </span>
                  </button>
                ))}
              </div>
            )}
          </aside>

          <section className="agent-form-pane">
            <div className="card-header">
              <h3>
                {selectedAgent
                  ? `Edit: ${selectedAgent.name}`
                  : isSelectingType
                    ? "New agent: choose type"
                    : "New agent: configure"}
              </h3>
              <div className="button-row">
                {isCreating && !isSelectingType && (
                  <button
                    className="btn-ghost"
                    onClick={() => setCreateFlowStep("select_type")}
                  >
                    Change type
                  </button>
                )}
                {!isSelectingType && (
                  <button className="btn-primary" onClick={save}>
                    Save
                  </button>
                )}
                <button className="btn-danger" onClick={remove} disabled={!selectedAgent}>
                  Delete
                </button>
              </div>
            </div>
            {isSelectingType ? (
              <section className="template-picker-panel">
                <p className="hint">
                  Step 1: choose the agent type. Step 2: complete only the simple inputs.
                  Prompts and advanced JSON stay prefilled by default.
                </p>
                <div className="template-picker-grid">
                  {templates.map((template) => (
                    <button
                      key={template.id}
                      className={
                        selectedTemplateId === template.id
                          ? "template-picker-item active"
                          : "template-picker-item"
                      }
                      onClick={() => chooseTemplateForCreate(template.id)}
                    >
                      <div className="template-picker-head">
                        <strong>{template.name}</strong>
                        <span className="agent-list-meta">{template.id}</span>
                      </div>
                      <p>{template.description}</p>
                    </button>
                  ))}
                </div>
              </section>
            ) : (
              <div className="grid two-columns">
              <label>
                {labelWithInfo(
                  "Template",
                  "Agent template defines default prompts and settings. Keep defaults for faster setup."
                )}
                <select
                  value={selectedTemplateId}
                  onChange={(event) => setSelectedTemplateId(event.target.value)}
                >
                  {templates.map((template) => (
                    <option key={template.id} value={template.id}>
                      {template.name}
                    </option>
                  ))}
                </select>
              </label>

              <label>
                {labelWithInfo("Agent type", "Technical type used by the backend executor.")}
                <input value={agentDraft.agent_type} readOnly />
              </label>

              <div className="full-width toolbar">
                <button
                  className="btn-ghost"
                  onClick={() => {
                    applyTemplate(selectedTemplateId, {
                      preserveName: true,
                      preserveDescription: true
                    });
                    onNotify("Template applied to current draft.");
                  }}
                >
                  Apply selected template
                </button>
                {selectedTemplateId && templateById.get(selectedTemplateId) && (
                  <span className="hint">
                    {templateById.get(selectedTemplateId)?.description}
                  </span>
                )}
              </div>

              <label>
                {labelWithInfo(
                  "Name",
                  "Displayed in Agent list and used by manager in orchestration traces."
                )}
                <input
                  value={agentDraft.name}
                  onChange={(event) =>
                    setAgentDraft((prev) => ({ ...prev, name: event.target.value }))
                  }
                />
              </label>

              <label>
                {labelWithInfo(
                  "Max rows",
                  "Hard limit for retrieved rows/chunks to keep responses fast and concise."
                )}
                <input
                  type="number"
                  min={1}
                  max={5000}
                  value={agentDraft.max_rows}
                  onChange={(event) =>
                    setAgentDraft((prev) => ({
                      ...prev,
                      max_rows: Number(event.target.value)
                    }))
                  }
                />
              </label>

              <label className="full-width">
                {labelWithInfo(
                  "Description",
                  "Explain this agent's responsibility. Used by manager for routing decisions."
                )}
                <input
                  value={agentDraft.description}
                  onChange={(event) =>
                    setAgentDraft((prev) => ({ ...prev, description: event.target.value }))
                  }
                />
              </label>

              {agentType === "sql_analyst" && (
                <>
                  <p className="field-help full-width">
                    <span
                      className="info-bubble"
                      title="Configure classic LLM SQL mode or a parameterized SQL use-case template."
                      aria-label="SQL agent help"
                    >
                      i
                    </span>
                    Use LLM SQL mode for free-form analytics, or parameterized mode for fixed
                    business use cases with required inputs.
                  </p>
                  <label className="full-width">
                    Allowed tables (comma-separated)
                    <input
                      value={allowedTablesText}
                      onChange={(event) => setAllowedTablesText(event.target.value)}
                      placeholder="orders, customers"
                    />
                  </label>
                  <label>
                    Preferred database ID (optional)
                    <input
                      value={databaseIdValue}
                      onChange={(event) =>
                        updateTemplateConfigField("database_id", event.target.value.trim())
                      }
                      placeholder="db-profile-id"
                    />
                  </label>
                  <label>
                    Preferred database name (optional)
                    <input
                      value={databaseNameValue}
                      onChange={(event) =>
                        updateTemplateConfigField("database_name", event.target.value.trim())
                      }
                      placeholder="Analytics CH"
                    />
                  </label>
                  <label className="full-width">
                    SQL execution mode
                    <select
                      value={sqlUseCaseModeValue}
                      onChange={(event) =>
                        updateTemplateConfigField("sql_use_case_mode", event.target.value)
                      }
                    >
                      <option value="llm_sql">LLM-generated SQL (default)</option>
                      <option value="parameterized_template">
                        Parameterized SQL use case
                      </option>
                    </select>
                  </label>

                  {sqlUseCaseModeValue === "parameterized_template" && (
                    <>
                      <label className="full-width">
                        SQL query template
                        <textarea
                          rows={5}
                          value={sqlQueryTemplateValue}
                          onChange={(event) =>
                            updateTemplateConfigField(
                              "sql_query_template",
                              event.target.value
                            )
                          }
                          placeholder={"SELECT * FROM tableA WHERE client = {{client}}"}
                        />
                      </label>
                      <p className="hint full-width">
                        Use placeholders like <code>{`{{client}}`}</code>. The manager/LLM will
                        extract each parameter from the user request and inject SQL-safe literals.
                      </p>

                      <section className="card full-width">
                        <div className="card-header">
                          <h4>Use-case parameters</h4>
                          <button
                            type="button"
                            className="btn-ghost"
                            onClick={addSqlUseCaseParameter}
                          >
                            Add parameter
                          </button>
                        </div>
                        {sqlUseCaseParameters.length === 0 ? (
                          <p className="hint">
                            Add at least one required parameter (example: <code>client</code>).
                          </p>
                        ) : (
                          <div className="grid">
                            {sqlUseCaseParameters.map((parameter, index) => (
                              <article className="card" key={`sql-param-${index}`}>
                                <div className="card-header">
                                  <h4>Parameter {index + 1}</h4>
                                  <button
                                    type="button"
                                    className="btn-danger"
                                    onClick={() => removeSqlUseCaseParameter(index)}
                                  >
                                    Remove
                                  </button>
                                </div>
                                <div className="grid two-columns">
                                  <label>
                                    Name
                                    <input
                                      value={parameter.name}
                                      onChange={(event) =>
                                        updateSqlUseCaseParameter(
                                          index,
                                          "name",
                                          event.target.value
                                        )
                                      }
                                      placeholder="client"
                                    />
                                  </label>
                                  <label>
                                    Type
                                    <select
                                      value={parameter.type}
                                      onChange={(event) =>
                                        updateSqlUseCaseParameter(
                                          index,
                                          "type",
                                          event.target.value
                                        )
                                      }
                                    >
                                      {SQL_PARAMETER_TYPES.map((item) => (
                                        <option key={item} value={item}>
                                          {item}
                                        </option>
                                      ))}
                                    </select>
                                  </label>
                                  <label className="checkbox-row">
                                    <input
                                      type="checkbox"
                                      checked={parameter.required}
                                      onChange={(event) =>
                                        updateSqlUseCaseParameter(
                                          index,
                                          "required",
                                          event.target.checked
                                        )
                                      }
                                    />
                                    Required
                                  </label>
                                  <label>
                                    Default value (optional)
                                    <input
                                      value={parameter.default_value}
                                      onChange={(event) =>
                                        updateSqlUseCaseParameter(
                                          index,
                                          "default_value",
                                          event.target.value
                                        )
                                      }
                                      placeholder="ACME"
                                    />
                                  </label>
                                  <label className="full-width">
                                    Description
                                    <input
                                      value={parameter.description}
                                      onChange={(event) =>
                                        updateSqlUseCaseParameter(
                                          index,
                                          "description",
                                          event.target.value
                                        )
                                      }
                                      placeholder="Client name to filter column client"
                                    />
                                  </label>
                                  <label className="full-width">
                                    Format hint (optional)
                                    <input
                                      value={parameter.format_hint}
                                      onChange={(event) =>
                                        updateSqlUseCaseParameter(
                                          index,
                                          "format_hint",
                                          event.target.value
                                        )
                                      }
                                      placeholder="Exact value from CRM, case-sensitive"
                                    />
                                  </label>
                                  <label className="full-width">
                                    Example (optional)
                                    <input
                                      value={parameter.example}
                                      onChange={(event) =>
                                        updateSqlUseCaseParameter(
                                          index,
                                          "example",
                                          event.target.value
                                        )
                                      }
                                      placeholder="ACME_CORP"
                                    />
                                  </label>
                                </div>
                              </article>
                            ))}
                          </div>
                        )}
                      </section>
                    </>
                  )}
                </>
              )}

              {agentType === "clickhouse_table_manager" && (
                <>
                  <p className="field-help full-width">
                    <span
                      className="info-bubble"
                      title="This agent creates tables and writes data in ClickHouse. Keep safety enabled by default."
                      aria-label="ClickHouse table manager help"
                    >
                      i
                    </span>
                    Use this agent for CREATE TABLE and DML workflows. By default, destructive
                    schema operations are blocked for existing tables.
                  </p>
                  <label>
                    Preferred database ID (optional)
                    <input
                      value={databaseIdValue}
                      onChange={(event) =>
                        updateTemplateConfigField("database_id", event.target.value.trim())
                      }
                      placeholder="clickhouse-profile-id"
                    />
                  </label>
                  <label>
                    Preferred database name (optional)
                    <input
                      value={databaseNameValue}
                      onChange={(event) =>
                        updateTemplateConfigField("database_name", event.target.value.trim())
                      }
                      placeholder="ClickHouse Prod"
                    />
                  </label>
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={protectExistingTablesValue}
                      onChange={(event) =>
                        updateTemplateConfigField(
                          "protect_existing_tables",
                          event.target.checked
                        )
                      }
                    />
                    Protect existing tables (block DROP/TRUNCATE/schema ALTER)
                  </label>
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={allowRowInsertsValue}
                      onChange={(event) =>
                        updateTemplateConfigField("allow_row_inserts", event.target.checked)
                      }
                    />
                    Allow INSERT
                  </label>
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={allowRowUpdatesValue}
                      onChange={(event) =>
                        updateTemplateConfigField("allow_row_updates", event.target.checked)
                      }
                    />
                    Allow row updates
                  </label>
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={allowRowDeletesValue}
                      onChange={(event) =>
                        updateTemplateConfigField("allow_row_deletes", event.target.checked)
                      }
                    />
                    Allow row deletes
                  </label>
                  <label>
                    Max statements per run
                    <input
                      type="number"
                      min={1}
                      max={40}
                      value={maxStatementsValue}
                      onChange={(event) =>
                        updateTemplateConfigField(
                          "max_statements",
                          Math.max(1, Math.min(40, Number(event.target.value) || 1))
                        )
                      }
                    />
                  </label>
                  <label>
                    SELECT preview rows
                    <input
                      type="number"
                      min={1}
                      max={500}
                      value={previewSelectRowsValue}
                      onChange={(event) =>
                        updateTemplateConfigField(
                          "preview_select_rows",
                          Math.max(1, Math.min(500, Number(event.target.value) || 1))
                        )
                      }
                    />
                  </label>
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={stopOnErrorValue}
                      onChange={(event) =>
                        updateTemplateConfigField("stop_on_error", event.target.checked)
                      }
                    />
                    Stop on first SQL error
                  </label>
                  {!protectExistingTablesValue && (
                    <p className="hint full-width">
                      Safety is disabled: this agent may execute destructive schema operations
                      (including DROP/TRUNCATE/ALTER). Use with caution.
                    </p>
                  )}
                </>
              )}

              {agentType === "unstructured_to_structured" && (
                <>
                  <p className="field-help full-width">
                    <span
                      className="info-bubble"
                      title="Define the output JSON structure your downstream tools expect."
                      aria-label="Extractor help"
                    >
                      i
                    </span>
                    Keep strict JSON enabled for reliable parsing. Adapt the schema to your
                    target fields.
                  </p>
                  <label className="checkbox-row full-width">
                    <input
                      type="checkbox"
                      checked={strictJsonValue}
                      onChange={(event) =>
                        updateTemplateConfigField("strict_json", event.target.checked)
                      }
                    />
                    Strict JSON output
                  </label>
                  <label className="full-width">
                    Output schema (JSON)
                    <textarea
                      rows={6}
                      value={outputSchemaValue}
                      onChange={(event) => {
                        try {
                          const parsed = JSON.parse(event.target.value) as unknown;
                          if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
                            updateTemplateConfigField("output_schema", parsed);
                          }
                        } catch {
                          // Keep previous valid schema while user edits invalid JSON.
                        }
                      }}
                    />
                  </label>
                </>
              )}

              {agentType === "email_cleaner" && (
                <>
                  <p className="field-help full-width">
                    <span
                      className="info-bubble"
                      title="Control how compact the cleaned email output should be."
                      aria-label="Email cleaner help"
                    >
                      i
                    </span>
                    Choose sections and max bullets to match your team communication style.
                  </p>
                  <label>
                    Maximum bullets per section
                    <input
                      type="number"
                      min={1}
                      max={30}
                      value={maxBulletsValue}
                      onChange={(event) =>
                        updateTemplateConfigField(
                          "max_bullets",
                          Math.max(1, Math.min(30, Number(event.target.value) || 1))
                        )
                      }
                    />
                  </label>
                  <label className="full-width">
                    Included sections (comma-separated)
                    <input
                      value={includeSectionsValue}
                      onChange={(event) =>
                        updateTemplateConfigField("include_sections", parseCsvList(event.target.value))
                      }
                      placeholder="summary, action_items, deadlines, risks"
                    />
                  </label>
                </>
              )}

              {(agentType === "file_assistant" || agentType === "rag_context") && (
                <>
                  <p className="field-help full-width">
                    <span
                      className="info-bubble"
                      title="Point to a local folder with source files the agent can read."
                      aria-label="File and RAG help"
                    >
                      i
                    </span>
                    Folder path is the key input. Then tune extensions and limits for better
                    retrieval quality.
                  </p>
                  <label className="full-width">
                    Folder path
                    <input
                      value={folderPathValue}
                      onChange={(event) =>
                        updateTemplateConfigField("folder_path", event.target.value)
                      }
                      placeholder="/Users/mathieumasson/Documents/AgentFiles"
                    />
                  </label>
                  <label className="full-width">
                    File extensions (comma-separated)
                    <input
                      value={fileExtensionsValue}
                      onChange={(event) =>
                        updateTemplateConfigField(
                          "file_extensions",
                          parseExtensionList(event.target.value)
                        )
                      }
                      placeholder=".txt, .md, .json"
                    />
                  </label>
                  <label>
                    Maximum files
                    <input
                      type="number"
                      min={1}
                      max={5000}
                      value={maxFilesValue}
                      onChange={(event) =>
                        updateTemplateConfigField(
                          "max_files",
                          Math.max(1, Math.min(5000, Number(event.target.value) || 1))
                        )
                      }
                    />
                  </label>
                  {agentType === "file_assistant" && (
                    <>
                      <label>
                        Max file size (KB)
                        <input
                          type="number"
                          min={50}
                          max={200000}
                          value={maxFileSizeKbValue}
                          onChange={(event) =>
                            updateTemplateConfigField(
                              "max_file_size_kb",
                              Math.max(50, Math.min(200000, Number(event.target.value) || 50))
                            )
                          }
                        />
                      </label>
                      <label>
                        Top K files
                        <input
                          type="number"
                          min={1}
                          max={50}
                          value={topKValue}
                          onChange={(event) =>
                            updateTemplateConfigField(
                              "top_k",
                              Math.max(1, Math.min(50, Number(event.target.value) || 1))
                            )
                          }
                        />
                      </label>
                    </>
                  )}
                  {agentType === "rag_context" && (
                    <>
                      <label>
                        Top K chunks
                        <input
                          type="number"
                          min={1}
                          max={50}
                          value={topKChunksValue}
                          onChange={(event) =>
                            updateTemplateConfigField(
                              "top_k_chunks",
                              Math.max(1, Math.min(50, Number(event.target.value) || 1))
                            )
                          }
                        />
                      </label>
                      <label>
                        Chunk size
                        <input
                          type="number"
                          min={200}
                          max={30000}
                          value={chunkSizeValue}
                          onChange={(event) =>
                            updateTemplateConfigField(
                              "chunk_size",
                              Math.max(200, Math.min(30000, Number(event.target.value) || 200))
                            )
                          }
                        />
                      </label>
                      <label>
                        Chunk overlap
                        <input
                          type="number"
                          min={0}
                          max={10000}
                          value={chunkOverlapValue}
                          onChange={(event) =>
                            updateTemplateConfigField(
                              "chunk_overlap",
                              Math.max(0, Math.min(10000, Number(event.target.value) || 0))
                            )
                          }
                        />
                      </label>
                    </>
                  )}
                </>
              )}

              {agentType === "text_file_manager" && (
                <>
                  <p className="field-help full-width">
                    <span
                      className="info-bubble"
                      title="Configure where text files are created/read and default file behavior."
                      aria-label="Text file manager help"
                    >
                      i
                    </span>
                    Set one folder path and one default file path for quick text operations.
                  </p>
                  <label className="full-width">
                    Folder path
                    <input
                      value={folderPathValue}
                      onChange={(event) =>
                        updateTemplateConfigField("folder_path", event.target.value)
                      }
                      placeholder="/Users/mathieumasson/Documents/TextFiles"
                    />
                  </label>
                  <label>
                    Default file path
                    <input
                      value={getConfigString("default_file_path", "notes.txt")}
                      onChange={(event) =>
                        updateTemplateConfigField("default_file_path", event.target.value)
                      }
                      placeholder="notes.txt"
                    />
                  </label>
                  <label>
                    Default encoding
                    <input
                      value={getConfigString("default_encoding", "utf-8")}
                      onChange={(event) =>
                        updateTemplateConfigField("default_encoding", event.target.value)
                      }
                      placeholder="utf-8"
                    />
                  </label>
                  <label>
                    Max chars read
                    <input
                      type="number"
                      min={200}
                      max={500000}
                      value={maxCharsReadValue}
                      onChange={(event) =>
                        updateTemplateConfigField(
                          "max_chars_read",
                          Math.max(200, Math.min(500000, Number(event.target.value) || 200))
                        )
                      }
                    />
                  </label>
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={autoCreateFolderValue}
                      onChange={(event) =>
                        updateTemplateConfigField("auto_create_folder", event.target.checked)
                      }
                    />
                    Auto create folder
                  </label>
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={allowOverwriteValue}
                      onChange={(event) =>
                        updateTemplateConfigField("allow_overwrite", event.target.checked)
                      }
                    />
                    Allow overwrite
                  </label>
                </>
              )}

              {agentType === "excel_manager" && (
                <>
                  <p className="field-help full-width">
                    <span
                      className="info-bubble"
                      title="Define workbook defaults so the agent can create and update spreadsheets immediately."
                      aria-label="Excel manager help"
                    >
                      i
                    </span>
                    Set folder/workbook/sheet defaults. The template preconfigures common Excel
                    actions.
                  </p>
                  <label className="full-width">
                    Folder path
                    <input
                      value={folderPathValue}
                      onChange={(event) =>
                        updateTemplateConfigField("folder_path", event.target.value)
                      }
                      placeholder="/Users/mathieumasson/Documents/ExcelFiles"
                    />
                  </label>
                  <label>
                    Workbook path
                    <input
                      value={getConfigString("workbook_path", "workbook.xlsx")}
                      onChange={(event) =>
                        updateTemplateConfigField("workbook_path", event.target.value)
                      }
                      placeholder="workbook.xlsx"
                    />
                  </label>
                  <label>
                    Default sheet
                    <input
                      value={getConfigString("default_sheet", "Sheet1")}
                      onChange={(event) =>
                        updateTemplateConfigField("default_sheet", event.target.value)
                      }
                      placeholder="Sheet1"
                    />
                  </label>
                  <label>
                    Max rows read
                    <input
                      type="number"
                      min={1}
                      max={10000}
                      value={maxRowsReadValue}
                      onChange={(event) =>
                        updateTemplateConfigField(
                          "max_rows_read",
                          Math.max(1, Math.min(10000, Number(event.target.value) || 1))
                        )
                      }
                    />
                  </label>
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={autoCreateFolderValue}
                      onChange={(event) =>
                        updateTemplateConfigField("auto_create_folder", event.target.checked)
                      }
                    />
                    Auto create folder
                  </label>
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={autoCreateWorkbookValue}
                      onChange={(event) =>
                        updateTemplateConfigField("auto_create_workbook", event.target.checked)
                      }
                    />
                    Auto create workbook
                  </label>
                </>
              )}

              {agentType === "elasticsearch_retriever" && (
                <>
                  <p className="field-help full-width">
                    <span
                      className="info-bubble"
                      title="Provide endpoint + index and optionally credentials to query Elasticsearch."
                      aria-label="Elasticsearch help"
                    >
                      i
                    </span>
                    Minimum required fields are base URL and index. Credentials are optional if
                    your cluster is open.
                  </p>
                  <label>
                    Database ID (optional)
                    <input
                      value={databaseIdValue}
                      onChange={(event) =>
                        updateTemplateConfigField("database_id", event.target.value.trim())
                      }
                    />
                  </label>
                  <label>
                    Database name (optional)
                    <input
                      value={databaseNameValue}
                      onChange={(event) =>
                        updateTemplateConfigField("database_name", event.target.value.trim())
                      }
                    />
                  </label>
                  <label className="full-width">
                    Base URL
                    <input
                      value={getConfigString("base_url", "http://localhost:9200")}
                      onChange={(event) =>
                        updateTemplateConfigField("base_url", event.target.value.trim())
                      }
                      placeholder="http://localhost:9200"
                    />
                  </label>
                  <label>
                    Index
                    <input
                      value={getConfigString("index", "")}
                      onChange={(event) =>
                        updateTemplateConfigField("index", event.target.value.trim())
                      }
                    />
                  </label>
                  <label>
                    Top K
                    <input
                      type="number"
                      min={1}
                      max={100}
                      value={topKValue}
                      onChange={(event) =>
                        updateTemplateConfigField(
                          "top_k",
                          Math.max(1, Math.min(100, Number(event.target.value) || 1))
                        )
                      }
                    />
                  </label>
                  <label className="full-width">
                    Fields (comma-separated)
                    <input
                      value={fieldsValue}
                      onChange={(event) =>
                        updateTemplateConfigField("fields", parseCsvList(event.target.value))
                      }
                      placeholder="title, content, *"
                    />
                  </label>
                  <label>
                    Username
                    <input
                      value={getConfigString("username", "")}
                      onChange={(event) =>
                        updateTemplateConfigField("username", event.target.value)
                      }
                    />
                  </label>
                  <label>
                    Password
                    <input
                      type="password"
                      value={getConfigString("password", "")}
                      onChange={(event) =>
                        updateTemplateConfigField("password", event.target.value)
                      }
                    />
                  </label>
                  <label className="full-width">
                    API key
                    <input
                      type="password"
                      value={getConfigString("api_key", "")}
                      onChange={(event) =>
                        updateTemplateConfigField("api_key", event.target.value)
                      }
                    />
                  </label>
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={verifySslValue}
                      onChange={(event) =>
                        updateTemplateConfigField("verify_ssl", event.target.checked)
                      }
                    />
                    Verify SSL
                  </label>
                </>
              )}

              {agentType === "word_manager" && (
                <>
                  <p className="field-help full-width">
                    <span
                      className="info-bubble"
                      title="Set one folder and one default .docx path for quick create/read/edit operations."
                      aria-label="Word manager help"
                    >
                      i
                    </span>
                    Configure defaults once, then ask the agent to create, read, append or replace
                    text in Word documents.
                  </p>
                  <label className="full-width">
                    Folder path
                    <input
                      value={folderPathValue}
                      onChange={(event) =>
                        updateTemplateConfigField("folder_path", event.target.value)
                      }
                      placeholder="/Users/mathieumasson/Documents/WordFiles"
                    />
                  </label>
                  <label>
                    Default document path
                    <input
                      value={getConfigString("document_path", "document.docx")}
                      onChange={(event) =>
                        updateTemplateConfigField("document_path", event.target.value)
                      }
                      placeholder="document.docx"
                    />
                  </label>
                  <label>
                    Max paragraphs read
                    <input
                      type="number"
                      min={1}
                      max={5000}
                      value={maxParagraphsReadValue}
                      onChange={(event) =>
                        updateTemplateConfigField(
                          "max_paragraphs_read",
                          Math.max(1, Math.min(5000, Number(event.target.value) || 1))
                        )
                      }
                    />
                  </label>
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={autoCreateFolderValue}
                      onChange={(event) =>
                        updateTemplateConfigField("auto_create_folder", event.target.checked)
                      }
                    />
                    Auto create folder
                  </label>
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={autoCreateDocumentValue}
                      onChange={(event) =>
                        updateTemplateConfigField("auto_create_document", event.target.checked)
                      }
                    />
                    Auto create document
                  </label>
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={allowOverwriteValue}
                      onChange={(event) =>
                        updateTemplateConfigField("allow_overwrite", event.target.checked)
                      }
                    />
                    Allow overwrite
                  </label>
                </>
              )}

              {agentType === "web_scraper" && (
                <>
                  <p className="field-help full-width">
                    <span
                      className="info-bubble"
                      title="Start URLs are the most important input; domain and depth controls keep scraping safe."
                      aria-label="Web scraper help"
                    >
                      i
                    </span>
                    Provide trusted start URLs. Then control page depth and link-following limits.
                  </p>
                  <label className="full-width">
                    Start URLs (comma-separated)
                    <input
                      value={startUrlsValue}
                      onChange={(event) =>
                        updateTemplateConfigField("start_urls", parseCsvList(event.target.value))
                      }
                      placeholder="https://example.com, https://example.com/docs"
                    />
                  </label>
                  <label className="full-width">
                    Allowed domains (comma-separated)
                    <input
                      value={allowedDomainsValue}
                      onChange={(event) =>
                        updateTemplateConfigField(
                          "allowed_domains",
                          parseCsvList(event.target.value).map((domain) => domain.toLowerCase())
                        )
                      }
                      placeholder="example.com, docs.example.com"
                    />
                  </label>
                  <label>
                    Max pages
                    <input
                      type="number"
                      min={1}
                      max={20}
                      value={maxPagesValue}
                      onChange={(event) =>
                        updateTemplateConfigField(
                          "max_pages",
                          Math.max(1, Math.min(20, Number(event.target.value) || 1))
                        )
                      }
                    />
                  </label>
                  <label>
                    Max links per page
                    <input
                      type="number"
                      min={1}
                      max={30}
                      value={maxLinksPerPageValue}
                      onChange={(event) =>
                        updateTemplateConfigField(
                          "max_links_per_page",
                          Math.max(1, Math.min(30, Number(event.target.value) || 1))
                        )
                      }
                    />
                  </label>
                  <label>
                    Max chars per page
                    <input
                      type="number"
                      min={500}
                      max={30000}
                      value={maxCharsPerPageValue}
                      onChange={(event) =>
                        updateTemplateConfigField(
                          "max_chars_per_page",
                          Math.max(500, Math.min(30000, Number(event.target.value) || 500))
                        )
                      }
                    />
                  </label>
                  <label>
                    Timeout (seconds)
                    <input
                      type="number"
                      min={3}
                      max={300}
                      value={timeoutSecondsValue}
                      onChange={(event) =>
                        updateTemplateConfigField(
                          "timeout_seconds",
                          Math.max(3, Math.min(300, Number(event.target.value) || 3))
                        )
                      }
                    />
                  </label>
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={includeUrlsFromQuestionValue}
                      onChange={(event) =>
                        updateTemplateConfigField(
                          "include_urls_from_question",
                          event.target.checked
                        )
                      }
                    />
                    Include URLs from prompt
                  </label>
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={searchFallbackValue}
                      onChange={(event) =>
                        updateTemplateConfigField("search_fallback", event.target.checked)
                      }
                    />
                    Search fallback when no URL/domain is provided
                  </label>
                  <label>
                    Search region
                    <input
                      value={getConfigString("region", "wt-wt")}
                      onChange={(event) =>
                        updateTemplateConfigField("region", event.target.value.trim())
                      }
                      placeholder="wt-wt"
                    />
                  </label>
                  <label>
                    Search safety
                    <select
                      value={getConfigString("safe_search", "moderate")}
                      onChange={(event) =>
                        updateTemplateConfigField("safe_search", event.target.value)
                      }
                    >
                      <option value="off">off</option>
                      <option value="moderate">moderate</option>
                      <option value="strict">strict</option>
                    </select>
                  </label>
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={followLinksValue}
                      onChange={(event) =>
                        updateTemplateConfigField("follow_links", event.target.checked)
                      }
                    />
                    Follow links
                  </label>
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={sameDomainOnlyValue}
                      onChange={(event) =>
                        updateTemplateConfigField("same_domain_only", event.target.checked)
                      }
                    />
                    Same domain only
                  </label>
                </>
              )}

              {agentType === "web_navigator" && (
                <>
                  <p className="field-help full-width">
                    <span
                      className="info-bubble"
                      title="This agent performs browser actions. Domain restriction is disabled; it can navigate freely."
                      aria-label="Web navigator help"
                    >
                      i
                    </span>
                    Choose a start URL (optional). If empty, the navigator will infer URL/domain
                    from the request. Navigation is unrestricted across domains.
                  </p>
                  <label className="full-width">
                    Start URL
                    <input
                      value={getConfigString("start_url", "")}
                      onChange={(event) =>
                        updateTemplateConfigField("start_url", event.target.value.trim())
                      }
                      placeholder="https://lemonde.fr (optional)"
                    />
                  </label>
                  <label>
                    Max steps
                    <input
                      type="number"
                      min={1}
                      max={30}
                      value={navigatorMaxStepsValue}
                      onChange={(event) =>
                        updateTemplateConfigField(
                          "max_steps",
                          Math.max(1, Math.min(30, Number(event.target.value) || 1))
                        )
                      }
                    />
                  </label>
                  <label>
                    Timeout (ms)
                    <input
                      type="number"
                      min={2000}
                      max={120000}
                      value={timeoutMsValue}
                      onChange={(event) =>
                        updateTemplateConfigField(
                          "timeout_ms",
                          Math.max(2000, Math.min(120000, Number(event.target.value) || 2000))
                        )
                      }
                    />
                  </label>
                  <label>
                    Captured content (chars)
                    <input
                      type="number"
                      min={500}
                      max={30000}
                      value={captureHtmlCharsValue}
                      onChange={(event) =>
                        updateTemplateConfigField(
                          "capture_html_chars",
                          Math.max(500, Math.min(30000, Number(event.target.value) || 500))
                        )
                      }
                    />
                  </label>
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={headlessValue}
                      onChange={(event) =>
                        updateTemplateConfigField("headless", event.target.checked)
                      }
                    />
                    Headless browser
                  </label>
                </>
              )}

              {agentType === "wikipedia_retriever" && (
                <>
                  <p className="field-help full-width">
                    <span
                      className="info-bubble"
                      title="Language and top K pages determine retrieval quality and response speed."
                      aria-label="Wikipedia help"
                    >
                      i
                    </span>
                    Keep language aligned with user requests and increase top K for broader context.
                  </p>
                  <label>
                    Language
                    <input
                      value={getConfigString("language", "en")}
                      onChange={(event) =>
                        updateTemplateConfigField("language", event.target.value.trim())
                      }
                      placeholder="en"
                    />
                  </label>
                  <label>
                    Top K pages
                    <input
                      type="number"
                      min={1}
                      max={10}
                      value={topKValue}
                      onChange={(event) =>
                        updateTemplateConfigField(
                          "top_k",
                          Math.max(1, Math.min(10, Number(event.target.value) || 1))
                        )
                      }
                    />
                  </label>
                  <label>
                    Summary sentences
                    <input
                      type="number"
                      min={1}
                      max={5}
                      value={summarySentencesValue}
                      onChange={(event) =>
                        updateTemplateConfigField(
                          "summary_sentences",
                          Math.max(1, Math.min(5, Number(event.target.value) || 1))
                        )
                      }
                    />
                  </label>
                </>
              )}

              {agentType === "rss_news" && (
                <>
                  <p className="field-help full-width">
                    <span
                      className="info-bubble"
                      title="Choose your RSS feeds and keywords. The agent ranks recent articles and creates a concise morning briefing."
                      aria-label="RSS news help"
                    >
                      i
                    </span>
                    Add your preferred feed URLs, define your interests, and tune freshness and
                    article limits for your breakfast briefing.
                  </p>
                  <label className="full-width">
                    RSS feed URLs (comma-separated)
                    <input
                      value={feedUrlsValue}
                      onChange={(event) =>
                        updateTemplateConfigField("feed_urls", parseCsvList(event.target.value))
                      }
                      placeholder="https://www.lemonde.fr/rss/une.xml, https://www.franceinfo.fr/titres.rss"
                    />
                  </label>
                  <label className="full-width">
                    Interests (comma-separated keywords)
                    <input
                      value={interestsValue}
                      onChange={(event) =>
                        updateTemplateConfigField("interests", parseCsvList(event.target.value))
                      }
                      placeholder="economie, ia, technologie, geopolitique"
                    />
                  </label>
                  <label className="full-width">
                    Excluded keywords (comma-separated)
                    <input
                      value={excludeKeywordsValue}
                      onChange={(event) =>
                        updateTemplateConfigField(
                          "exclude_keywords",
                          parseCsvList(event.target.value)
                        )
                      }
                      placeholder="sports, people"
                    />
                  </label>
                  <label>
                    Top articles in briefing
                    <input
                      type="number"
                      min={1}
                      max={20}
                      value={topKValue}
                      onChange={(event) =>
                        updateTemplateConfigField(
                          "top_k",
                          Math.max(1, Math.min(20, Number(event.target.value) || 1))
                        )
                      }
                    />
                  </label>
                  <label>
                    Max items fetched per feed
                    <input
                      type="number"
                      min={1}
                      max={200}
                      value={maxItemsPerFeedValue}
                      onChange={(event) =>
                        updateTemplateConfigField(
                          "max_items_per_feed",
                          Math.max(1, Math.min(200, Number(event.target.value) || 1))
                        )
                      }
                    />
                  </label>
                  <label>
                    Lookback window (hours)
                    <input
                      type="number"
                      min={1}
                      max={336}
                      value={hoursLookbackValue}
                      onChange={(event) =>
                        updateTemplateConfigField(
                          "hours_lookback",
                          Math.max(1, Math.min(336, Number(event.target.value) || 1))
                        )
                      }
                    />
                  </label>
                  <label>
                    Language hint
                    <input
                      value={languageHintValue}
                      onChange={(event) =>
                        updateTemplateConfigField("language_hint", event.target.value.trim())
                      }
                      placeholder="fr"
                    />
                  </label>
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={includeGeneralIfNoMatchValue}
                      onChange={(event) =>
                        updateTemplateConfigField(
                          "include_general_if_no_match",
                          event.target.checked
                        )
                      }
                    />
                    Include general news when no interest keyword matches
                  </label>
                </>
              )}

              <label className="full-width">
                <div className="toolbar">
                  <span>Advanced template configuration (JSON)</span>
                  <button
                    type="button"
                    className="btn-ghost"
                    onClick={() =>
                      setShowAdvancedTemplateConfig((previous) => !previous)
                    }
                  >
                    {showAdvancedTemplateConfig ? "Hide advanced JSON" : "Show advanced JSON"}
                  </button>
                </div>
                {showAdvancedTemplateConfig && (
                  <textarea
                    rows={8}
                    value={templateConfigText}
                    onChange={(event) => setTemplateConfigText(event.target.value)}
                  />
                )}
                {hasAssistedTemplateFields && parsedTemplateConfig === null && (
                  <span className="hint">
                    Template config JSON is invalid. Fix it to keep using assisted fields.
                  </span>
                )}
                {agentDraft.agent_type === "sql_analyst" && (
                  <span className="hint">
                    Optional for SQL agents: set `database_id` or `database_name` to route queries
                    to a specific database profile. For parameterized use cases, define
                    `sql_query_template` with placeholders like <code>{`{{client}}`}</code> and
                    configure `sql_parameters`.
                  </span>
                )}
                {agentDraft.agent_type === "clickhouse_table_manager" && (
                  <span className="hint">
                    Recommended for ClickHouse Table Manager: configure `database_id` or
                    `database_name`, keep `protect_existing_tables=true` by default, and tune
                    `allow_row_inserts`, `allow_row_updates`, `allow_row_deletes`,
                    and `max_statements`.
                  </span>
                )}
                {agentDraft.agent_type === "elasticsearch_retriever" && (
                  <span className="hint">
                    Optional for Elasticsearch retriever: set `database_id` or `database_name` to reuse an Elasticsearch profile from Database Settings.
                  </span>
                )}
                {agentDraft.agent_type === "web_scraper" && (
                  <span className="hint">
                    Recommended for Web Scraper: configure `start_urls` or ask with URL/domain, keep `search_fallback` enabled, then tune `max_pages` and `follow_links`.
                  </span>
                )}
                {agentDraft.agent_type === "web_navigator" && (
                  <span className="hint">
                    Recommended for Web Navigator: set optional `start_url` (or let it infer from request), `max_steps` and `timeout_ms`. Domain restriction is disabled. Requires Playwright on backend.
                  </span>
                )}
                {agentDraft.agent_type === "text_file_manager" && (
                  <span className="hint">
                    Recommended for Text File Manager: configure `folder_path` and optional `default_file_path` in template config.
                  </span>
                )}
                {agentDraft.agent_type === "excel_manager" && (
                  <span className="hint">
                    Recommended for Excel Manager: configure `folder_path`, `workbook_path`, `default_sheet`, and `max_rows_read`.
                  </span>
                )}
                {agentDraft.agent_type === "word_manager" && (
                  <span className="hint">
                    Recommended for Word Manager: configure `folder_path`, `document_path`, and `max_paragraphs_read`.
                  </span>
                )}
                {agentDraft.agent_type === "rss_news" && (
                  <span className="hint">
                    Recommended for RSS / News: configure `feed_urls`, `interests`, `top_k`,
                    `max_items_per_feed`, and `hours_lookback`.
                  </span>
                )}
              </label>

              <label className="full-width">
                <div className="toolbar">
                  <span>Advanced prompts (optional)</span>
                  <button
                    type="button"
                    className="btn-ghost"
                    onClick={() =>
                      setShowAdvancedPrompts((previous) => !previous)
                    }
                  >
                    {showAdvancedPrompts ? "Hide prompts" : "Show prompts"}
                  </button>
                </div>
                <span className="hint">
                  Templates prefill these prompts. Edit only if you need custom behavior.
                </span>
                {showAdvancedPrompts && (
                  <div className="grid">
                    <label className="full-width">
                      System prompt
                      <textarea
                        rows={3}
                        value={agentDraft.system_prompt}
                        onChange={(event) =>
                          setAgentDraft((prev) => ({
                            ...prev,
                            system_prompt: event.target.value
                          }))
                        }
                      />
                    </label>

                    <label className="full-width">
                      SQL prompt template
                      <textarea
                        rows={6}
                        value={agentDraft.sql_prompt_template}
                        onChange={(event) =>
                          setAgentDraft((prev) => ({
                            ...prev,
                            sql_prompt_template: event.target.value
                          }))
                        }
                      />
                    </label>

                    <label className="full-width">
                      Answer prompt template
                      <textarea
                        rows={6}
                        value={agentDraft.answer_prompt_template}
                        onChange={(event) =>
                          setAgentDraft((prev) => ({
                            ...prev,
                            answer_prompt_template: event.target.value
                          }))
                        }
                      />
                    </label>
                  </div>
                )}
              </label>

              <label className="checkbox-row full-width">
                <input
                  type="checkbox"
                  checked={agentDraft.enabled}
                  onChange={(event) =>
                    setAgentDraft((prev) => ({ ...prev, enabled: event.target.checked }))
                  }
                />
                Agent enabled
              </label>
            </div>
            )}

            {selectedAgent && (
              <section className="card">
                <div className="card-header">
                  <h3>Agent audit history</h3>
                </div>
                <p className="hint">
                  Last 5 previous configurations are kept automatically on updates and restores.
                </p>

                {isAuditLoading ? (
                  <p className="hint">Loading history...</p>
                ) : agentAuditHistory.length === 0 ? (
                  <p className="hint">No previous configuration saved yet.</p>
                ) : (
                  <div className="timeline">
                    {agentAuditHistory.map((entry) => (
                      <article className="timeline-item" key={entry.version_id}>
                        <div className="timeline-header">
                          <span className="timeline-type">
                            {entry.reason === "update" ? "Update snapshot" : "Restore snapshot"}
                          </span>
                          <span className="timeline-time">
                            {new Date(entry.created_at).toLocaleString()}
                          </span>
                        </div>
                        <p className="hint">
                          Name: <strong>{entry.snapshot.name}</strong> • Type:{" "}
                          <strong>{entry.snapshot.agent_type}</strong> • Max rows:{" "}
                          <strong>{entry.snapshot.max_rows}</strong> •{" "}
                          {entry.snapshot.enabled ? "Enabled" : "Disabled"}
                        </p>
                        <div className="button-row">
                          <button
                            className="btn-ghost"
                            onClick={() => restoreVersion(entry.version_id)}
                          >
                            Restore this version
                          </button>
                        </div>
                      </article>
                    ))}
                  </div>
                )}
              </section>
            )}
          </section>
        </div>
      </section>
    </div>
  );
}
