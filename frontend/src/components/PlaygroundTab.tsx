import { useMemo, useState } from "react";

import { runAgent, runManagerStream } from "../api";
import {
  AgentRunResponse,
  AgentConfig,
  ConversationTurn,
  ManagerConfig,
  ManagerEvent,
  ManagerFinalEvent,
  WebhookConfig
} from "../types";

interface PlaygroundTabProps {
  agents: AgentConfig[];
  managerConfig: ManagerConfig;
  webhookConfig: WebhookConfig;
  onNotify: (message: string, tone?: "success" | "error") => void;
}

type PlaygroundMode = "single" | "manager";
type WorkflowTone = "manager" | "agent" | "success" | "warning" | "error";

interface WorkflowCard {
  id: string;
  title: string;
  subtitle: string;
  detail: string;
  tone: WorkflowTone;
}

interface ParsedManagerSummary {
  worked: string[];
  notWorked: string[];
  called: string[];
}

interface TimelineEntry {
  event: ManagerEvent;
  card: WorkflowCard;
}

function normalizeBulletLine(value: string): string {
  return value.replace(/^[-*]\s+/, "").trim();
}

function parseManagerSummary(raw: string): ParsedManagerSummary {
  const sections: ParsedManagerSummary = {
    worked: [],
    notWorked: [],
    called: []
  };

  let current: keyof ParsedManagerSummary | null = null;
  const lines = raw.split(/\r?\n/).map((line) => line.trim());

  for (const line of lines) {
    if (!line) {
      continue;
    }

    const lowered = line.toLowerCase();
    if (lowered.startsWith("what worked")) {
      current = "worked";
      continue;
    }
    if (lowered.startsWith("what did not work")) {
      current = "notWorked";
      continue;
    }
    if (lowered.startsWith("agents called and purpose")) {
      current = "called";
      continue;
    }

    if (!current) {
      continue;
    }

    const cleaned = normalizeBulletLine(line);
    if (!cleaned || cleaned.toLowerCase() === "none") {
      continue;
    }
    sections[current].push(cleaned);
  }

  return sections;
}

function clampPercent(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.min(100, Math.round(value)));
}

const MAX_PLAYGROUND_MEMORY_TURNS = 12;

export function PlaygroundTab({
  agents,
  managerConfig,
  webhookConfig,
  onNotify
}: PlaygroundTabProps) {
  const [mode, setMode] = useState<PlaygroundMode>("manager");
  const [agentId, setAgentId] = useState<string>(agents[0]?.id ?? "");
  const [question, setQuestion] = useState<string>(
    "Build a concise business summary of what happened in the last run and explain confidence."
  );

  const [singleResult, setSingleResult] = useState<AgentRunResponse | null>(null);
  const [managerEvents, setManagerEvents] = useState<ManagerEvent[]>([]);
  const [managerFinal, setManagerFinal] = useState<ManagerFinalEvent | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [isWorkflowOpen, setIsWorkflowOpen] = useState(false);
  const [useMemory, setUseMemory] = useState(true);
  const [conversationHistory, setConversationHistory] = useState<ConversationTurn[]>([]);

  const activeAgent = useMemo(
    () => agents.find((item) => item.id === agentId),
    [agents, agentId]
  );

  const workflowCards = useMemo<WorkflowCard[]>(() => {
    return managerEvents.map((event, index) => {
      const base: WorkflowCard = {
        id: `${String(event.ts)}-${index}`,
        title: String(event.type),
        subtitle: String(event.ts),
        detail: "",
        tone: "manager"
      };
      const agentName = String(event.agent_name ?? event.agent_id ?? "Agent");
      const questionText = String(event.question ?? "").trim();

      if (event.type === "manager_start") {
        return {
          ...base,
          title: "Manager started",
          subtitle: `Question: ${String(event.question ?? "")}`,
          detail: `Budget: steps=${String(event.max_steps ?? "-")} • calls=${String(event.max_agent_calls ?? "-")}`,
          tone: "manager"
        };
      }

      if (event.type === "manager_decision") {
        const callCount = Array.isArray(event.calls) ? event.calls.length : 0;
        return {
          ...base,
          title: `Decision: ${String(event.status ?? "continue")}`,
          subtitle: String(event.rationale ?? "").trim() || "Decision computed",
          detail:
            callCount > 0
              ? `Proposed calls: ${String(callCount)}`
              : "No call proposed",
          tone: "manager"
        };
      }

      if (event.type === "manager_warning") {
        return {
          ...base,
          title: "Manager warning",
          subtitle: String(event.message ?? "").trim() || "No warning details",
          detail: "",
          tone: "warning"
        };
      }

      if (event.type === "agent_call_started") {
        return {
          ...base,
          title: `Call started: ${agentName}`,
          subtitle: questionText ? `Task: ${questionText}` : "Agent task started",
          detail: event.database_name ? `Database: ${String(event.database_name)}` : "",
          tone: "agent"
        };
      }

      if (event.type === "agent_call_completed") {
        return {
          ...base,
          title: `Call completed: ${agentName}`,
          subtitle: String(event.answer ?? "").slice(0, 180) || "No answer text",
          detail: `Rows: ${String(event.row_count ?? 0)}`,
          tone: "success"
        };
      }

      if (event.type === "agent_call_failed") {
        return {
          ...base,
          title: `Call failed: ${agentName}`,
          subtitle: String(event.error ?? "").trim() || "Execution failed",
          detail: "",
          tone: "error"
        };
      }

      if (event.type === "agent_marked_unavailable") {
        return {
          ...base,
          title: `Agent unavailable: ${agentName}`,
          subtitle:
            String(event.reason ?? "").trim() || "Agent was marked unavailable at runtime",
          detail: "",
          tone: "warning"
        };
      }

      if (event.type === "manager_final") {
        const status = String(event.status ?? "done");
        return {
          ...base,
          title: `Final status: ${status}`,
          subtitle: String(event.answer ?? "").slice(0, 220),
          detail: String(event.missing_information ?? ""),
          tone: status === "done" ? "success" : "warning"
        };
      }

      return base;
    });
  }, [managerEvents]);

  const timelineEntries = useMemo<TimelineEntry[]>(() => {
    const reversedEvents = [...managerEvents].reverse();
    const reversedCards = [...workflowCards].reverse();
    return reversedEvents.map((event, index) => ({
      event,
      card:
        reversedCards[index] ?? {
          id: `${String(event.ts)}-fallback-${index}`,
          title: String(event.type),
          subtitle: String(event.ts),
          detail: "",
          tone: "manager"
        }
    }));
  }, [managerEvents, workflowCards]);

  const managerSummaryText = String(managerFinal?.manager_summary ?? "").trim();
  const managerSummary = useMemo(
    () => parseManagerSummary(managerSummaryText),
    [managerSummaryText]
  );

  const reasoningTrail = useMemo(() => {
    const rows: string[] = [];
    for (const event of managerEvents) {
      if (event.type !== "manager_decision") {
        continue;
      }
      const stepValue = event.step === undefined ? "-" : String(event.step);
      const status = String(event.status ?? "continue").toUpperCase();
      const rationale = String(event.rationale ?? "").trim() || "No rationale provided.";
      rows.push(`Step ${stepValue} [${status}] - ${rationale}`);
    }
    return rows.slice(-8);
  }, [managerEvents]);

  const methodTrail = useMemo(() => {
    type StartedCall = {
      agentName: string;
      question: string;
      databaseName: string;
    };

    const keyFromEvent = (event: ManagerEvent) =>
      `${String(event.step ?? "")}:${String(event.call_index ?? "")}:${String(event.agent_id ?? "")}`;

    const starts = new Map<string, StartedCall>();
    const rows: string[] = [];

    for (const event of managerEvents) {
      if (event.type === "agent_call_started") {
        starts.set(keyFromEvent(event), {
          agentName: String(event.agent_name ?? event.agent_id ?? "Agent"),
          question: String(event.question ?? "").trim(),
          databaseName: String(event.database_name ?? "").trim()
        });
        continue;
      }

      if (event.type === "agent_call_completed") {
        const key = keyFromEvent(event);
        const started = starts.get(key);
        const agentName = started?.agentName ?? String(event.agent_name ?? event.agent_id ?? "Agent");
        const task = started?.question || "Task not provided";
        const rowCount = String(event.row_count ?? 0);
        rows.push(`${agentName}: ${task} -> success (${rowCount} rows)`);
        starts.delete(key);
        continue;
      }

      if (event.type === "agent_call_failed") {
        const key = keyFromEvent(event);
        const started = starts.get(key);
        const agentName = started?.agentName ?? String(event.agent_name ?? event.agent_id ?? "Agent");
        const task = started?.question || "Task not provided";
        const reason = String(event.error ?? "Execution failed").trim();
        rows.push(`${agentName}: ${task} -> failed (${reason})`);
        starts.delete(key);
      }
    }

    for (const started of starts.values()) {
      const dbInfo = started.databaseName ? ` on ${started.databaseName}` : "";
      rows.push(`${started.agentName}: ${started.question || "Task not provided"}${dbInfo} -> started`);
    }

    return rows.slice(-12);
  }, [managerEvents]);

  const judgeVerdict = String(managerFinal?.judge_verdict ?? "partial").trim().toLowerCase();
  const judgeConfidence = Number(managerFinal?.judge_confidence ?? NaN);
  const confidencePercent = Number.isFinite(judgeConfidence)
    ? clampPercent(judgeConfidence)
    : null;
  const judgeRationale = String(managerFinal?.judge_rationale ?? "").trim();
  const judgeChecksPassed = Array.isArray(managerFinal?.judge_checks_passed)
    ? managerFinal.judge_checks_passed
    : [];
  const judgeChecksFailed = Array.isArray(managerFinal?.judge_checks_failed)
    ? managerFinal.judge_checks_failed
    : [];
  const judgeRecommendations = Array.isArray(managerFinal?.judge_recommendations)
    ? managerFinal.judge_recommendations
    : [];
  const webhookReplacementEnabled =
    webhookConfig.enabled &&
    webhookConfig.replace_playground &&
    webhookConfig.url.trim().length > 0;

  const executeSingle = async () => {
    if (webhookReplacementEnabled) {
      onNotify(
        "Playground is replaced by external webhook UI. Disable replacement in Configuration to run locally.",
        "error"
      );
      return;
    }
    if (!agentId) {
      onNotify("No agent selected.", "error");
      return;
    }
    if (!question.trim()) {
      onNotify("Question is empty.", "error");
      return;
    }

    try {
      setIsRunning(true);
      setManagerEvents([]);
      setManagerFinal(null);
      const memoryForRequest = useMemory ? conversationHistory : [];
      const response = await runAgent(agentId, {
        question,
        conversation_history: memoryForRequest
      });
      setSingleResult(response);
      setConversationHistory((previous) => {
        const next: ConversationTurn[] = [
          ...previous,
          { role: "user", content: question.trim() },
          { role: "assistant", content: String(response.answer ?? "").trim() || "(empty answer)" }
        ];
        return next.slice(-MAX_PLAYGROUND_MEMORY_TURNS);
      });
      onNotify("Execution completed.");
    } catch (error) {
      onNotify(error instanceof Error ? error.message : "Execution failed.", "error");
    } finally {
      setIsRunning(false);
    }
  };

  const executeManager = async () => {
    if (webhookReplacementEnabled) {
      onNotify(
        "Playground is replaced by external webhook UI. Disable replacement in Configuration to run locally.",
        "error"
      );
      return;
    }
    if (!question.trim()) {
      onNotify("Question is empty.", "error");
      return;
    }

    try {
      setIsRunning(true);
      setSingleResult(null);
      setManagerEvents([]);
      setManagerFinal(null);
      setIsWorkflowOpen(true);
      const memoryForRequest = useMemory ? conversationHistory : [];

      const final = await runManagerStream(
        {
          question,
          max_steps: managerConfig.max_steps,
          max_agent_calls: managerConfig.max_agent_calls,
          conversation_history: memoryForRequest
        },
        (event) => {
          setManagerEvents((previous) => [...previous, event]);
        }
      );

      setManagerFinal(final);
      setConversationHistory((previous) => {
        const next: ConversationTurn[] = [
          ...previous,
          { role: "user", content: question.trim() },
          { role: "assistant", content: String(final.answer ?? "").trim() || "(empty answer)" }
        ];
        return next.slice(-MAX_PLAYGROUND_MEMORY_TURNS);
      });
      if (final.status === "done") {
        onNotify("Manager completed the request.");
      } else {
        onNotify("Manager stopped before full completion.", "error");
      }
    } catch (error) {
      onNotify(error instanceof Error ? error.message : "Manager execution failed.", "error");
    } finally {
      setIsRunning(false);
    }
  };

  const execute = async () => {
    if (mode === "single") {
      await executeSingle();
      return;
    }
    await executeManager();
  };

  return (
    <div className="tab-layout">
      <section className="card playground-card">
        <header className="card-header">
          <h2>LangGraph Playground</h2>
          <div className="button-row">
            {mode === "manager" && (
              <button className="btn-ghost" onClick={() => setIsWorkflowOpen(true)}>
                Open workflow view
              </button>
            )}
            <button
              className="btn-primary"
              onClick={execute}
              disabled={isRunning || webhookReplacementEnabled}
            >
              {isRunning ? "Running..." : mode === "manager" ? "Run manager" : "Run"}
            </button>
          </div>
        </header>

        {webhookReplacementEnabled && (
          <p className="hint webhook-replacement-banner">
            Playground execution is currently delegated to external UI via webhook:{" "}
            <strong>{webhookConfig.url}</strong>. You still receive manager steps and final answer in
            the webhook payloads.
          </p>
        )}

        <p className="hint playground-intro">
          Write a natural-language request. The manager will orchestrate agents, then produce a
          detailed final answer with reasoning, method and confidence.
        </p>

        <div className="grid two-columns">
          <label>
            Mode
            <select value={mode} onChange={(event) => setMode(event.target.value as PlaygroundMode)}>
              <option value="manager">Multi-agent manager</option>
              <option value="single">Single agent</option>
            </select>
          </label>

          {mode === "single" && (
            <label>
              Agent
              <select value={agentId} onChange={(event) => setAgentId(event.target.value)}>
                {agents.map((agent) => (
                  <option value={agent.id} key={agent.id}>
                    {agent.name}
                  </option>
                ))}
              </select>
            </label>
          )}

          <label className="full-width">
            Request
            <textarea
              rows={4}
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Example: Summarize what happened in the sales dashboard and explain confidence."
            />
          </label>

          <div className="full-width playground-memory-toolbar">
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={useMemory}
                onChange={(event) => setUseMemory(event.target.checked)}
              />
              Use conversation memory
            </label>
            <span className="hint">
              Stored turns: <strong>{conversationHistory.length}</strong> / {MAX_PLAYGROUND_MEMORY_TURNS}
            </span>
            <button
              type="button"
              className="btn-ghost"
              onClick={() => setConversationHistory([])}
              disabled={conversationHistory.length === 0}
            >
              Clear memory
            </button>
          </div>
        </div>

        {mode === "single" ? (
          <div className="hint">
            Active agent: <strong>{activeAgent?.name ?? "None"}</strong>
            {" • Database selection is handled by each agent configuration."}
          </div>
        ) : (
          <div className="hint">
            Manager orchestration is streamed in real time. Budget: steps <strong>{managerConfig.max_steps}</strong>
            {" • "}
            calls <strong>{managerConfig.max_agent_calls}</strong>.
          </div>
        )}

        {conversationHistory.length > 0 && (
          <section className="memory-card">
            <div className="card-header">
              <h3>Conversation memory</h3>
              <span className="hint">Latest context used for follow-up questions</span>
            </div>
            <div className="memory-list">
              {conversationHistory.map((turn, index) => (
                <article
                  key={`${turn.role}-${index}-${turn.content.slice(0, 24)}`}
                  className={`memory-item ${turn.role}`}
                >
                  <span className="memory-role">{turn.role === "user" ? "You" : "Assistant"}</span>
                  <p>{turn.content}</p>
                </article>
              ))}
            </div>
          </section>
        )}

        {mode === "manager" && managerFinal && (
          <section className="manager-outcome">
            <div className="manager-outcome-head">
              <span className={`status-pill ${managerFinal.status}`}>
                Status: {managerFinal.status}
              </span>
              <span className={`judge-pill ${judgeVerdict || "partial"}`}>
                Sanity judge: {judgeVerdict || "partial"}
              </span>
            </div>

            <article className="manager-answer-hero">
              <h3>Manager final answer</h3>
              <p>{managerFinal.answer || "(empty)"}</p>
            </article>

            <div className="manager-metrics-row">
              <div className="metric-chip">
                <span>Steps</span>
                <strong>{managerFinal.steps}</strong>
              </div>
              <div className="metric-chip">
                <span>Agent calls</span>
                <strong>{managerFinal.agent_calls}</strong>
              </div>
              <div className="metric-chip">
                <span>Confidence</span>
                <strong>
                  {confidencePercent === null ? "n/a" : `${String(confidencePercent)}%`}
                </strong>
              </div>
            </div>

            <article className="confidence-card">
              <div className="confidence-head">
                <h4>Trust level</h4>
                <strong>
                  {confidencePercent === null ? "n/a" : `${String(confidencePercent)}%`}
                </strong>
              </div>
              <div className="confidence-track">
                <div
                  className={`confidence-fill ${judgeVerdict || "partial"}`}
                  style={{ width: `${String(confidencePercent ?? 0)}%` }}
                />
              </div>
              <p className="hint">{judgeRationale || "No judge rationale available."}</p>
            </article>

            <div className="manager-details-grid">
              <article className="manager-detail-card">
                <h4>How the manager thought</h4>
                {reasoningTrail.length === 0 ? (
                  <p className="hint">No explicit decision rationale was captured.</p>
                ) : (
                  <ul className="summary-list">
                    {reasoningTrail.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                )}
              </article>

              <article className="manager-detail-card">
                <h4>Method and chain used</h4>
                {methodTrail.length === 0 ? (
                  <p className="hint">No agent method trace available yet.</p>
                ) : (
                  <ul className="summary-list">
                    {methodTrail.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                )}
              </article>
            </div>

            <article className="manager-detail-card">
              <h4>Execution summary</h4>
              <p className="manager-summary-text">
                {managerSummaryText || "Summary unavailable."}
              </p>
            </article>

            <div className="manager-details-grid">
              <article className="manager-detail-card">
                <h4>What worked</h4>
                {managerSummary.worked.length > 0 ? (
                  <ul className="summary-list">
                    {managerSummary.worked.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="hint">No explicit success bullet was reported.</p>
                )}

                <h4>Judge checks passed</h4>
                {judgeChecksPassed.length > 0 ? (
                  <ul className="summary-list">
                    {judgeChecksPassed.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="hint">none</p>
                )}
              </article>

              <article className="manager-detail-card">
                <h4>What did not work</h4>
                {managerSummary.notWorked.length > 0 ? (
                  <ul className="summary-list">
                    {managerSummary.notWorked.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="hint">No explicit failure bullet was reported.</p>
                )}

                <h4>Judge checks failed</h4>
                {judgeChecksFailed.length > 0 ? (
                  <ul className="summary-list">
                    {judgeChecksFailed.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="hint">none</p>
                )}
              </article>
            </div>

            <article className="manager-detail-card">
              <h4>Recommendations</h4>
              {judgeRecommendations.length > 0 ? (
                <ul className="summary-list">
                  {judgeRecommendations.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              ) : (
                <p className="hint">No recommendation provided.</p>
              )}

              {managerFinal.missing_information && (
                <p className="hint">
                  Missing information: <strong>{managerFinal.missing_information}</strong>
                </p>
              )}
            </article>
          </section>
        )}
      </section>

      {mode === "single" && singleResult && (
        <section className="card">
          {singleResult.sql ? (
            <>
              <h3>Generated SQL</h3>
              <pre>{singleResult.sql}</pre>
            </>
          ) : (
            <p className="hint">No SQL generated for this agent type.</p>
          )}

          <h3>Agent answer</h3>
          <p>{singleResult.answer || "(empty)"}</p>

          <h3>Result rows</h3>
          <pre>{JSON.stringify(singleResult.rows, null, 2)}</pre>

          <h3>Execution details</h3>
          <pre>{JSON.stringify(singleResult.details ?? {}, null, 2)}</pre>
        </section>
      )}

      {mode === "manager" && (
        <section className="card">
          <h3>Execution timeline</h3>
          <p className="hint">Most recent action is shown first. Open raw event only when needed.</p>
          {timelineEntries.length === 0 ? (
            <p className="hint">No events yet. Run the manager to start orchestration.</p>
          ) : (
            <div className="timeline">
              {timelineEntries.map((entry) => (
                <article
                  className={`timeline-item timeline-item-${entry.card.tone}`}
                  key={entry.card.id}
                >
                  <div className="timeline-header">
                    <span className="timeline-type">{entry.card.title}</span>
                    <span className="timeline-time">{entry.card.subtitle}</span>
                  </div>
                  {entry.card.detail && <p className="timeline-detail">{entry.card.detail}</p>}
                  <details className="timeline-raw">
                    <summary>Raw event</summary>
                    <pre>{JSON.stringify(entry.event, null, 2)}</pre>
                  </details>
                </article>
              ))}
            </div>
          )}
        </section>
      )}

      {mode === "manager" && isWorkflowOpen && (
        <div className="modal-backdrop" onClick={() => setIsWorkflowOpen(false)}>
          <section className="workflow-modal" onClick={(event) => event.stopPropagation()}>
            <header className="card-header">
              <h2>Live Workflow View</h2>
              <button className="btn-ghost" onClick={() => setIsWorkflowOpen(false)}>
                Close
              </button>
            </header>
            <p className="hint">
              Real-time execution graph from manager start to latest event.
            </p>

            {workflowCards.length === 0 ? (
              <p className="hint">No events yet. Run the manager to populate the workflow.</p>
            ) : (
              <div className="workflow-track">
                {workflowCards.map((card) => (
                  <article className={`workflow-node ${card.tone}`} key={card.id}>
                    <div className="workflow-node-header">
                      <strong>{card.title}</strong>
                      <span className="timeline-time">{card.subtitle}</span>
                    </div>
                    {card.detail && <p>{card.detail}</p>}
                  </article>
                ))}
              </div>
            )}
          </section>
        </div>
      )}
    </div>
  );
}
