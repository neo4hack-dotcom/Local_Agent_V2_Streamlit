import { useEffect, useState } from "react";

import { fetchAgents, fetchConfig } from "./api";
import { AgentsTab } from "./components/AgentsTab";
import { ConfigurationTab } from "./components/ConfigurationTab";
import { PlaygroundTab } from "./components/PlaygroundTab";
import { AgentConfig, AppConfig } from "./types";

type Tab = "configuration" | "agents" | "playground";

interface NoticeState {
  tone: "success" | "error";
  message: string;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>("configuration");
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [agents, setAgents] = useState<AgentConfig[]>([]);
  const [loading, setLoading] = useState(true);
  const [startupError, setStartupError] = useState<string | null>(null);
  const [notice, setNotice] = useState<NoticeState | null>(null);

  const notify = (message: string, tone: "success" | "error" = "success") => {
    setNotice({ message, tone });
    window.setTimeout(() => {
      setNotice((previous) => (previous?.message === message ? null : previous));
    }, 4000);
  };

  const refreshData = async () => {
    const [nextConfig, nextAgents] = await Promise.all([fetchConfig(), fetchAgents()]);
    setConfig(nextConfig);
    setAgents(nextAgents);
    setStartupError(null);
  };

  useEffect(() => {
    const bootstrap = async () => {
      try {
        await refreshData();
      } catch (error) {
        const message =
          error instanceof Error ? error.message : "Unable to load configuration.";
        setStartupError(message);
        notify(
          message,
          "error"
        );
      } finally {
        setLoading(false);
      }
    };

    void bootstrap();
  }, []);

  if (loading) {
    return <div className="loading">Loading Local Agent Studio...</div>;
  }

  if (!config) {
    return (
      <div className="loading">
        <div>
          <p>Configuration unavailable.</p>
          <p className="hint">
            Start backend + frontend with <code>npm run setup</code> then <code>npm run dev</code>.
          </p>
          {startupError && (
            <p className="hint">
              API error: <strong>{startupError}</strong>
            </p>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="app-root">
      <aside className="sidebar">
        <div className="brand">
          <span className="eyebrow">AI Operations</span>
          <h1>Local Agent Studio</h1>
          <p>
            Configure your LLM and data connections, then run LangGraph agents
            without writing code.
          </p>
        </div>

        <nav className="menu">
          <button
            className={activeTab === "configuration" ? "menu-item active" : "menu-item"}
            onClick={() => setActiveTab("configuration")}
          >
            Settings
          </button>
          <button
            className={activeTab === "agents" ? "menu-item active" : "menu-item"}
            onClick={() => setActiveTab("agents")}
          >
            Agents
          </button>
          <button
            className={activeTab === "playground" ? "menu-item active" : "menu-item"}
            onClick={() => setActiveTab("playground")}
          >
            Playground
          </button>
        </nav>
      </aside>

      <main className="content">
        {notice && (
          <div className={notice.tone === "error" ? "notice error" : "notice success"}>
            {notice.message}
          </div>
        )}

        {activeTab === "configuration" && (
          <ConfigurationTab
            config={config}
            agents={agents}
            onRefresh={refreshData}
            onNotify={notify}
          />
        )}

        {activeTab === "agents" && (
          <AgentsTab
            agents={agents}
            managerConfig={config.manager}
            onRefresh={refreshData}
            onNotify={notify}
          />
        )}

        {activeTab === "playground" && (
          <PlaygroundTab
            agents={agents}
            managerConfig={config.manager}
            webhookConfig={config.webhook}
            onNotify={notify}
          />
        )}
      </main>
    </div>
  );
}
