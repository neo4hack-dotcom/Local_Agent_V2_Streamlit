# VSCode Simple Python Runner

This folder provides a simple Python program that reuses the **same backend engine** as the Local Agent app:

- agent execution (`AgentExecutor`)
- multi-agent manager orchestration (`MultiAgentManager`)
- local LLM support (Ollama or HTTP)
- database connectors (ClickHouse / Oracle / Elasticsearch)

## 1) Configure

Edit:

- `/Users/mathieumasson/.codex/worktrees/0f71/New project/Local_Agent/vscode_simple_runner/config_page.py`

This file is fully commented and contains:

- `LLM_CONFIG`
- `MANAGER_CONFIG`
- `DATABASES`
- `ACTIVE_DATABASE_ID`
- `AGENTS`

## 2) Run from VSCode terminal

From project root:

```bash
cd /Users/mathieumasson/.codex/worktrees/0f71/New\ project/Local_Agent
python vscode_simple_runner/run_vscode_agent_app.py
```

## 2-bis) Run V2 UI (Streamlit)

Install Streamlit once in your Python environment:

```bash
pip install streamlit
```

Then launch the UI:

```bash
cd /Users/mathieumasson/.codex/worktrees/0f71/New\ project/Local_Agent
streamlit run vscode_simple_runner/streamlit_app.py
```

The V2 UI provides tabs for:

- Configuration (LLM + manager + DB)
- Agents (list/create/edit/delete)
- Playground (single-agent and multi-agent manager)

## 3) Menu actions

The runner exposes:

1. configuration summary
2. LLM connection test
3. DB connection tests
4. single agent run
5. manager multi-agent run
6. config reload (after editing `config_page.py`)

## Notes

- For local LLM, verify Ollama is running (`http://localhost:11434` by default).
- Keep secrets in environment variables when possible.
- `web_navigator` requires Playwright runtime installed in backend environment.
- V2 runtime changes are saved in:
  `/Users/mathieumasson/.codex/worktrees/0f71/New project/Local_Agent/vscode_simple_runner/runtime_config_v2.json`
