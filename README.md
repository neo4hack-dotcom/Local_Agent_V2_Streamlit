# Local Agent Studio

Fullstack application to manage LangGraph AI agents that can work with ClickHouse, Oracle, or Elasticsearch, using a React configuration interface.

## Features

- React frontend with menu:
  - LLM settings (`ollama` or HTTP endpoint)
  - Database connection settings (ClickHouse / Oracle / Elasticsearch)
  - External webhook settings (optional replacement for internal Playground UI)
  - AI agent configuration (prompts, allowed tables, limits)
  - Execution playground:
    - Single-agent execution (question -> SQL -> result -> answer)
    - Multi-agent manager mode (dynamic orchestration with real-time trace)
- Python FastAPI backend:
  - Settings API and CRUD for agents/database profiles
  - LangGraph-based agent runner
  - Multi-agent manager orchestration with streamed timeline events
  - Progressive SQL orchestration support: manager can chain multiple SQL-capable agents step-by-step (schema discovery -> scoped queries -> cross-table analysis)
  - Autonomous data-analysis memory: tracks discovered tables/columns/scopes and auto-continues SQL exploration when evidence is insufficient
  - Real-time webhook forwarding of manager events (steps + final answer + optional full timeline)
  - Automatic LLM-as-a-Judge sanity check before manager final response (verdict + confidence)
  - Read-only DB connectors (SELECT only)

## Available agent types

- `sql_analyst`: Generate and execute SQL on ClickHouse/Oracle to answer data questions, or run parameterized SQL use-case templates (for example `client = {{client}}`) with LLM-extracted inputs.
- `clickhouse_table_manager`: Create ClickHouse tables and run controlled DML/DDL workflows (insert/update/delete depending on safety settings).
- `unstructured_to_structured`: Extract structured JSON from free text.
- `email_cleaner`: Remove noise from emails and keep only essential information.
- `file_assistant`: Read/search files from a configured folder and answer with grounded context.
- `text_file_manager`: Open/read/create/edit plain text files in a configured folder.
- `excel_manager`: Create/read/edit/append data in Excel workbooks (`.xlsx`).
- `word_manager`: Create/read/edit Word documents (`.docx`) in a configured folder.
- `elasticsearch_retriever`: Query Elasticsearch and summarize relevant evidence.
- `rag_context`: Retrieve business context from local documents before answering.
- `web_scraper`: Scrape configured web pages and extract useful content (can infer domains from prompts and use search fallback when enabled).
- `web_navigator`: Navigate websites step-by-step (open pages, click, fill forms) without domain allowlist blocking.
- `wikipedia_retriever`: Retrieve and summarize information from Wikipedia.
- `rss_news`: Aggregate RSS/Atom feeds, filter by interests, and generate a short morning news briefing.

In addition to these agent types, a **Multi-Agent Manager** can orchestrate several agents dynamically in the Playground.

## External webhook mode (Open WebUI compatible)

You can forward the same manager execution details used by the Playground to an external UI through webhook.

1. Open **Settings > External Webhook UI**.
2. Configure:
   - `Enable webhook forwarding`
   - `Webhook URL`
   - optional `Auth token` / `Headers JSON`
   - optional `Replace built-in Playground with external UI`
3. Click **Test webhook** then **Save**.

When enabled, every manager run sends event payloads to your webhook:
- `manager_start`
- `manager_decision`
- `manager_warning`
- `agent_call_started`
- `agent_call_completed`
- `agent_call_failed`
- `agent_marked_unavailable`
- `manager_final`

Payload envelope fields:
- `kind`: `manager_event`
- `run`: includes `run_id`, question, limits, channel (`manager_stream` or `manager_sync`)
- `event`: exact manager event object (same structure as Playground timeline)
- `is_final`: true on `manager_final`
- `timeline`: optional complete timeline on final event (if enabled)

### Open WebUI bridge (ready-to-run)

If your Open WebUI endpoint expects a chat-like payload (`content`) instead of raw manager events, use the built-in bridge:

1. Export your Open WebUI webhook URL:

```bash
export OPEN_WEBUI_WEBHOOK_URL="https://<openwebui-host>/api/v1/channels/<channel_id>/webhook/<token>"
```

2. Optional bridge settings:

```bash
export OPEN_WEBUI_BRIDGE_PORT=8010
export OPEN_WEBUI_AUTH_TOKEN=""                       # optional
export OPEN_WEBUI_EXTRA_HEADERS_JSON='{"X-App":"LAS"}' # optional
export OPEN_WEBUI_BRIDGE_VERIFY_SSL=true
export OPEN_WEBUI_BRIDGE_TIMEOUT_SECONDS=15
```

3. Start bridge:

```bash
npm run dev:webhook-bridge
```

4. In **Configuration > External Webhook UI** set:
   - `Enable webhook forwarding` = ON
   - `Webhook URL` = `http://localhost:8010/webhook/agent-events`
   - click **Test webhook** then **Save**

Bridge health endpoint:
- `GET http://localhost:8010/health`

## Architecture

- `frontend/`: React UI (Vite + TypeScript)
- `backend/`: FastAPI + LangGraph logic

## Requirements

- Node.js 20+
- Python 3.11+
- A local LLM (example: Ollama) or compatible HTTP endpoint
- Access to a ClickHouse or Oracle database

## One-click setup and start

Run these commands from the project root:

```bash
npm run setup
npm run dev
```

This will:
- install frontend dependencies
- create/update `backend/.venv` and install backend dependencies
- install Playwright Chromium browser binaries (required by `web_navigator`)
- start frontend (`http://localhost:5173`) and backend (`http://localhost:8000`)

## Start backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
python -m playwright install chromium
uvicorn app.main:app --reload --port 8000
```

## Start frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend expects the API at `http://localhost:8000/api`.

## Initial setup

1. Open the **Settings** tab.
2. Configure the LLM:
   - `ollama`: base URL `http://localhost:11434`, model example `llama3.1`
   - `http`: set HTTP endpoint and optional headers/API key
3. Create a ClickHouse or Oracle DB profile.
4. Test the connection and set it as active.
5. Open the **Agents** tab and adjust prompts/allowed tables.
6. In **Playground**, choose:
   - **Multi-agent manager** to let the manager route tasks across agents dynamically.
   - **Single agent** to run one selected agent directly.

## Security notes

- DB credentials are stored locally as JSON in `backend/data`.
- `settings.json` and `agents.json` are git-ignored.
- `sql_analyst` executes `SELECT` queries only.
- `clickhouse_table_manager` can execute DDL/DML on ClickHouse with configurable safety guards (enabled by default).
