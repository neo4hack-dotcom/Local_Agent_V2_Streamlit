# Backend API

FastAPI backend for the Local Agent Studio application.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
python -m playwright install chromium
uvicorn app.main:app --reload --port 8000
```

## Environment variables

- `FRONTEND_ORIGIN` (default: `http://localhost:5173`)

## Main endpoints

- `GET /health`
- `GET /api/config`
- `PUT /api/config/llm`
- `POST /api/config/llm/test`
- `GET /api/config/llm/models`
- `GET/POST/PUT/DELETE /api/databases`
- `PUT /api/databases/active/{database_id}`
- `POST /api/databases/{database_id}/test`
- `GET/POST/PUT/DELETE /api/agents`
- `POST /api/agents/{agent_id}/run`
- `POST /api/agents/manager/run`
- `POST /api/agents/manager/run/stream` (NDJSON stream)
