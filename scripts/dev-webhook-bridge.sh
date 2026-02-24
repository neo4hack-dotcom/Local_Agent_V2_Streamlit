#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="$ROOT_DIR/backend/.venv"

if [ ! -d "$VENV_PATH" ]; then
  echo "Backend virtualenv not found. Run 'npm run setup' first."
  exit 1
fi

if [ -z "${OPEN_WEBUI_WEBHOOK_URL:-}" ]; then
  echo "OPEN_WEBUI_WEBHOOK_URL is not set."
  echo "Example:"
  echo "  export OPEN_WEBUI_WEBHOOK_URL=\"https://openwebui.example/api/v1/channels/<id>/webhook/<token>\""
  exit 1
fi

BRIDGE_HOST="${OPEN_WEBUI_BRIDGE_HOST:-0.0.0.0}"
BRIDGE_PORT="${OPEN_WEBUI_BRIDGE_PORT:-8010}"

# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"
cd "$ROOT_DIR/backend"
exec uvicorn app.open_webui_bridge:app --reload --host "$BRIDGE_HOST" --port "$BRIDGE_PORT"
