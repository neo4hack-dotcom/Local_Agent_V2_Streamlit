#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="$ROOT_DIR/backend/.venv"

if [ ! -d "$VENV_PATH" ]; then
  echo "Backend virtualenv not found. Run 'npm run setup' first."
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"
cd "$ROOT_DIR/backend"
exec uvicorn app.main:app --reload --port 8000
