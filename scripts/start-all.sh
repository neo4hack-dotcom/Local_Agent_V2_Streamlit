#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -d "$ROOT_DIR/backend/.venv" ]; then
  echo "Backend virtualenv not found. Run 'npm run setup' first."
  exit 1
fi

cleanup() {
  jobs -p | xargs -r kill >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

(
  cd "$ROOT_DIR/frontend"
  npm run dev
) &

(
  "$ROOT_DIR/scripts/dev-backend.sh"
) &

wait
