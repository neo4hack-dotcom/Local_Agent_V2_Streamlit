#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

printf "\n[1/3] Installing frontend dependencies...\n"
npm --prefix "$ROOT_DIR/frontend" install

printf "\n[2/3] Creating backend virtualenv and installing backend dependencies...\n"
python3 -m venv "$ROOT_DIR/backend/.venv"
# shellcheck disable=SC1091
source "$ROOT_DIR/backend/.venv/bin/activate"
pip install -U pip
pip install -e "$ROOT_DIR/backend"

printf "\n[3/3] Installing Playwright Chromium browser binaries (for Web Navigator agent)...\n"
if ! python -m playwright install chromium; then
  printf "\nWarning: Playwright Chromium installation failed.\n"
  printf "Web Navigator agent may be unavailable until you run manually in backend venv:\n"
  printf "  python -m playwright install chromium\n"
fi

printf "\nSetup complete.\n"
printf "Run 'npm run start:all' from project root to start frontend + backend.\n"
