from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = BACKEND_ROOT / "data"
SETTINGS_FILE = DATA_DIR / "settings.json"
AGENTS_FILE = DATA_DIR / "agents.json"

DEFAULT_FRONTEND_ORIGIN = "http://localhost:5173"
