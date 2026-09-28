"""Project-local paths used by scripts and agents."""

import os
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
RESULTS_ROOT = Path(os.environ.get("CO_PI_TREE_RESULTS_ROOT", PROJECT_ROOT)).expanduser()

PROMPTS_DIR = SRC_DIR / "prompts"
POLICY_DIR = SRC_DIR / "policy"
REPLAY_DIR = SRC_DIR / "rendered_replays"
STATE_LOG_PATH = PROJECT_ROOT / "state_log.txt"
MAIN_PY = SRC_DIR / "main.py"
MODELS_DIR = SRC_DIR / "models"
LOCAL_LIB_DIRS = (
    PROJECT_ROOT / "lib" / "overcooked_ai",
    PROJECT_ROOT / "lib" / "stable-baselines",
)


def ensure_local_libs() -> None:
    """Prefer the bundled local libraries over globally installed editable copies."""
    for lib_dir in reversed(LOCAL_LIB_DIRS):
        if not lib_dir.exists():
            continue
        lib_path = str(lib_dir)
        while lib_path in sys.path:
            sys.path.remove(lib_path)
        sys.path.insert(0, lib_path)


def results_dir(name: str) -> Path:
    return RESULTS_ROOT / name


ensure_local_libs()
