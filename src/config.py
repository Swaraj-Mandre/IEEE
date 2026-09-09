"""Single source of truth for paths and model settings, read from .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _path(var: str, default: str) -> Path:
    """Resolve a configured path against the project root so scripts run from any directory."""
    value = Path(os.getenv(var) or default)
    return value if value.is_absolute() else PROJECT_ROOT / value


MODEL_PATH = _path("MODEL_PATH", "models/Phi-3-mini-4k-instruct-q4.gguf")
GRAPH_PATH = _path("GRAPH_PATH", "data/graph/kg.pkl")
VECTORSTORE_PATH = _path("VECTORSTORE_PATH", "data/vectorstore")
SESSION_DIR = _path("SESSION_DIR", "data/sessions")

# Phi-3 Mini was trained at 4096 tokens; anything lower throws away context.
N_CTX = int(os.getenv("N_CTX", 4096))
N_THREADS = int(os.getenv("N_THREADS", 4))
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", 0))
