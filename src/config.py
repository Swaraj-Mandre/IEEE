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
CHUNK_DIR = _path("CHUNK_DIR", "data/chunks")

# Chapters the graph and the index are built from. Add a chunk file here to
# widen coverage; the graph build and the index build both read this list.
CHUNK_FILES = [
    CHUNK_DIR / "ch04_describing_motion_chunks.json",
    CHUNK_DIR / "ch06_forces_and_motion_chunks.json",
]

# The longest prompt either agent builds is 639 tokens, so 2048 leaves 3x room.
# The window is pure KV cache: Phi-3's full 4096 costs about 800 MB more RAM.
N_CTX = int(os.getenv("N_CTX", 2048))
N_THREADS = int(os.getenv("N_THREADS", 4))
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", 0))
