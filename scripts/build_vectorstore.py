"""Build the search index from the chunk files listed in src/config.py.

Run once after ingestion, and again whenever the chunks or the chapter list change.
The index is not tracked in git, so a fresh clone needs this before the tutor runs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.vector_store import build_vector_store


def main() -> None:
    store = build_vector_store(force_rebuild=True)
    print(f"\nDone. {store.count()} chunks indexed.")
    print("Next: python scripts/run_checks.py")


if __name__ == "__main__":
    main()
