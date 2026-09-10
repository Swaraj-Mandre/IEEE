"""Flat vector index over the textbook chunks.

The corpus is a few hundred chunks, so an exact brute-force search is both
faster and more accurate than an approximate index, and the store is two plain
files that survive any library upgrade.
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src import config

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# BGE models are trained with this prefix on queries only, never on documents.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

EMBEDDINGS_FILE = "embeddings.npy"
CHUNKS_FILE = "chunks.json"


@dataclass
class VectorStore:
    """Chunk embeddings and their metadata, row-aligned, plus a keyword index."""
    embeddings: np.ndarray
    chunks: list[dict]
    bm25: BM25Okapi = field(init=False, repr=False)
    # Filled in by the retriever on first use; see concept_frequencies().
    concept_df: dict[str, int] | None = field(default=None, repr=False)

    def __post_init__(self):
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

    def count(self) -> int:
        return len(self.chunks)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Keeps digits so "9.8" and "1 kg" stay searchable."""
    return re.findall(r"[a-z0-9]+", text.lower())


def get_embedding_model() -> SentenceTransformer:
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Embedding model loaded.\n")
    return model


def embed(model: SentenceTransformer, texts: list[str], is_query: bool = False) -> np.ndarray:
    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]
    return model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,  # lets cosine similarity be a plain dot product
    ).astype(np.float32)


def build_vector_store(chunk_files=None, force_rebuild: bool = False) -> VectorStore:
    """Embed every chunk and write the index to disk."""
    if not force_rebuild and (config.VECTORSTORE_PATH / EMBEDDINGS_FILE).exists():
        store = get_collection()
        print(f"Vector store already has {store.count()} chunks. Use force_rebuild=True to regenerate.")
        return store

    all_chunks = []
    for chunk_file in chunk_files or config.CHUNK_FILES:
        path = Path(chunk_file)
        if not path.is_absolute():
            path = config.PROJECT_ROOT / path
        if not path.exists():
            print(f"WARNING: {path.name} not found, skipping.")
            continue
        chunks = json.loads(path.read_text(encoding="utf-8"))
        all_chunks.extend(chunks)
        print(f"Loaded {len(chunks)} chunks from {path.name}")

    if not all_chunks:
        raise RuntimeError("No chunks found. Run scripts/run_ingestion.py first.")

    print(f"\nTotal chunks to embed: {len(all_chunks)}")
    model = get_embedding_model()
    embeddings = embed(model, [c["text"] for c in all_chunks])

    config.VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)
    np.save(config.VECTORSTORE_PATH / EMBEDDINGS_FILE, embeddings)
    (config.VECTORSTORE_PATH / CHUNKS_FILE).write_text(
        json.dumps(all_chunks, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nVector store built: {len(all_chunks)} chunks, {embeddings.nbytes / 1e6:.1f} MB")
    print(f"Stored at: {config.VECTORSTORE_PATH}")
    return VectorStore(embeddings, all_chunks)


def get_collection() -> VectorStore:
    """Load the index for querying. Does not rebuild."""
    embeddings_path = config.VECTORSTORE_PATH / EMBEDDINGS_FILE
    chunks_path = config.VECTORSTORE_PATH / CHUNKS_FILE

    if not embeddings_path.exists() or not chunks_path.exists():
        raise RuntimeError(
            "Vector store not found. Run scripts/build_vectorstore.py to build it."
        )

    return VectorStore(
        embeddings=np.load(embeddings_path),
        chunks=json.loads(chunks_path.read_text(encoding="utf-8")),
    )


def query_vector_store(
    store: VectorStore,
    embed_model: SentenceTransformer,
    query: str,
    n_results: int = 5,
) -> list[dict]:
    """Return the n most similar chunks, scored by cosine similarity."""
    query_vec = embed(embed_model, [query], is_query=True)[0]
    scores = store.embeddings @ query_vec

    return _top_hits(store, scores, n_results)


def keyword_search(store: VectorStore, query: str, n_results: int = 5) -> list[dict]:
    """Return the n best chunks by BM25. Catches exact terms like "F = ma" that embeddings blur."""
    scores = np.asarray(store.bm25.get_scores(tokenize(query)), dtype=np.float32)
    return _top_hits(store, scores, n_results)


def _top_hits(store: VectorStore, scores: np.ndarray, n_results: int) -> list[dict]:
    n_results = min(n_results, len(scores))
    top = np.argpartition(-scores, n_results - 1)[:n_results]
    top = top[np.argsort(-scores[top])]

    return [
        {
            "chunk_id": store.chunks[i]["chunk_id"],
            "text": store.chunks[i]["text"],
            "source_file": store.chunks[i]["source_file"],
            "page_num": store.chunks[i]["page_num"],
            "section": store.chunks[i].get("section"),
            "similarity": round(float(scores[i]), 4),
        }
        for i in top
    ]
