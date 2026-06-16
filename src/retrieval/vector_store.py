import os
import json
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "data/vectorstore")
COLLECTION_NAME = "vidhya_setu_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embedding_model() -> SentenceTransformer:
    """
    Load sentence-transformers model for chunk embedding.
    22MB model, downloads once and caches locally.
    CPU inference is fast enough — no GPU needed for embeddings.
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Embedding model loaded.\n")
    return model


def get_chroma_client() -> chromadb.PersistentClient:
    """
    Get ChromaDB persistent client pointed at D drive.
    PersistentClient saves to disk — survives restarts.
    """
    Path(VECTORSTORE_PATH).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=VECTORSTORE_PATH)
    return client


def build_vector_store(
    chunk_files: list[str],
    force_rebuild: bool = False
) -> chromadb.Collection:
    """
    Embed all chunks and store in ChromaDB.

    Args:
        chunk_files: List of paths to chunk JSON files
        force_rebuild: If True, delete existing collection and rebuild

    Returns:
        ChromaDB collection ready for querying
    """
    client = get_chroma_client()

    # Handle force rebuild
    if force_rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("Deleted existing collection for rebuild.")
        except Exception:
            pass

    # Check if collection already exists with data
    try:
        collection = client.get_collection(COLLECTION_NAME)
        existing_count = collection.count()
        if existing_count > 0 and not force_rebuild:
            print(f"Vector store already has {existing_count} chunks.")
            print("Skipping rebuild. Use force_rebuild=True to regenerate.")
            return collection
    except Exception:
        pass

    # Create fresh collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # Cosine similarity for text
    )

    # Load embedding model
    embed_model = get_embedding_model()

    # Load and embed all chunks
    all_chunks = []
    for chunk_file in chunk_files:
        if not Path(chunk_file).exists():
            print(f"WARNING: {chunk_file} not found, skipping.")
            continue
        with open(chunk_file, encoding="utf-8") as f:
            chunks = json.load(f)
        all_chunks.extend(chunks)
        print(f"Loaded {len(chunks)} chunks from {chunk_file}")

    print(f"\nTotal chunks to embed: {len(all_chunks)}")
    print("Embedding chunks (this takes ~2 minutes)...")

    # Process in batches to avoid memory issues
    # ChromaDB has a 41665 item limit per add() call
    batch_size = 100
    total_batches = (len(all_chunks) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(all_chunks))
        batch = all_chunks[start:end]

        # Extract texts for embedding
        texts = [c["text"] for c in batch]

        # Generate embeddings
        embeddings = embed_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,  # Normalize for cosine similarity
        ).tolist()

        # Prepare ChromaDB inputs
        ids = [c["chunk_id"] for c in batch]
        metadatas = [
            {
                "page_num": c["page_num"],
                "source_file": c["source_file"],
                "chunk_index": c["chunk_index"],
                "estimated_tokens": c["estimated_tokens"],
            }
            for c in batch
        ]

        # Add to ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        if (batch_idx + 1) % 5 == 0 or batch_idx == total_batches - 1:
            print(f"  Embedded {end}/{len(all_chunks)} chunks")

    final_count = collection.count()
    print(f"\nVector store built: {final_count} chunks indexed.")
    print(f"Stored at: {VECTORSTORE_PATH}")
    return collection


def get_collection() -> chromadb.Collection:
    """
    Get existing ChromaDB collection for querying.
    Call this in retriever — does not rebuild.
    """
    client = get_chroma_client()
    try:
        collection = client.get_collection(COLLECTION_NAME)
        return collection
    except Exception:
        raise RuntimeError(
            "Vector store not found. Run scripts/run_retrieval_test.py first "
            "to build the vector store."
        )


def query_vector_store(
    collection: chromadb.Collection,
    embed_model: SentenceTransformer,
    query: str,
    n_results: int = 5,
) -> list[dict]:
    """
    Query vector store for semantically similar chunks.

    Returns:
        List of dicts with text, metadata, and similarity score
    """
    # Embed the query
    query_embedding = embed_model.encode(
        [query],
        normalize_embeddings=True,
    ).tolist()[0]

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    # Format results
    formatted = []
    for i in range(len(results["ids"][0])):
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity: 1 - (distance/2)
        distance = results["distances"][0][i]
        similarity = round(1 - (distance / 2), 4)

        formatted.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "source_file": results["metadatas"][0][i]["source_file"],
            "page_num": results["metadatas"][0][i]["page_num"],
            "similarity": similarity,
        })

    return formatted