import pickle
import re
import networkx as nx
from sentence_transformers import SentenceTransformer
from src.retrieval.vector_store import (
    get_collection,
    query_vector_store,
    get_embedding_model,
)


def load_graph(graph_path: str = "data/graph/kg.pkl") -> nx.DiGraph:
    """Load the knowledge graph from disk."""
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    return G


def extract_concepts_from_chunks(
    chunks: list[dict],
    G: nx.DiGraph,
) -> list[tuple[str, float]]:
    """
    Given retrieved chunks, find which graph nodes they mention.
    Returns list of (concept, relevance_score) tuples.

    Strategy: check if any graph node name appears in the chunk text.
    Simple string matching — fast and sufficient for our graph size.
    """
    concept_scores = {}
    graph_nodes = set(G.nodes())

    for chunk in chunks:
        text_lower = chunk["text"].lower()
        similarity = chunk["similarity"]

        for node in graph_nodes:
            # Skip garbage nodes
            if node in ("none", "n/a", "") or len(node) < 3:
                continue

            # Check if concept appears in chunk text
            if node in text_lower:
                if node not in concept_scores:
                    concept_scores[node] = 0
                # Weight by similarity score of the chunk
                concept_scores[node] += similarity

    # Sort by accumulated score
    ranked = sorted(
        concept_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked


def reciprocal_rank_fusion(
    graph_results: list[tuple[str, float]],
    vector_chunks: list[dict],
    G: nx.DiGraph,
    k: int = 60,
) -> list[dict]:
    """
    Fuse graph-based concept ranking with vector search results.

    Reciprocal Rank Fusion formula: score = 1 / (k + rank)
    Higher score = more relevant.

    This is the core of GraphRAG — neither source alone is sufficient:
    - Graph alone: finds structurally connected concepts but may miss
      semantically relevant ones not well-connected in the graph.
    - Vector alone: finds semantically similar chunks but ignores
      conceptual structure and prerequisite order.
    - Fusion: balances both signals.
    """
    fused_scores = {}

    # Score from graph-based concept matching
    for rank, (concept, score) in enumerate(graph_results[:20]):
        fused_scores[concept] = fused_scores.get(concept, 0)
        fused_scores[concept] += 1 / (k + rank + 1)

    # Score from vector search — extract concepts mentioned in top chunks
    vector_concepts = extract_concepts_from_chunks(vector_chunks, G)
    for rank, (concept, score) in enumerate(vector_concepts[:20]):
        fused_scores[concept] = fused_scores.get(concept, 0)
        fused_scores[concept] += 1 / (k + rank + 1)

    # Sort by fused score
    ranked = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    # Build final result with metadata
    results = []
    for concept, score in ranked[:10]:
        results.append({
            "concept": concept,
            "fused_score": round(score, 6),
            "in_degree": G.in_degree(concept),   # How many prerequisites
            "out_degree": G.out_degree(concept),  # How many depend on it
        })

    return results


def retrieve(
    question: str,
    G: nx.DiGraph,
    collection,
    embed_model: SentenceTransformer,
    n_vector_results: int = 5,
) -> dict:
    """
    Main retrieval function. Given a student question, returns:
    - top_concept: best matching concept in the graph
    - fused_results: ranked list of relevant concepts
    - supporting_chunks: text chunks that support the answer
    - query: original question

    This is called by the Instructor Agent.
    """
    # Vector search
    vector_chunks = query_vector_store(
        collection, embed_model, question, n_results=n_vector_results
    )

    # Extract concepts mentioned in retrieved chunks
    graph_results = extract_concepts_from_chunks(vector_chunks, G)

    # Fuse rankings
    fused_results = reciprocal_rank_fusion(graph_results, vector_chunks, G)

    # Select top concept
    top_concept = fused_results[0]["concept"] if fused_results else None

    return {
        "query": question,
        "top_concept": top_concept,
        "fused_results": fused_results,
        "supporting_chunks": vector_chunks,
    }