import math
import pickle

import networkx as nx
from sentence_transformers import SentenceTransformer

from src import config
from src.retrieval.vector_store import (
    VectorStore,
    keyword_search,
    query_vector_store,
    tokenize,
)

RRF_K = 60
# How much to favour a concept the student named in their own question.
QUESTION_MATCH_BOOST = 3.0


def load_graph(graph_path=None) -> nx.DiGraph:
    """Load the knowledge graph from disk."""
    with open(graph_path or config.GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    return G


def reciprocal_rank_fusion(*rankings: list[dict], k: int = RRF_K) -> list[dict]:
    """
    Merge several chunk rankings into one.

    RRF scores by position rather than by score, because BM25 and cosine
    similarity are on scales that cannot be compared directly.
    """
    scores: dict[str, float] = {}
    by_id: dict[str, dict] = {}

    for ranking in rankings:
        for rank, chunk in enumerate(ranking):
            chunk_id = chunk["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank + 1)
            by_id.setdefault(chunk_id, chunk)

    ordered = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return [{**by_id[cid], "fused_score": round(score, 6)} for cid, score in ordered]


def concept_frequencies(store: VectorStore, G: nx.DiGraph) -> dict[str, int]:
    """Count how many chunks mention each concept. Computed once, then cached on the store."""
    if store.concept_df is None:
        nodes = [n for n in G.nodes() if len(n) >= 3]
        counts = dict.fromkeys(nodes, 0)
        for chunk in store.chunks:
            text = chunk["text"].lower()
            for node in nodes:
                if node in text:
                    counts[node] += 1
        store.concept_df = counts
    return store.concept_df


def extract_concepts_from_chunks(
    chunks: list[dict],
    G: nx.DiGraph,
    concept_df: dict[str, int] | None = None,
    total_chunks: int = 0,
    question: str = "",
) -> list[tuple[str, float]]:
    """
    Find which graph concepts the retrieved chunks talk about.

    Three signals decide the winner:
      - how strongly the top chunks mention it, longest match only, so a chunk
        about "newton's second law of motion" does not also credit "motion"
      - how rare it is in the corpus, or "object" wins every time
      - whether the student's own words name it
    """
    nodes = sorted(G.nodes(), key=len, reverse=True)
    concept_scores: dict[str, float] = {}

    for rank, chunk in enumerate(chunks):
        text_lower = chunk["text"].lower()
        weight = 1 / (RRF_K + rank + 1)
        matched: list[str] = []

        for node in nodes:
            if len(node) < 3 or node not in text_lower:
                continue
            if any(node in longer for longer in matched):
                continue
            matched.append(node)
            concept_scores[node] = concept_scores.get(node, 0.0) + weight

    question_words = set(tokenize(question))
    for node, score in concept_scores.items():
        if concept_df and total_chunks:
            df = concept_df.get(node, 1) or 1
            score *= math.log(1 + total_chunks / df)
        if question_words:
            node_words = tokenize(node)
            overlap = sum(w in question_words for w in node_words) / len(node_words)
            score *= 1 + QUESTION_MATCH_BOOST * overlap
        concept_scores[node] = score

    # Ties break on concept name so the same question always gives the same answer.
    return sorted(concept_scores.items(), key=lambda x: (-x[1], x[0]))


def retrieve(
    question: str,
    G: nx.DiGraph,
    store: VectorStore,
    embed_model: SentenceTransformer,
    n_results: int = 5,
) -> dict:
    """
    Answer a student question with a starting concept and the text to explain it from.

    Runs two independent searches - dense embeddings for meaning, BM25 for exact
    wording - fuses them, then resolves the result against the knowledge graph.
    """
    dense_hits = query_vector_store(store, embed_model, question, n_results=n_results)
    keyword_hits = keyword_search(store, question, n_results=n_results)
    fused_chunks = reciprocal_rank_fusion(dense_hits, keyword_hits)[:n_results]

    ranked_concepts = extract_concepts_from_chunks(
        fused_chunks, G, concept_frequencies(store, G), store.count(), question
    )

    results = [
        {
            "concept": concept,
            "score": round(score, 6),
            "in_degree": G.in_degree(concept),
            "out_degree": G.out_degree(concept),
        }
        for concept, score in ranked_concepts[:10]
    ]

    return {
        "query": question,
        "top_concept": results[0]["concept"] if results else None,
        "fused_results": results,
        "supporting_chunks": fused_chunks,
    }
