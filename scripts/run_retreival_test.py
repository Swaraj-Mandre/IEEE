import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.vector_store import build_vector_store, get_embedding_model
from src.retrieval.retriever import load_graph, retrieve
from src.retrieval.path_tracker import (
    create_session,
    get_current_concept,
    advance,
    backtrack,
    session_to_dict,
)

CHUNK_FILES = [
    "data/chunks/ch04_describing_motion_chunks.json",
    "data/chunks/ch06_forces_and_motion_chunks.json",
]

TEST_QUESTIONS = [
    "What is acceleration?",
    "What is the difference between speed and velocity?",
    "How does Newton second law work?",
    "What is uniform motion?",
    "What is inertia?",
]


def run_retrieval_test():

    # Step 1: Build vector store
    print("Building vector store...")
    collection = build_vector_store(CHUNK_FILES, force_rebuild=True)

    # Step 2: Load graph and embedding model
    print("Loading graph...")
    G = load_graph()
    embed_model = get_embedding_model()
    print("Nodes: " + str(G.number_of_nodes()))
    print("Edges: " + str(G.number_of_edges()))

    # Step 3: Test retrieval for each question
    print("\n--- RETRIEVAL TEST ---")
    for question in TEST_QUESTIONS:
        print("\nQuestion: " + question)
        result = retrieve(question, G, collection, embed_model)
        print("Top concept: " + str(result["top_concept"]))
        print("Top 3 results:")
        for r in result["fused_results"][:3]:
            print("  " + r["concept"] + " | score: " + str(round(r["fused_score"], 4)))
        if result["supporting_chunks"]:
            chunk = result["supporting_chunks"][0]
            print("Best chunk: " + chunk["text"][:150])

    # Step 4: Test path tracker
    print("\n--- PATH TRACKER TEST ---")
    candidates = [n for n in G.nodes() if "newton" in n.lower()]
    target = candidates[0] if candidates else "acceleration"
    print("Target concept: " + target)

    session = create_session("student01", target, G)
    print("Prerequisite chain: " + str(session.prerequisite_chain))
    print("Current concept: " + get_current_concept(session))

    print("Simulating confusion - backtracking...")
    backtrack(session)
    print("After backtrack: " + get_current_concept(session))

    print("Student understood - advancing...")
    advance(session)
    print("After advance: " + get_current_concept(session))

    # Save session
    Path("data/sessions").mkdir(parents=True, exist_ok=True)
    with open("data/sessions/test_session.json", "w", encoding="utf-8") as f:
        json.dump(session_to_dict(session), f, indent=2)

    print("\nSession saved to data/sessions/test_session.json")
    print("\nPHASE 3 COMPLETE")


if __name__ == "__main__":
    run_retrieval_test()