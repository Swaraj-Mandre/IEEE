"""Smoke test for the tutoring loop: retrieve a concept, explain it, then route on a reply.

Mirrors what src/ui/app.py does, so a pass here means the app path works.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.instructor import load_model, generate_explanation
from src.agents.diagnostic import (
    analyse_response,
    compute_gap_score,
    should_advance,
    should_backtrack,
)
from src.retrieval.path_tracker import (
    advance,
    backtrack,
    create_session,
    get_current_concept,
    save_session_checkpoint,
)
from src.retrieval.retriever import load_graph, retrieve
from src.retrieval.vector_store import get_collection, get_embedding_model

TEST_REPLIES = [
    "I don't understand what you mean",
    "Got it, makes sense now",
    "Can you explain again?",
    "I know this already",
]


def run_agent_test():
    print("=" * 60)
    print("VIDHYA-SETU : AGENT SYSTEM TEST")
    print("=" * 60)

    model = load_model()
    graph = load_graph()
    collection = get_collection()
    embed_model = get_embedding_model()

    result = retrieve("What is acceleration?", graph, collection, embed_model)
    target = result["top_concept"]
    chunks = result["supporting_chunks"]
    print(f"\nRetrieved target concept: {target}")

    session = create_session("test_student", target, graph)

    print("\n--- INSTRUCTOR AGENT ---")
    concept = get_current_concept(session)
    print(f"Teaching: {concept}\n")
    print(generate_explanation(model, concept, chunks, level="normal"))

    print("\n--- DIAGNOSTIC AGENT ---")
    for reply in TEST_REPLIES:
        diagnosis = analyse_response(reply, concept)
        gap = compute_gap_score(
            len(session.mastered_concepts), len(session.prerequisite_chain)
        )
        print(f"\nReply:    {reply}")
        print(f"Verdict:  {diagnosis['verdict']} | gap {gap}")
        print(f"Routes to: {_route(diagnosis, gap)}")

    print("\n--- ROUTING ---")
    confused = analyse_response("I don't understand this at all", concept)
    gap = compute_gap_score(len(session.mastered_concepts), len(session.prerequisite_chain))
    if should_backtrack(confused, gap):
        backtrack(session)
    print(f"After confusion: {get_current_concept(session)}")

    understood = analyse_response("got it, makes sense", concept)
    if should_advance(understood):
        advance(session)
    print(f"After understanding: {get_current_concept(session)}")

    save_session_checkpoint(session)
    print(f"\nPASSED. Checkpoint written for '{session.student_id}'.")


def _route(diagnosis: dict, gap: float) -> str:
    if should_backtrack(diagnosis, gap):
        return "BACKTRACK"
    if should_advance(diagnosis):
        return "ADVANCE"
    return "RE-EXPLAIN"


if __name__ == "__main__":
    run_agent_test()
