import sys
import json
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.instructor import load_model, generate_explanation
from src.agents.diagnostic import analyse_response, compute_gap_score, should_backtrack, should_advance
from src.agents.orchestrator import build_graph, save_session_checkpoint
from src.retrieval.path_tracker import create_session, get_current_concept, session_to_dict
from src.retrieval.retriever import load_graph, retrieve
from src.retrieval.vector_store import get_collection, get_embedding_model


def run_agent_test():
    print("=" * 60)
    print("VIDHYA-SETU : AGENT SYSTEM TEST")
    print("=" * 60)

    print("\nLoading model...")
    model = load_model()

    print("Loading graph and retrieval components...")
    G = load_graph()
    collection = get_collection()
    embed_model = get_embedding_model()

    target = "acceleration"
    candidates = [n for n in G.nodes() if "newton" in n.lower()]
    if candidates:
        target = candidates[0]

    print("Target concept: " + target)
    session = create_session("test_student", target, G)

    result = retrieve(target, G, collection, embed_model)
    chunks = result["supporting_chunks"]

    print("\n--- INSTRUCTOR AGENT TEST ---")
    concept = get_current_concept(session)
    print("Teaching concept: " + concept)
    explanation = generate_explanation(model, concept, chunks, level="normal")
    print("\nExplanation:")
    print(explanation)

    print("\n--- DIAGNOSTIC AGENT TEST ---")
    test_replies = [
        "I don't understand what you mean",
        "Got it, makes sense now",
        "Can you explain again?",
        "Yes I understand",
    ]

    for reply in test_replies:
        result_diag = analyse_response(reply, concept)
        gap = compute_gap_score(
            len(session.mastered_concepts),
            len(session.prerequisite_chain)
        )
        bt = should_backtrack(result_diag, gap)
        adv = should_advance(result_diag)
        print("\nReply: " + reply)
        print("Verdict: " + result_diag["verdict"])
        print("Gap score: " + str(gap))
        print("Backtrack: " + str(bt) + " | Advance: " + str(adv))

    print("\n--- FULL PIPELINE TEST ---")
    print("Building LangGraph state machine...")
    app = build_graph(model)

    initial_state = {
        "session": session_to_dict(session),
        "student_input": "",
        "current_explanation": "",
        "diagnostic_result": {},
        "gap_score": 1.0,
        "explanation_level": "normal",
        "supporting_chunks": chunks,
        "action": "explain",
    }

    print("Running instructor node...")
    output = app.invoke(initial_state)
    print("\nExplanation from pipeline:")
    print(output["current_explanation"])

    print("\nSimulating student confusion...")
    output["student_input"] = "I don't understand this at all"
    output2 = app.invoke({
        **output,
        "action": "diagnose",
    })

    print("\nPHASE COMPLETED")
    print("Check data/sessions/ for session checkpoint")


if __name__ == "__main__":
    run_agent_test()