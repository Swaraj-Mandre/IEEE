import json
import pickle
from pathlib import Path
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END

from src.agents.instructor import generate_explanation
from src.agents.diagnostic import (
    analyse_response,
    compute_gap_score,
    should_backtrack,
    should_advance,
    should_simplify,
)
from src.retrieval.path_tracker import (
    LearningSession,
    get_current_concept,
    advance,
    backtrack,
    session_to_dict,
    session_from_dict,
)
from src.retrieval.retriever import retrieve
from src.retrieval.vector_store import get_collection, get_embedding_model


# Shared state passed between all nodes in the LangGraph state machine
class AgentState(TypedDict):
    session: dict                    # LearningSession serialized as dict
    student_input: str               # Latest message from student
    current_explanation: str         # Last explanation from Instructor Agent
    diagnostic_result: dict          # Latest result from Diagnostic Agent
    gap_score: float                 # Current gap score
    explanation_level: str           # simple or normal
    supporting_chunks: list          # Retrieved chunks for current concept
    action: str                      # next action: explain, backtrack, advance, end


def instructor_node(state: AgentState, model) -> AgentState:
    session = session_from_dict(state["session"])
    concept = get_current_concept(session)
    level = state.get("explanation_level", "normal")
    chunks = state.get("supporting_chunks", [])

    print("\n[Instructor] Teaching: " + concept + " | Level: " + level)

    explanation = generate_explanation(model, concept, chunks, level)
    session.exchange_count += 1

    save_session_checkpoint(session)

    return {
        **state,
        "session": session_to_dict(session),
        "current_explanation": explanation,
        "action": "wait_for_student",
    }


def diagnostic_node(state: AgentState) -> AgentState:
    session = session_from_dict(state["session"])
    student_reply = state["student_input"]
    concept = get_current_concept(session)

    mastered = len(session.mastered_concepts)
    total = len(session.prerequisite_chain)
    gap_score = compute_gap_score(mastered, total)

    result = analyse_response(student_reply, concept)

    print("\n[Diagnostic] Verdict: " + result["verdict"] + " | Gap score: " + str(gap_score))

    if should_simplify(result, session.backtrack_count):
        action = "simplify"
    elif should_backtrack(result, gap_score):
        action = "backtrack"
    elif should_advance(result):
        action = "advance"
    else:
        action = "re_explain"

    return {
        **state,
        "session": session_to_dict(session),
        "diagnostic_result": result,
        "gap_score": gap_score,
        "action": action,
    }


def backtrack_node(state: AgentState) -> AgentState:
    session = session_from_dict(state["session"])
    backtrack(session)
    save_session_checkpoint(session)
    return {
        **state,
        "session": session_to_dict(session),
        "explanation_level": "simple",
        "action": "explain",
    }


def advance_node(state: AgentState) -> AgentState:
    session = session_from_dict(state["session"])
    new_concept = advance(session)
    save_session_checkpoint(session)

    is_complete = new_concept == session.target_concept and len(session.mastered_concepts) == len(session.prerequisite_chain) - 1

    return {
        **state,
        "session": session_to_dict(session),
        "explanation_level": "normal",
        "action": "end" if is_complete else "explain",
    }


def route_after_diagnostic(state: AgentState) -> str:
    action = state["action"]
    if action in ("backtrack", "simplify"):
        return "backtrack"
    elif action == "advance":
        return "advance"
    else:
        return "instructor"


def save_session_checkpoint(session: LearningSession):
    Path("data/sessions").mkdir(parents=True, exist_ok=True)
    path = "data/sessions/" + session.student_id + "_session.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session_to_dict(session), f, indent=2, ensure_ascii=False)


def build_graph(model):
    graph = StateGraph(AgentState)

    graph.add_node("instructor", lambda state: instructor_node(state, model))
    graph.add_node("diagnostic", diagnostic_node)
    graph.add_node("backtrack", backtrack_node)
    graph.add_node("advance", advance_node)

    graph.set_entry_point("instructor")

    graph.add_edge("instructor", END)
    graph.add_edge("backtrack", "instructor")
    graph.add_edge("advance", "instructor")

    graph.add_conditional_edges(
        "diagnostic",
        route_after_diagnostic,
        {
            "instructor": "instructor",
            "backtrack": "backtrack",
            "advance": "advance",
        }
    )

    return graph.compile()