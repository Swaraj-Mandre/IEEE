import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gradio as gr

from src.agents.instructor import load_model, generate_explanation
from src.agents.diagnostic import analyse_response, compute_gap_score, should_backtrack, should_advance
from src.retrieval.path_tracker import (
    create_session,
    get_current_concept,
    advance,
    backtrack,
    session_to_dict,
)
from src.retrieval.retriever import load_graph, retrieve
from src.retrieval.vector_store import get_collection, get_embedding_model


print("Loading model and components, please wait...")
MODEL = load_model()
GRAPH = load_graph()
COLLECTION = get_collection()
EMBED_MODEL = get_embedding_model()
print("Ready.")

SESSION = {"obj": None, "chunks": []}


def start_session(topic):
    if not topic.strip():
        return "Please enter a topic to learn about.", "", ""

    result = retrieve(topic, GRAPH, COLLECTION, EMBED_MODEL)
    target = result["top_concept"]

    if not target:
        return "Could not find a matching concept. Try a different topic.", "", ""

    session = create_session("ui_student", target, GRAPH)
    SESSION["obj"] = session
    SESSION["chunks"] = result["supporting_chunks"]

    concept = get_current_concept(session)
    explanation = generate_explanation(MODEL, concept, SESSION["chunks"], level="normal")

    chain_text = " -> ".join(session.prerequisite_chain)

    Path("data/sessions").mkdir(parents=True, exist_ok=True)
    with open("data/sessions/ui_student_session.json", "w", encoding="utf-8") as f:
        json.dump(session_to_dict(session), f, indent=2, ensure_ascii=False)

    return explanation, chain_text, concept


def respond_to_student(student_reply):
    session = SESSION["obj"]
    if session is None:
        return "Please start a session first by entering a topic above.", "", ""

    concept = get_current_concept(session)
    result = analyse_response(student_reply, concept)
    mastered = len(session.mastered_concepts)
    total = len(session.prerequisite_chain)
    gap = compute_gap_score(mastered, total)

    status_line = "Diagnostic: " + result["verdict"] + " | Gap score: " + str(gap)

    if should_backtrack(result, gap):
        backtrack(session)
        new_concept = get_current_concept(session)
        explanation = generate_explanation(MODEL, new_concept, SESSION["chunks"], level="simple")
        status_line += " | Action: BACKTRACK to simpler concept"

    elif should_advance(result):
        new_concept = advance(session)
        if new_concept == session.target_concept and mastered >= total - 1:
            explanation = "Great work! You have completed the learning path for: " + session.target_concept
            status_line += " | Action: SESSION COMPLETE"
        else:
            explanation = generate_explanation(MODEL, new_concept, SESSION["chunks"], level="normal")
            status_line += " | Action: ADVANCE to next concept"

    else:
        new_concept = concept
        explanation = generate_explanation(MODEL, new_concept, SESSION["chunks"], level="normal")
        status_line += " | Action: RE-EXPLAIN same concept"

    with open("data/sessions/ui_student_session.json", "w", encoding="utf-8") as f:
        json.dump(session_to_dict(session), f, indent=2, ensure_ascii=False)

    return explanation, status_line, new_concept


with gr.Blocks(title="Vidhya-Setu") as demo:
    gr.Markdown("# Vidhya-Setu — Offline AI Tutor")
    gr.Markdown("Enter a Class 9 Science topic (Motion or Force) to begin your learning session.")

    with gr.Row():
        topic_input = gr.Textbox(label="What do you want to learn about?", placeholder="e.g. acceleration, Newton's laws, velocity")
        start_btn = gr.Button("Start Learning", variant="primary")

    chain_display = gr.Textbox(label="Your Learning Path", interactive=False)
    current_concept_display = gr.Textbox(label="Current Concept", interactive=False)
    explanation_display = gr.Textbox(label="Tutor Explanation", lines=8, interactive=False)

    gr.Markdown("---")
    gr.Markdown("Reply below to continue the conversation (e.g. 'I don't understand' or 'got it, makes sense')")

    with gr.Row():
        reply_input = gr.Textbox(label="Your reply", placeholder="Type your response here")
        reply_btn = gr.Button("Send Reply")

    status_display = gr.Textbox(label="System Status", interactive=False)

    start_btn.click(
        start_session,
        inputs=[topic_input],
        outputs=[explanation_display, chain_display, current_concept_display],
    )

    reply_btn.click(
        respond_to_student,
        inputs=[reply_input],
        outputs=[explanation_display, status_display, current_concept_display],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)