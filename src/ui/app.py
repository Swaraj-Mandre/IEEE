import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gradio as gr

from src.agents.instructor import load_model, generate_explanation
from src.agents.diagnostic import (
    analyse_response,
    compute_gap_score,
    should_backtrack,
    should_advance,
)
from src.retrieval.path_tracker import (
    create_session,
    get_current_concept,
    advance,
    backtrack,
    can_backtrack,
    save_session_checkpoint,
)
from src.retrieval.retriever import load_graph, retrieve
from src.retrieval.vector_store import get_collection, get_embedding_model

print("Loading model and components, please wait...")
MODEL = load_model()
GRAPH = load_graph()
STORE = get_collection()
EMBED_MODEL = get_embedding_model()
print("Ready.")


def start_session(topic, state):
    """Find the concept the student asked about and explain the first step of its chain."""
    if not topic.strip():
        return "Please enter a topic to learn about.", "", "", state

    result = retrieve(topic, GRAPH, STORE, EMBED_MODEL)
    target = result["top_concept"]

    if not target:
        return "Could not find a matching concept. Try a different topic.", "", "", state

    # Each browser tab gets its own id, so two students never share a checkpoint file.
    session = create_session(f"student_{uuid.uuid4().hex[:8]}", target, GRAPH)
    state = {"session": session, "chunks": result["supporting_chunks"]}

    concept = get_current_concept(session)
    explanation = generate_explanation(MODEL, concept, state["chunks"], level="normal")
    save_session_checkpoint(session)

    return explanation, " -> ".join(session.prerequisite_chain), concept, state


def respond_to_student(student_reply, state):
    """Read the student's reply and decide whether to go back, move on, or explain again."""
    session = state.get("session")
    if session is None:
        return "Please start a session first by entering a topic above.", "", "", state

    concept = get_current_concept(session)
    result = analyse_response(student_reply, concept)

    session.exchange_count += 1
    gap = compute_gap_score(len(session.mastered_concepts), len(session.prerequisite_chain))
    status = (
        f"Diagnostic: {result['verdict']} | Gap score: {gap} "
        f"| Exchange {session.exchange_count}"
    )

    if should_backtrack(result, gap):
        # Check before the call, because at the start of the chain there is nowhere to step back to.
        at_start = not can_backtrack(session)
        new_concept = backtrack(session)
        explanation = generate_explanation(MODEL, new_concept, state["chunks"], level="simple")
        status += (
            " | Action: re-explaining with a simpler analogy (already at the first concept)"
            if at_start
            else f" | Action: stepped back (backtrack {session.backtrack_count})"
        )

    elif should_advance(result):
        last = len(session.prerequisite_chain) - 1
        if session.current_position >= last:
            # Final concept understood - record it and close out the path.
            if session.target_concept not in session.mastered_concepts:
                session.mastered_concepts.append(session.target_concept)
            new_concept = session.target_concept
            explanation = "Great work! You have completed the learning path for: " + new_concept
            status += " | Action: session complete"
        else:
            new_concept = advance(session)
            explanation = generate_explanation(MODEL, new_concept, state["chunks"], level="normal")
            status += " | Action: advanced to the next concept"

    else:
        new_concept = concept
        explanation = generate_explanation(MODEL, new_concept, state["chunks"], level="normal")
        status += " | Action: re-explaining the same concept"

    save_session_checkpoint(session)

    return explanation, status, new_concept, state


with gr.Blocks(title="Vidhya-Setu") as demo:
    gr.Markdown("# Vidhya-Setu - Offline AI Tutor")
    gr.Markdown("Enter a Class 9 Science topic (Motion or Force) to begin your learning session.")

    # Per-browser state. A module-level dict would let one student overwrite another's session.
    state = gr.State({})

    with gr.Row():
        topic_input = gr.Textbox(
            label="What do you want to learn about?",
            placeholder="e.g. acceleration, Newton's laws, velocity",
        )
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
        inputs=[topic_input, state],
        outputs=[explanation_display, chain_display, current_concept_display, state],
    )

    reply_btn.click(
        respond_to_student,
        inputs=[reply_input, state],
        outputs=[explanation_display, status_display, current_concept_display, state],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
