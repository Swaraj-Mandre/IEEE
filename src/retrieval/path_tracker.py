import json
import networkx as nx
from dataclasses import dataclass, field

from src import config


# Garbage nodes the SLM occasionally produces
# These are filtered out during path construction
GARBAGE_NODES = {
    "none", "n/a", "na", "null", "unknown",
    "the", "a", "an", "it", "this", "that",
}


@dataclass
class LearningSession:
    """
    Tracks a student's learning state across the session.
    Serialized to JSON after every exchange, so a power cut, a restart or a
    browser refresh loses at most one turn.
    """
    student_id: str
    target_concept: str                    # What the student asked about
    prerequisite_chain: list[str]          # Full ordered path, root → target
    current_position: int = 0             # Index in prerequisite_chain
    mastered_concepts: list[str] = field(default_factory=list)
    confused_concepts: list[str] = field(default_factory=list)
    backtrack_count: int = 0              # How many times we stepped back
    exchange_count: int = 0              # Total student exchanges


def get_prerequisite_chain(
    G: nx.DiGraph,
    target_concept: str,
    max_depth: int = 3,
    max_length: int = 8,
) -> list[str]:
    """
    Build an ordered prerequisite chain leading up to the target concept.

    Walks backwards from the target one level at a time. max_depth limits how far
    back it looks; max_length caps the chain, because a single concept can have
    hundreds of ancestors and no student will work through those.

    Example output for "acceleration":
        ["distance", "time", "speed", "velocity", "acceleration"]

    Returns:
        Ordered list from most foundational to target concept
    """
    if target_concept not in G.nodes():
        # Not in the graph, so the chain is just the concept itself.
        return [target_concept]

    ancestors: list[str] = []
    seen = {target_concept}
    frontier = {target_concept}

    for _ in range(max_depth):
        if len(ancestors) >= max_length - 1:
            break

        # Collect this level's prerequisites, keeping the strongest edge when a
        # concept is reached from more than one place.
        candidates: dict[str, int] = {}
        for node in frontier:
            for pred in G.predecessors(node):
                if pred in seen or pred.lower() in GARBAGE_NODES or len(pred) < 3:
                    continue
                weight = G.edges[pred, node]["weight"]
                candidates[pred] = max(candidates.get(pred, 0), weight)

        if not candidates:
            break

        # Edges confirmed by more chunks are more trustworthy, so they fill the chain first.
        ranked = sorted(candidates, key=lambda n: (-candidates[n], n))
        seen.update(ranked)
        ancestors.extend(ranked[: max_length - 1 - len(ancestors)])
        frontier = set(ranked)

    subgraph = G.subgraph(set(ancestors) | {target_concept}).copy()

    # Lexicographical sort, not plain topological: several orderings are valid here
    # and the plain version picks a different one on every run.
    try:
        ordered = list(nx.lexicographical_topological_sort(subgraph))
        # Ensure target is last
        if target_concept in ordered:
            ordered.remove(target_concept)
        ordered.append(target_concept)
    except nx.NetworkXUnfeasible:
        # Fallback if subgraph somehow has cycles
        ordered = ancestors + [target_concept]

    return ordered


def create_session(
    student_id: str,
    target_concept: str,
    G: nx.DiGraph,
) -> LearningSession:
    """Create a new learning session for a student."""
    chain = get_prerequisite_chain(G, target_concept)

    session = LearningSession(
        student_id=student_id,
        target_concept=target_concept,
        prerequisite_chain=chain,
        current_position=0,
    )

    print(f"\nNew session for student '{student_id}'")
    print(f"Target concept: {target_concept}")
    print(f"Prerequisite chain ({len(chain)} steps):")
    for i, concept in enumerate(chain):
        marker = " <- START" if i == 0 else ""
        marker = " <- TARGET" if concept == target_concept else marker
        print(f"  {i+1}. {concept}{marker}")

    return session


def get_current_concept(session: LearningSession) -> str:
    """Return the concept the student should be learning right now."""
    if session.current_position >= len(session.prerequisite_chain):
        return session.target_concept
    return session.prerequisite_chain[session.current_position]


def advance(session: LearningSession) -> str:
    """
    Move to the next concept in the chain.
    Called when the diagnostic agent confirms understanding.
    Returns the new current concept.
    """
    if session.current_position < len(session.prerequisite_chain) - 1:
        session.current_position += 1
        concept = get_current_concept(session)
        session.mastered_concepts.append(
            session.prerequisite_chain[session.current_position - 1]
        )
        return concept
    return session.target_concept


def can_backtrack(session: LearningSession) -> bool:
    """True if there is an earlier concept to step back to."""
    return session.current_position > 0


def backtrack(session: LearningSession) -> str:
    """
    Step back to an easier prerequisite when the student is confused.

    At the start of the chain there is nowhere to go, so the same concept is
    returned and the caller falls back to an analogy instead.
    """
    confused = get_current_concept(session)
    if confused not in session.confused_concepts:
        session.confused_concepts.append(confused)

    if not can_backtrack(session):
        print("  [BACKTRACK] Already at the first concept. "
              "Switching to analogy-based explanation.")
        return confused

    session.current_position -= 1
    session.backtrack_count += 1
    simpler = get_current_concept(session)
    print(f"  [BACKTRACK #{session.backtrack_count}] Stepping back to: {simpler}")
    return simpler


def compute_gap_score(session: LearningSession) -> float:
    """
    Compute how much of the prerequisite chain is unmastered.

    Formula:
        gap_score = unmastered_prerequisites / total_prerequisites

    Interpretation:
        0.0 = student has mastered everything up to current point
        1.0 = student has mastered nothing

    Threshold: gap_score > 0.4 triggers backtracking.
    This threshold and formula go in the paper's method section.
    """
    total = len(session.prerequisite_chain)
    if total == 0:
        return 0.0

    mastered = len(session.mastered_concepts)
    gap_score = 1 - (mastered / total)
    return round(gap_score, 4)


def session_to_dict(session: LearningSession) -> dict:
    """
    Serialize session to dict for JSON checkpointing.
    Called after every exchange to ensure power-cut safety.
    """
    return {
        "student_id": session.student_id,
        "target_concept": session.target_concept,
        "prerequisite_chain": session.prerequisite_chain,
        "current_position": session.current_position,
        "mastered_concepts": session.mastered_concepts,
        "confused_concepts": session.confused_concepts,
        "backtrack_count": session.backtrack_count,
        "exchange_count": session.exchange_count,
    }


def session_from_dict(data: dict) -> LearningSession:
    """Restore session from JSON checkpoint."""
    return LearningSession(
        student_id=data["student_id"],
        target_concept=data["target_concept"],
        prerequisite_chain=data["prerequisite_chain"],
        current_position=data["current_position"],
        mastered_concepts=data["mastered_concepts"],
        confused_concepts=data["confused_concepts"],
        backtrack_count=data["backtrack_count"],
        exchange_count=data["exchange_count"],
    )


def save_session_checkpoint(session: LearningSession) -> None:
    """Write the session to disk after every exchange so a power cut loses at most one turn."""
    config.SESSION_DIR.mkdir(parents=True, exist_ok=True)
    path = config.SESSION_DIR / f"{session.student_id}_session.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session_to_dict(session), f, indent=2, ensure_ascii=False)