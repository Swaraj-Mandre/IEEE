import pickle
import networkx as nx
from dataclasses import dataclass, field


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
    This object is serialized to JSON for session checkpointing
    — survives power cuts, restarts, and browser refreshes.
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
) -> list[str]:
    """
    Build an ordered prerequisite chain from roots to target concept.

    Uses BFS backwards from target to find all ancestors,
    then topological sort to order them correctly.

    Example output for "acceleration":
        ["distance", "time", "speed", "velocity", "acceleration"]

    Args:
        G: Knowledge graph
        target_concept: The concept the student wants to learn
        max_depth: Maximum prerequisite depth to traverse

    Returns:
        Ordered list from most foundational to target concept
    """
    if target_concept not in G.nodes():
        # Concept not in graph — return just the concept itself
        return [target_concept]

    # BFS backwards to collect all ancestors within max_depth
    ancestors = set()
    frontier = {target_concept}

    for depth in range(max_depth):
        next_frontier = set()
        for node in frontier:
            for pred in G.predecessors(node):
                # Filter garbage nodes
                if pred.lower() in GARBAGE_NODES:
                    continue
                if len(pred) < 3:
                    continue
                if pred not in ancestors:
                    ancestors.add(pred)
                    next_frontier.add(pred)
        if not next_frontier:
            break
        frontier = next_frontier

    # Build subgraph of ancestors + target
    relevant_nodes = ancestors | {target_concept}
    subgraph = G.subgraph(relevant_nodes).copy()

    # Topological sort gives correct learning order
    try:
        ordered = list(nx.topological_sort(subgraph))
        # Ensure target is last
        if target_concept in ordered:
            ordered.remove(target_concept)
        ordered.append(target_concept)
    except nx.NetworkXUnfeasible:
        # Fallback if subgraph somehow has cycles
        ordered = list(ancestors) + [target_concept]

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
        marker = " ← START" if i == 0 else ""
        marker = " ← TARGET" if concept == target_concept else marker
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


def backtrack(session: LearningSession) -> str:
    """
    Step back to an easier prerequisite.
    Called when diagnostic agent detects confusion.
    This is the core novelty claim of the paper.

    Returns the simpler concept to explain instead.
    """
    if session.current_position > 0:
        session.current_position -= 1
        session.backtrack_count += 1
        confused = get_current_concept(session)
        session.confused_concepts.append(confused)
        print(f"  [BACKTRACK #{session.backtrack_count}] "
              f"Stepping back to: {confused}")
        return confused
    else:
        # Already at root — cannot go further back
        print("  [BACKTRACK] Already at root concept. "
              "Switching to analogy-based explanation.")
        return get_current_concept(session)


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