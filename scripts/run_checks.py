"""Regression checks for the whole pipeline. Run this after any change.

Covers the knowledge graph, the search index, retrieval quality, prerequisite
chains and the diagnostic rules. No language model is loaded, so it finishes in
under a minute. Exits non-zero if anything regressed.

Usage:  python scripts/run_checks.py
"""
import contextlib
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx
import numpy as np

from src.agents.diagnostic import (
    CONFUSION_SIGNALS,
    UNDERSTANDING_SIGNALS,
    analyse_response,
    compute_gap_score,
    should_advance,
    should_backtrack,
)
from src.retrieval.path_tracker import (
    advance,
    backtrack,
    can_backtrack,
    create_session,
    get_current_concept,
    get_prerequisite_chain,
)
from src.retrieval.retriever import extract_concepts_from_chunks, load_graph, retrieve
from src.retrieval.vector_store import get_collection, get_embedding_model

TOP1, TOP3 = "top1", "top3"

# Floors, not targets. A graph built from the two physics chapters lands well
# above these; dropping under one means extraction broke.
MIN_NODES, MIN_EDGES = 100, 100
MAX_CHAIN = 8

# Question -> the concept the tutor should start the student on.
RETRIEVAL_CASES = [
    ("What is acceleration?", "acceleration", TOP1),
    ("What is friction?", "friction", TOP1),
    ("What is momentum?", "momentum", TOP1),
    ("What is displacement?", "displacement", TOP1),
    ("How does Newton second law work?", "newton's second law of motion", TOP1),
    ("What is the third law of motion?", "newton's third law of motion", TOP1),
    ("What is a balanced force?", "balanced forces", TOP1),
    ("What is uniform motion?", "uniform motion", TOP3),
    ("How do you calculate distance travelled?", "distance travelled", TOP3),
    ("What is the difference between speed and velocity?", "average velocity", TOP3),
]

# Questions the system does not answer well today. Kept here so the reason stays
# written down, and so a future fix shows up instead of passing unnoticed.
KNOWN_GAPS = {
    "What is inertia?": "inertia appears in one chunk only, so it never became a node",
    "What is free fall?": "free fall is not a node; the chapters treat it as gravitational acceleration",
}

DIAGNOSTIC_CASES = [
    ("I don't understand", "confused"),
    ("nothing is clear to me", "confused"),
    ("Can you explain again?", "confused"),
    ("this is too hard", "confused"),
    # These read as positive if the negation is missed, which is the failure
    # mode the whole-word longest-first matching exists to prevent.
    ("not clear", "confused"),
    ("it is not very clear", "confused"),
    ("this makes no sense", "confused"),
    ("does not make sense", "confused"),
    ("I'm not sure", "confused"),
    ("no idea", "confused"),
    ("makes sense now", "understood"),
    ("Got it, thanks", "understood"),
    ("I know this already", "understood"),
    ("crystal clear", "understood"),
    ("clear now, thanks", "understood"),
    ("okay", "understood"),
    ("yes", "understood"),
    ("the sky is blue", "unclear"),
]

CHAIN_TARGETS = ["acceleration", "friction", "velocity", "momentum", "force"]


# Written out here rather than imported from graph_builder, so that loosening
# the filter and rebuilding the graph shows up as a failure instead of passing
# by definition.
BOOK_FURNITURE = re.compile(r"\b(fig|figure|table|activity|chapter|section|grade|page|exercise)\b", re.I)
FILLER_WORDS = {"none", "n/a", "null", "unknown", "the", "a", "an", "it", "this", "that", "example"}


def looks_like_a_concept(name: str) -> bool:
    """A node should be a short noun phrase, not a filler word or a cross-reference."""
    return (
        len(name) >= 3
        and name.lower() not in FILLER_WORDS
        and len(name.split()) <= 5
        and not BOOK_FURNITURE.search(name)
        and bool(re.search(r"[a-z]{3}", name.lower()))
    )


class Report:
    """Collects pass/fail lines and decides the exit code."""

    def __init__(self):
        self.failures = []
        self.total = 0

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        self.total += 1
        print(f"  [{'ok  ' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
        if not ok:
            self.failures.append(label)
        return ok

    def section(self, name: str) -> None:
        print(f"\n{name}")

    def summary(self) -> int:
        print("\n" + "-" * 62)
        if self.failures:
            print(f"{len(self.failures)} of {self.total} checks FAILED:")
            for f in self.failures:
                print(f"  - {f}")
            return 1
        print(f"All {self.total} checks passed.")
        return 0


def check_graph(report: Report, G: nx.DiGraph) -> None:
    report.section("Knowledge graph")
    n, e = G.number_of_nodes(), G.number_of_edges()

    report.check(n >= MIN_NODES, f"at least {MIN_NODES} concepts", f"{n} nodes")
    report.check(e >= MIN_EDGES, f"at least {MIN_EDGES} relationships", f"{e} edges")
    report.check(nx.is_directed_acyclic_graph(G), "no cycles (valid DAG)")
    report.check(not list(nx.selfloop_edges(G)), "no concept is its own prerequisite")

    bad = [n for n in G.nodes() if not looks_like_a_concept(n)]
    report.check(not bad, "every node is a real concept", f"rejected: {bad[:5]}" if bad else "")

    orphans = [n for n in G.nodes() if G.degree(n) == 0]
    report.check(not orphans, "no unreachable concepts", f"{len(orphans)} orphans" if orphans else "")


def check_index(report: Report, store) -> None:
    report.section("Search index")
    rows, chunks = store.embeddings.shape[0], store.count()

    report.check(rows == chunks, "one vector per chunk", f"{rows} vectors, {chunks} chunks")
    report.check(chunks > 100, "index is populated", f"{chunks} chunks")

    # Cosine similarity is a plain dot product only while the vectors are unit length.
    norms = np.linalg.norm(store.embeddings, axis=1)
    report.check(bool(np.allclose(norms, 1.0, atol=1e-3)), "vectors are normalised")

    required = {"chunk_id", "text", "source_file", "page_num"}
    missing = [c["chunk_id"] for c in store.chunks if not required <= set(c)]
    report.check(not missing, "every chunk carries the fields the UI reads")

    tagged = sum(1 for c in store.chunks if c.get("section"))
    report.check(tagged / chunks > 0.8, "most chunks know their section", f"{tagged}/{chunks}")


def check_retrieval(report: Report, G: nx.DiGraph, store, embed_model) -> None:
    report.section("Retrieval")

    for question, expected, strictness in RETRIEVAL_CASES:
        result = retrieve(question, G, store, embed_model)
        ranked = [c["concept"] for c in result["fused_results"]]
        window = ranked[:1] if strictness == TOP1 else ranked[:3]
        report.check(expected in window, f"{question} -> {expected}", f"got {ranked[:3]}")

    # Same question twice must give the same answer, or a student gets a
    # different lesson each time they ask.
    first = retrieve(RETRIEVAL_CASES[0][0], G, store, embed_model)
    second = retrieve(RETRIEVAL_CASES[0][0], G, store, embed_model)
    report.check(first["fused_results"] == second["fused_results"], "repeat query is stable")

    if KNOWN_GAPS:
        print("\n  Known gaps (reported, not failures):")
        for question, reason in KNOWN_GAPS.items():
            top = retrieve(question, G, store, embed_model)["top_concept"]
            print(f"    {question} -> {top}  ({reason})")


def check_chains(report: Report, G: nx.DiGraph) -> None:
    report.section("Prerequisite chains")

    for target in CHAIN_TARGETS:
        chain = get_prerequisite_chain(G, target)
        ok = (
            chain[-1] == target
            and len(chain) <= MAX_CHAIN
            and len(set(chain)) == len(chain)
        )
        report.check(ok, f"chain for '{target}'", f"{len(chain)} steps")

    # A chain that reorders between calls sends the student through a different
    # lesson each session.
    report.check(
        all(get_prerequisite_chain(G, t) == get_prerequisite_chain(G, t) for t in CHAIN_TARGETS),
        "chains are stable across calls",
    )

    missing = get_prerequisite_chain(G, "not-a-real-concept")
    report.check(missing == ["not-a-real-concept"], "unknown concept degrades gracefully")


def check_session(report: Report, G: nx.DiGraph) -> None:
    report.section("Session routing")

    # These call the same functions the UI does, and those log to the console.
    with contextlib.redirect_stdout(io.StringIO()):
        session = create_session("check_student", CHAIN_TARGETS[0], G)
        start = get_current_concept(session)
        at_start = can_backtrack(session)
        stayed = backtrack(session)
        no_count = session.backtrack_count
        recorded = start in session.confused_concepts

        advance(session)
        moved = get_current_concept(session)
        can_go_back = can_backtrack(session)
        back_to = backtrack(session)
        counted = session.backtrack_count

        # Walk off the end; the chain must stop at the target rather than overrun.
        for _ in range(len(session.prerequisite_chain) + 2):
            advance(session)
        ended_on = get_current_concept(session)

    report.check(not at_start, "starts at the first concept")
    report.check(stayed == start, "backtrack at the start stays put")
    report.check(no_count == 0, "no backtrack counted when nothing moved")
    report.check(recorded, "the confusing concept is recorded")
    report.check(moved != start, "advance moves forward")
    report.check(can_go_back, "can step back once past the start")
    report.check(back_to == start, "backtrack returns to the earlier concept")
    report.check(counted == 1, "a real backtrack is counted")
    report.check(ended_on == session.target_concept, "chain ends at the target")


def check_diagnostic(report: Report) -> None:
    report.section("Diagnostic rules")
    wrong = []
    for reply, expected in DIAGNOSTIC_CASES:
        verdict = analyse_response(reply, "acceleration")["verdict"]
        if verdict != expected:
            wrong.append(f"{reply!r} -> {verdict}, expected {expected}")
    report.check(
        not wrong, f"{len(DIAGNOSTIC_CASES)} replies classified", "; ".join(wrong) if wrong else ""
    )

    # Every signal phrase must still classify as its own list. This catches an
    # edit to one list quietly cancelling a phrase in the other.
    self_wrong = [
        p for p, want in
        [(s, "confused") for s in CONFUSION_SIGNALS] + [(s, "understood") for s in UNDERSTANDING_SIGNALS]
        if analyse_response(p, "acceleration")["verdict"] != want
    ]
    report.check(
        not self_wrong,
        f"all {len(CONFUSION_SIGNALS) + len(UNDERSTANDING_SIGNALS)} signal phrases self-classify",
        f"wrong: {self_wrong[:5]}" if self_wrong else "",
    )

    report.check(compute_gap_score(0, 8) == 1.0, "gap score is 1.0 with nothing mastered")
    report.check(compute_gap_score(8, 8) == 0.0, "gap score is 0.0 when everything is mastered")

    confused = analyse_response("I don't understand", "acceleration")
    understood = analyse_response("makes sense", "acceleration")
    report.check(should_backtrack(confused, 0.5), "confusion routes to backtrack")
    report.check(should_advance(understood), "understanding routes to advance")
    report.check(not should_advance(confused), "confusion never routes to advance")


def fingerprint(G: nx.DiGraph, store) -> dict:
    """Everything whose order could drift, reduced to one comparable blob."""
    chunks = store.chunks[:5]
    return {
        "chains": {t: get_prerequisite_chain(G, t) for t in CHAIN_TARGETS},
        "concepts": extract_concepts_from_chunks(chunks, G, question="what is force")[:10],
    }


def check_hash_order(report: Report) -> None:
    """Run the ordering-sensitive parts under two hash seeds.

    Python randomises string hashing per process, so anything that walks a set
    can come out in a different order on the school machine than it did here.
    """
    report.section("Ordering (two hash seeds)")
    results = []
    for seed in ("0", "1"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        proc = subprocess.run(
            [sys.executable, __file__, "--fingerprint"],
            capture_output=True, text=True, env=env, cwd=Path(__file__).parent.parent,
        )
        if proc.returncode != 0:
            report.check(False, f"fingerprint under PYTHONHASHSEED={seed}", proc.stderr.strip()[-200:])
            return
        results.append(json.loads(proc.stdout))

    report.check(results[0] == results[1], "same output under different hash seeds")


def main() -> int:
    if "--fingerprint" in sys.argv:
        print(json.dumps(fingerprint(load_graph(), get_collection())))
        return 0

    print("=" * 62)
    print("VIDHYA-SETU - REGRESSION CHECKS")
    print("=" * 62)

    report = Report()
    G = load_graph()
    store = get_collection()

    check_graph(report, G)
    check_index(report, store)
    check_chains(report, G)
    check_session(report, G)
    check_diagnostic(report)
    check_hash_order(report)

    embed_model = get_embedding_model()
    check_retrieval(report, G, store, embed_model)

    return report.summary()


if __name__ == "__main__":
    sys.exit(main())
