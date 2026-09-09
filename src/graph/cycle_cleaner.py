"""Turn the extracted graph into a DAG.

A cycle means the SLM claimed A is needed before B and B before A, which cannot
both be true. Edges confirmed by more chunks are kept over weaker ones.
"""
import networkx as nx


def _remove_mutual_edges(G: nx.DiGraph) -> int:
    """Drop the weaker side of every A->B / B->A pair."""
    removed = 0
    for u, v in sorted(G.edges()):
        if not G.has_edge(u, v) or not G.has_edge(v, u):
            continue
        forward = G.edges[u, v]["weight"]
        backward = G.edges[v, u]["weight"]
        # Equal weights are broken on name so the result is reproducible.
        loser = (v, u) if forward > backward or (forward == backward and u < v) else (u, v)
        G.remove_edge(*loser)
        removed += 1
    return removed


def _break_remaining_cycles(G: nx.DiGraph, max_passes: int = 10000) -> int:
    """Break longer cycles by removing their weakest edge, until the graph is acyclic."""
    removed = 0
    for _ in range(max_passes):
        try:
            cycle = nx.find_cycle(G, orientation="original")
        except nx.NetworkXNoCycle:
            break
        edges = [(u, v) for u, v, *_ in cycle]
        weakest = min(edges, key=lambda e: (G.edges[e[0], e[1]]["weight"], e[0], e[1]))
        G.remove_edge(*weakest)
        removed += 1
    return removed


def remove_cycles(G: nx.DiGraph) -> tuple[nx.DiGraph, dict]:
    """
    Return an acyclic copy of the graph plus a summary of what was removed.

    Returns:
        (dag, stats) where stats has edges_before, mutual_removed,
        cycles_broken, edges_after and is_dag.
    """
    dag = G.copy()
    before = dag.number_of_edges()

    mutual = _remove_mutual_edges(dag)
    broken = _break_remaining_cycles(dag)

    stats = {
        "edges_before": before,
        "mutual_removed": mutual,
        "cycles_broken": broken,
        "edges_after": dag.number_of_edges(),
        "is_dag": nx.is_directed_acyclic_graph(dag),
    }
    return dag, stats
