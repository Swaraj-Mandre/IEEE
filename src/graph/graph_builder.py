import pickle
import json
import networkx as nx
from collections import defaultdict
from pathlib import Path


def build_graph(all_triples: list[dict]) -> nx.DiGraph:
    """
    Build a directed knowledge graph from extracted triples.

    Graph structure:
        Nodes = STEM concepts (strings, lowercased for deduplication)
        Edges = prerequisite relationships
                edge A → B means "A must be learned before B"

    Node attributes stored:
        - original_forms: set of original capitalizations seen
        - source_chunks: list of chunk_ids that mentioned this concept
        - mention_count: how many times this concept appeared

    Edge attributes stored:
        - weight: how many different chunks support this relationship
                  (higher weight = more confident relationship)
        - source_chunks: which chunks produced this edge
    """
    G = nx.DiGraph()

    # Track weighting
    edge_support = defaultdict(list)

    for triple in all_triples:
        # Normalize to lowercase for deduplication
        # "Velocity" and "velocity" are the same concept
        concept = triple["concept"].strip().lower()
        prerequisite = triple["prerequisite"].strip().lower()
        chunk_id = triple.get("chunk_id", "unknown")

        if not G.has_node(concept):
            G.add_node(concept,
                original_forms=set(),
                source_chunks=[],
                mention_count=0
            )
        G.nodes[concept]["original_forms"].add(triple["concept"].strip())
        G.nodes[concept]["source_chunks"].append(chunk_id)
        G.nodes[concept]["mention_count"] += 1

        if not G.has_node(prerequisite):
            G.add_node(prerequisite,
                original_forms=set(),
                source_chunks=[],
                mention_count=0
            )
        G.nodes[prerequisite]["original_forms"].add(triple["prerequisite"].strip())
        G.nodes[prerequisite]["source_chunks"].append(chunk_id)
        G.nodes[prerequisite]["mention_count"] += 1

        edge_key = (prerequisite, concept)
        edge_support[edge_key].append(chunk_id)

    # Add edges
    for (prereq, concept), supporting_chunks in edge_support.items():
        G.add_edge(
            prereq,
            concept,
            weight=len(supporting_chunks),
            source_chunks=supporting_chunks,
        )

    return G


def get_graph_stats(G: nx.DiGraph) -> dict:
    """
    Compute graph statistics for the paper's dataset section.(IEEE)
    """
    # Find root nodes (no prerequisites themselves)
    root_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]

    # Find leaf nodes (no concepts depend on them)
    leaf_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]

    # Find the most connected concept (highest total degree)
    if G.nodes():
        most_connected = max(G.nodes(), key=lambda n: G.degree(n))
    else:
        most_connected = "none"

    # Check for cycles — cycles in a prerequisite graph are logical errors
    # e.g., A requires B and B requires A is impossible
    has_cycles = not nx.is_directed_acyclic_graph(G)
    cycle_count = 0
    if has_cycles:
        cycles = list(nx.simple_cycles(G))
        cycle_count = len(cycles)

    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "root_nodes_count": len(root_nodes),
        "leaf_nodes_count": len(leaf_nodes),
        "most_connected_concept": most_connected,
        "most_connected_degree": G.degree(most_connected) if G.nodes() else 0,
        "is_dag": not has_cycles,
        "cycle_count": cycle_count,
        "avg_degree": round(
            sum(d for _, d in G.degree()) / G.number_of_nodes(), 2
        ) if G.number_of_nodes() > 0 else 0,
        "root_nodes_sample": root_nodes[:10],
        "leaf_nodes_sample": leaf_nodes[:10],
    }


def save_graph(G: nx.DiGraph, output_path: str) -> None:
    """
    Save graph as pickle for fast loading in Phase 3.
    Pickle preserves all node/edge attributes including sets.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Convert sets to lists before pickling
    # Sets are not JSON-serializable — needed for audit export
    for node in G.nodes():
        if isinstance(G.nodes[node].get("original_forms"), set):
            G.nodes[node]["original_forms"] = list(
                G.nodes[node]["original_forms"]
            )

    with open(output_path, "wb") as f:
        pickle.dump(G, f)

    print(f"  Graph saved to {output_path}")
    print(f"  File size: {Path(output_path).stat().st_size / 1024:.1f} KB")


def save_audit_json(
    G: nx.DiGraph,
    all_triples: list[dict],
    output_path: str
) -> None:
    """
    Save human-readable audit file.
    You and your team manually review this to validate extraction quality.
    This is the most important quality gate in Phase 2.
    """
    audit = {
        "graph_stats": get_graph_stats(G),
        "all_edges": [
            {
                "prerequisite": u,
                "concept": v,
                "weight": G.edges[u, v]["weight"],
                "supporting_chunks": G.edges[u, v]["source_chunks"],
            }
            for u, v in G.edges()
        ],
        "all_triples_raw": all_triples,
        "high_confidence_edges": [
            {"prerequisite": u, "concept": v, "weight": G.edges[u, v]["weight"]}
            for u, v in G.edges()
            if G.edges[u, v]["weight"] >= 2
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    print(f"  Audit JSON saved to {output_path}")