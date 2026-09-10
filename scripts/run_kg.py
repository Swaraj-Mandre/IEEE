import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.graph.extractor import load_model, load_grammar, extract_triples_from_chunk
from src.graph.graph_builder import build_graph, save_graph, save_audit_json, get_graph_stats
from src.graph.cycle_cleaner import remove_cycles
from src.ingestion.chunker import load_chunks


def run_kg_pipeline():
    print("=" * 60)
    print("VIDHYA-SETU - KNOWLEDGE GRAPH CONSTRUCTION")
    print("=" * 60)

    # Load chunks
    all_chunks = []
    for chunk_file in config.CHUNK_FILES:
        if not chunk_file.exists():
            print(f"ERROR: {chunk_file.name} not found. Run scripts/run_ingestion.py first.")
            return
        chunks = load_chunks(str(chunk_file))
        all_chunks.extend(chunks)
        print(f"Loaded {len(chunks)} chunks from {chunk_file.name}")

    print(f"\nTotal chunks to process: {len(all_chunks)}")
    print("Estimated time: 15-40 minutes on CPU, 2-5 minutes with GPU\n")

    # Load SLM
    model = load_model()
    grammar = load_grammar()

    # Extract triples from each chunk
    all_triples = []
    failed_chunks = 0
    start_time = time.time()

    print("Extracting concept-prerequisite triples...")
    print("-" * 40)

    for i, chunk in enumerate(all_chunks):
        # Progress indicator every 10 chunks
        if i % 10 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (len(all_chunks) - i) / rate if rate > 0 else 0
            print(f"  Progress: {i}/{len(all_chunks)} chunks "
                  f"| {elapsed:.0f}s elapsed "
                  f"| ~{remaining:.0f}s remaining "
                  f"| {len(all_triples)} triples so far")

        triples = extract_triples_from_chunk(model, chunk, grammar)

        if triples:
            all_triples.extend(triples)
        else:
            failed_chunks += 1

    total_time = time.time() - start_time

    print(f"\nExtraction complete:")
    print(f"  Total triples extracted: {len(all_triples)}")
    print(f"  Chunks with no output:   {failed_chunks}/{len(all_chunks)}")
    print(f"  Total time:              {total_time:.0f}s")
    print(f"  Avg per chunk:           {total_time/len(all_chunks):.1f}s")

    if len(all_triples) < 50:
        print("\nWARNING: Very few triples extracted.")
        print("Check your MODEL_PATH in .env and verify the model loaded correctly.")
        print("Run: python -c \"from src.graph.extractor import load_model; load_model()\"")

    # Build graph
    print("\nBuilding knowledge graph...")
    G = build_graph(all_triples)

    # Cycles are extraction errors, not real prerequisites, so strip them before saving.
    G, cycle_stats = remove_cycles(G)
    print(f"  Cycle removal: {cycle_stats['edges_before']} -> {cycle_stats['edges_after']} edges "
          f"({cycle_stats['mutual_removed']} mutual, {cycle_stats['cycles_broken']} longer cycles)")

    # Print stats
    stats = get_graph_stats(G)
    print("\nGRAPH STATISTICS (these go in your IEEE paper):")
    print("-" * 40)
    print(f"  Total concepts (nodes):     {stats['total_nodes']}")
    print(f"  Total relationships (edges): {stats['total_edges']}")
    print(f"  Root concepts (no prereqs):  {stats['root_nodes_count']}")
    print(f"  Is valid DAG (no cycles):    {stats['is_dag']}")
    print(f"  Cycles detected:             {stats['cycle_count']}")
    print(f"  Most connected concept:      {stats['most_connected_concept']}")
    print(f"  Average degree:              {stats['avg_degree']}")

    if not stats["is_dag"]:
        print(f"\n  WARNING: {stats['cycle_count']} cycles found.")
        print("  Cycles = logical errors in prerequisite relationships.")
        print("  Review kg_audit.json and remove incorrect extractions.")

    # Save outputs
    print("\nSaving outputs...")
    graph_dir = config.GRAPH_PATH.parent
    graph_dir.mkdir(parents=True, exist_ok=True)
    save_graph(G, str(config.GRAPH_PATH))
    save_audit_json(G, all_triples, str(graph_dir / "kg_audit.json"))
    save_sample_review(G, graph_dir / "kg_sample_review.json")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
1. Rebuild the index so it matches the new graph:
     python scripts/build_vectorstore.py

2. Confirm nothing regressed:
     python scripts/run_checks.py

3. Spot-check the extraction by hand in data/graph/kg_sample_review.json.
   For each edge ask: is the prerequisite really needed first, are both
   ends real concepts, and is the direction right (prerequisite -> concept)?
   Correct:   speed -> velocity
   Wrong:     newton's law -> force   (force is needed to understand the law)
   Aim for 70%+ acceptance on the high-confidence edges. Below that, the
   extraction prompt in src/graph/extractor.py needs work.
""")


def save_sample_review(G, output_path: Path) -> None:
    """Write a short, hand-checkable slice of the graph. The full graph is too big to read."""
    edges = sorted(G.edges(), key=lambda e: (-G.edges[e]["weight"], e))

    sample = {
        "how_to_review": (
            "Check each edge: is the prerequisite genuinely needed before the "
            "concept, are both ends real concepts, and is the direction right?"
        ),
        "high_confidence_edges": [
            {"prerequisite": u, "concept": v, "weight": G.edges[u, v]["weight"]}
            for u, v in edges
            if G.edges[u, v]["weight"] >= 2
        ][:20],
        "sample_edges": [
            {"prerequisite": u, "concept": v, "weight": G.edges[u, v]["weight"]}
            for u, v in edges[:20]
        ],
    }

    output_path.write_text(json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Sample review saved to {output_path}")


if __name__ == "__main__":
    run_kg_pipeline()