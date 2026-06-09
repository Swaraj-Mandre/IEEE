import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.extractor import load_model, extract_triples_from_chunk
from src.graph.graph_builder import build_graph, save_graph, save_audit_json, get_graph_stats
from src.ingestion.chunker import load_chunks

# Add more files here after validation passes
TARGET_CHUNKS = [
    "data/chunks/iesc108_chunks.json",  # Chapter 8: Motion
    "data/chunks/iesc109_chunks.json",  # Chapter 9: Force and Laws of Motion
]


def run_kg_pipeline():
    print("=" * 60)
    print("VIDHYA-SETU — KNOWLEDGE GRAPH CONSTRUCTION")
    print("=" * 60)

    # Load chunks
    all_chunks = []
    for chunk_file in TARGET_CHUNKS:
        if not Path(chunk_file).exists():
            print(f"ERROR: {chunk_file} not found. Run Phase 1 first.")
            return
        chunks = load_chunks(chunk_file)
        all_chunks.extend(chunks)
        print(f"Loaded {len(chunks)} chunks from {chunk_file}")

    print(f"\nTotal chunks to process: {len(all_chunks)}")
    print("Estimated time: 15-40 minutes on CPU, 2-5 minutes with GPU\n")

    # Load SLM
    model = load_model()

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

        triples = extract_triples_from_chunk(model, chunk)

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
    Path("data/graph").mkdir(parents=True, exist_ok=True)
    save_graph(G, "data/graph/kg.pkl")
    save_audit_json(G, all_triples, "data/graph/kg_audit.json")

    # Manual validation instructions
    print("\n" + "=" * 60)
    print("MANUAL VALIDATION REQUIRED BEFORE PHASE 3")
    print("=" * 60)
    print("""
Open data/graph/kg_audit.json in VS Code.

Check 'all_edges' section. For each edge ask:
  1. Does the prerequisite make sense to learn before the concept?
  2. Are both terms real STEM concepts (not vague words)?
  3. Is the direction correct? (prerequisite → concept)

Good example:   speed → velocity         (correct direction)
Bad example:    newton's law → force      (direction wrong — force
                                           is needed to understand laws)
Bad example:    matter → matter           (self-loop, rejected already)
Bad example:    the → displacement        (not a concept)

Check 'high_confidence_edges' first — these appear in 2+ chunks
and are almost certainly correct.

Record how many edges you accept vs reject.
Target: >70% acceptance rate on high_confidence_edges.
If below 70%, the prompt needs tuning — tell your advisor.
""")

    # Save a quick-view sample for manual review
    sample_path = "data/graph/kg_sample_review.json"
    sample = {
        "instructions": "Review these 20 edges manually before Phase 3",
        "high_confidence_edges": stats.get("high_confidence_edges", [])[:20]
        if "high_confidence_edges" in stats
        else [],
        "sample_edges": [
            {"prerequisite": u, "concept": v, "weight": G.edges[u, v]["weight"]}
            for u, v in list(G.edges())[:20]
        ],
    }

    # Get high confidence from audit
    hc = [
        {"prerequisite": u, "concept": v, "weight": G.edges[u, v]["weight"]}
        for u, v in G.edges()
        if G.edges[u, v]["weight"] >= 2
    ]
    sample["high_confidence_edges"] = hc[:20]

    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)

    print(f"20-edge sample saved to {sample_path}")
    print("Review this file before proceeding to Phase 3.\n")


if __name__ == "__main__":
    run_kg_pipeline()