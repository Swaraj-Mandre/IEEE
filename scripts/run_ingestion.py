import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pdf_extractor import extract_pdf
from src.ingestion.chunker import chunk_pages, save_chunks


def run_ingestion():
    raw_pdf_dir = Path("data/raw_pdfs")
    chunks_dir = Path("data/chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Folders starting with "_" hold superseded sources and are not ingested.
    pdf_files = [
        p for p in raw_pdf_dir.rglob("*.pdf")
        if not any(part.startswith("_") for part in p.relative_to(raw_pdf_dir).parts)
    ]

    if not pdf_files:
        print("ERROR: No PDFs found in data/raw_pdfs/")
        print("Download NCERT PDFs from https://ncert.nic.in/textbook.php")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process\n")

    summary = []

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")

        #Extract text from PDF
        pages = extract_pdf(str(pdf_path))

        #Chunk the extracted text
        chunks = chunk_pages(pages)

        if not chunks:
            print(f"  WARNING: No valid chunks produced from {pdf_path.name}")
            continue

        #Save chunks to JSON
        output_file = chunks_dir / f"{pdf_path.stem}_chunks.json"
        save_chunks(chunks, str(output_file))

        summary.append({
            "file": pdf_path.name,
            "total_pages": len(pages),
            "valid_pages": sum(1 for p in pages if p["is_valid"]),
            "total_chunks": len(chunks),
            "avg_tokens": round(
                sum(c["estimated_tokens"] for c in chunks) / len(chunks)
            ),
        })

        print()

    print("=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    for s in summary:
        print(f"File:          {s['file']}")
        print(f"Pages:         {s['valid_pages']}/{s['total_pages']} valid")
        print(f"Chunks:        {s['total_chunks']}")
        print(f"Avg tokens:    ~{s['avg_tokens']}")
        print("-" * 40)

    # Saving summary for paper reference
    summary_path = Path("data/chunks/ingestion_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    run_ingestion()