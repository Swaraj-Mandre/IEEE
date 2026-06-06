import json
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(pages: list[dict], chunk_size: int = 400, overlap: int = 80) -> list[dict]:
    """
    Split page texts into overlapping chunks for retrieval.

    chunk_size: ~400 chars ≈ 80-100 tokens. Good balance for
                a small SLM context window.
    overlap:    80 chars overlap so concepts at page boundaries
                are not split mid-sentence.

    Returns:
        List of chunk dicts with unique IDs and metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []

    for page in pages:
        if not page["is_valid"]:
            continue

        raw_chunks = splitter.split_text(page["text"])

        for i, chunk_text in enumerate(raw_chunks):
            alpha_chars = sum(1 for c in chunk_text if c.isalpha())
            total_chars = len(chunk_text)

            if total_chars == 0:
                continue

            alpha_ratio = alpha_chars / total_chars
            estimated_tokens = total_chars / 4

            if alpha_ratio < 0.60 or estimated_tokens < 50:
                print(f"  [DROPPED] Page {page['page_num']} chunk {i}: "
                      f"alpha_ratio={alpha_ratio:.2f}, est_tokens={estimated_tokens:.0f}")
                continue

            chunk_id = hashlib.md5(
                f"{page['source_file']}_p{page['page_num']}_c{i}".encode()
            ).hexdigest()[:12]

            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "page_num": page["page_num"],
                "chunk_index": i,
                "source_file": page["source_file"],
                "char_count": total_chars,
                "estimated_tokens": round(estimated_tokens),
            })

    return chunks


def save_chunks(chunks: list[dict], output_path: str) -> None:
    """Save chunks to a JSON file for inspection and reuse."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(chunks)} chunks to {output_path}")


def load_chunks(input_path: str) -> list[dict]:
    """Load previously saved chunks."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)