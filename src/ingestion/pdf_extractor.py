import fitz #PyMuPDF
import re
from pathlib import Path 

def clean_text(raw: str) -> str:
    """
    Remove common PDF artifacts from extracted text.
    NCERT PDFs have headers, footers, and page numbers that
    pollute the content. This strips them out.
    """
    # Remove (page numbers)
    lines = raw.split("\n")
    lines = [l for l in lines if not re.fullmatch(r"\s*\d+\s*", l)]

    # remove Whitespace
    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip()

    # remove NCERT "typical" govern words
    text = re.sub(r"NCERT not to be republished", "", text, flags=re.IGNORECASE)
    text = re.sub(r"© NCERT", "", text, flags=re.IGNORECASE)

    return text

def extract_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from every page of a PDF.

    Returns:
        List of dicts: [{page_num, text, char_count, is_valid}]
        is_valid = False means the page had no usable text (scanned image).
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    results = []

    try:
        doc = fitz.open(str(path))
    except Exception as e:
        raise RuntimeError(f"Cannot open PDF {pdf_path}: {e}")

    print(f"  Extracting: {path.name} ({len(doc)} pages)")

    for page_num in range(len(doc)):
        page = doc[page_num]

        raw_text = page.get_text("text")
        cleaned = clean_text(raw_text)
        # below 100 will be identify as image page
        is_valid = len(cleaned) >= 100

        results.append({
            "page_num": page_num + 1,  
            "text": cleaned,
            "char_count": len(cleaned),
            "is_valid": is_valid,
            "source_file": path.name,
        })

    doc.close()

    valid_pages = sum(1 for r in results if r["is_valid"])
    print(f"  Done: {valid_pages}/{len(results)} pages have usable text")

    return results