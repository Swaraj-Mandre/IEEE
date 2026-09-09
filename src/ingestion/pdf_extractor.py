import re
from pathlib import Path

import pymupdf4llm

# Diagram labels and figure captions survive PDF extraction as fragments like
# "O B A<br>20 m 0 m 40 m", which the SLM then reads as concepts. Drop them.
PICTURE_TEXT = re.compile(r"<!-- Start of picture text -->.*?<!-- End of picture text -->", re.S)
FIGURE_CAPTION = re.compile(r"^_Fig\..*?_\s*$", re.M)

BOILERPLATE = re.compile(r"NCERT not to be republished|©\s*NCERT", re.I)

# NCERT PDFs use typographic quotes and dashes. The SLM answers with ASCII ones,
# so "newton's second law" would not match the text unless both are normalised.
PUNCTUATION = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-",
    " ": " ", "…": "...",
})
PAGE_NUMBER = re.compile(r"^\s*\d+\s*$", re.M)
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.M)

# Numbered headings like "4.1.2 Distance travelled" mark real sections; the rest
# are sidebars such as "Note" or "Activity 4.1".
SECTION_HEADING = re.compile(r"^\*{0,2}(\d+(?:\.\d+)+)\s+(.+?)\*{0,2}$")


def clean_text(raw: str) -> str:
    """Strip diagram text, captions, page numbers and NCERT boilerplate from a page."""
    text = raw.translate(PUNCTUATION)
    text = PICTURE_TEXT.sub(" ", text)
    text = FIGURE_CAPTION.sub(" ", text)
    text = BOILERPLATE.sub("", text)
    text = PAGE_NUMBER.sub(" ", text)
    text = re.sub(r"<br>|<[^>]+>", " ", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"[*_`]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def find_section(page_markdown: str) -> str | None:
    """Return the last numbered section heading on a page, e.g. '4.1.2 Distance travelled'."""
    section = None
    for _, title in HEADING.findall(page_markdown):
        match = SECTION_HEADING.match(title.strip())
        if match:
            section = f"{match.group(1)} {match.group(2).strip()}"
    return section


def extract_pdf(pdf_path: str) -> list[dict]:
    """
    Extract a PDF page by page as Markdown, tagging each page with its section.

    Returns:
        List of dicts: [{page_num, text, section, char_count, is_valid, source_file}]
        is_valid = False means the page had no usable text (likely a figure page).
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        pages = pymupdf4llm.to_markdown(str(path), page_chunks=True, show_progress=False)
    except Exception as e:
        raise RuntimeError(f"Cannot open PDF {pdf_path}: {e}")

    print(f"  Extracting: {path.name} ({len(pages)} pages)")

    results = []
    current_section = None

    for page_num, page in enumerate(pages, start=1):
        markdown = page["text"]
        current_section = find_section(markdown) or current_section
        cleaned = clean_text(markdown)

        results.append({
            "page_num": page_num,
            "text": cleaned,
            "section": current_section,
            "char_count": len(cleaned),
            "is_valid": len(cleaned) >= 100,
            "source_file": path.name,
        })

    valid_pages = sum(1 for r in results if r["is_valid"])
    sections = len({r["section"] for r in results if r["section"]})
    print(f"  Done: {valid_pages}/{len(results)} pages have usable text, {sections} sections found")

    return results
