# ingestion/pdf_parser.py
import fitz  # PyMuPDF
from pathlib import Path


def extract_raw_text(pdf_path: str) -> list[dict]:
    """
    Extract raw text from a PDF, page by page.
    This is a placeholder — Day 2 replaces/augments this with GROBID
    for proper section-aware structure. Today's goal is just proving
    the extraction pipeline works end to end.

    Returns:
        list of {"page_number": int, "text": str}
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []
    doc = fitz.open(pdf_path)
    try:
        for i, page in enumerate(doc):
            text = page.get_text()
            pages.append({
                "page_number": i + 1,
                "text": text.strip(),
            })
    finally:
        doc.close()

    return pages


def extract_full_text(pdf_path: str) -> str:
    """Convenience wrapper: all pages joined into one string."""
    pages = extract_raw_text(pdf_path)
    return "\n\n".join(p["text"] for p in pages)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py <path_to_pdf>")
        sys.exit(1)

    pages = extract_raw_text(sys.argv[1])
    print(f"Extracted {len(pages)} pages")
    print("--- First page preview ---")
    print(pages[0]["text"][:300])