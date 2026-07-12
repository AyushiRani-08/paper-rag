import arxiv
from pathlib import Path
from urllib.request import urlretrieve

def fetch_arxiv_paper(arxiv_id: str, out_dir: str = "data/raw_pdfs") -> dict:
    """Fetch a paper's metadata and PDF from arXiv by ID (e.g. '1706.03762')."""
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    result = next(client.results(search))

    out_path = Path(out_dir) / f"{arxiv_id}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(result.pdf_url, out_path)

    return {
        "arxiv_id": arxiv_id,
        "title": result.title,
        "authors": [a.name for a in result.authors],
        "abstract": result.summary,
        "published": str(result.published),
        "pdf_path": str(out_path),
    }