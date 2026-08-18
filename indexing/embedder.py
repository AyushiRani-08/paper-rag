# indexing/embedder.py
import sys
from pathlib import Path

# Add project root to sys.path so we can import from ingestion/
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ingestion.pdf_parser import chunk_pdf
from sentence_transformers import SentenceTransformer


class Embedder:

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        print(f"Loading embedding model '{model_name}'...")
        self.model = SentenceTransformer(model_name)

    def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        """Takes raw text chunks and attaches their embedding vectors."""
        if not chunks:
            return []

        # Extract only text strings for batch vectorization
        texts = [chunk["text"] for chunk in chunks]

        # Batch encode vectors (normalize_embeddings=True optimizes for cosine similarity)
        embeddings = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=True
        )

        # Attach embedding vectors back to the chunk dictionaries
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i]

        return chunks

    def embed_query(self, query: str) -> list[float]:
        """Encodes a user query with BGE prompt formatting for asymmetric search."""
        formatted_query = f"Represent this sentence for searching relevant passages: {query}"
        return self.model.encode(formatted_query, normalize_embeddings=True)


if __name__ == "__main__":
    # Test path using your actual PDF file in data/raw_pdfs/
    sample_pdf = "data/raw_pdfs/1706.03761.pdf"

    print(f"1. Chunking PDF: {sample_pdf}")
    chunks = chunk_pdf(sample_pdf, chunk_size=500, overlap_pct=0.10)
    print(f"Generated {len(chunks)} chunks.")

    print("\n2. Generating embeddings...")
    embedder = Embedder()
    embedded_chunks = embedder.embed_chunks(chunks)

    print("\n--- Summary ---")
    print(f"Total embedded chunks: {len(embedded_chunks)}")
    print(f"Vector dimension: {len(embedded_chunks[0]['embedding'])}")
    print(f"Sample preview (first 5 vector values):")
    print(embedded_chunks[0]["embedding"][:5])