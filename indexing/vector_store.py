# indexing/vector_store.py
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import chromadb
from ingestion.pdf_parser import chunk_pdf
from indexing.embedder import Embedder


class VectorStore:

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "papers",
    ):
        self.persist_dir = Path(persist_directory)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize persistent client
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine"
            }
        )
        self.embedder = Embedder()

    def add_pdf(self, pdf_path: str):
        """Chunks, embeds, and stores a PDF with its metadata."""
        path_obj = Path(pdf_path)
        paper_id = path_obj.stem

        print(f"1. Chunking {pdf_path}...")
        chunks = chunk_pdf(pdf_path, chunk_size=500, overlap_pct=0.10)
        if not chunks:
            print("No chunks generated.")
            return

        print("2. Generating embeddings...")
        embedded_chunks = self.embedder.embed_chunks(chunks)

        print("3. Storing in ChromaDB...")
        ids = [f"{paper_id}_chunk_{c['chunk_id']}" for c in embedded_chunks]
        documents = [c["text"] for c in embedded_chunks]
        embeddings = [c["embedding"].tolist() for c in embedded_chunks]
        metadatas = [
            {
                "paper_id": paper_id,
                "page_number": c["page_number"],
                "word_count": c["word_count"],
            }
            for c in embedded_chunks
        ]

        # Upsert documents and vector records
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(
            f"✅ Successfully stored {len(embedded_chunks)} chunks for {paper_id}."
        )


if __name__ == "__main__":
    store = VectorStore()
    store.add_pdf("data/raw_pdfs/1706.03762.pdf")