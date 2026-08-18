#ChromaDB computes cosine distance and returns the top $K$ chunks with their text, page numbers, and similarity distances.
# retrieval/retriever.py
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from indexing.embedder import Embedder
from indexing.vector_store import VectorStore


class Retriever:

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "papers",
    ):
        self.vector_store = VectorStore(
            persist_directory=persist_directory, collection_name=collection_name
        )
        self.collection = self.vector_store.collection
        self.embedder = self.vector_store.embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        paper_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Searches ChromaDB for the top_k most relevant chunks.

        Args:
            query: The user question.
            top_k: Number of relevant chunks to return.
            paper_id: Optional filter to restrict search to a specific paper.
        """
        # 1. Embed query with BGE prompt formatting
        query_vector = self.embedder.embed_query(query).tolist()

        # 2. Set optional metadata filter
        where_filter = {"paper_id": paper_id} if paper_id else None

        # 3. Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # 4. Format outputs into a clean list of dictionaries
        retrieved_chunks = []
        if results["documents"] and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]
            ids = results["ids"][0]

            for i in range(len(docs)):
                retrieved_chunks.append(
                    {
                        "id": ids[i],
                        "text": docs[i],
                        "metadata": metas[i],
                        # Cosine distance: 0 = identical, 2 = opposite
                        # Cosine similarity approx = 1 - distance
                        "distance": distances[i],
                        "similarity_score": round(1.0 - distances[i], 4),
                    }
                )

        return retrieved_chunks


if __name__ == "__main__":
    retriever = Retriever()

    # Test Query against "Attention Is All You Need"
    test_query = "What is the architecture of Multi-Head Attention?"
    print(f"Query: '{test_query}'\n")

    results = retriever.retrieve(query=test_query, top_k=2)

    for rank, item in enumerate(results, start=1):
        print(
            f"--- Rank {rank} (Similarity Score: {item['similarity_score']}) ---"
        )
        print(f"Chunk ID : {item['id']}")
        print(f"Page     : {item['metadata']['page_number']}")
        print(f"Excerpt  : {item['text'][:250]}...\n")