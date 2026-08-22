
import os
from pathlib import Path
import sys
from typing import Any, Dict, List
from dotenv import load_dotenv
from groq import Groq


sys.path.append(str(Path(__file__).resolve().parent.parent))

from retrieval.retriever import Retriever


load_dotenv()


class RAGGenerator:

  def __init__(
      self,
      retriever: Retriever | None = None,
      model_name: str = "openai/gpt-oss-20b"
  ):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
      raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")

    self.retriever = retriever or Retriever()
    self.model_name = model_name
    self.client = Groq(api_key=api_key)

  def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Formats retrieved chunks with citations into a unified context block."""
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
      page = chunk["metadata"].get("page_number", "Unknown")
      paper_id = chunk["metadata"].get("paper_id", "Unknown")
      text = chunk["text"].strip()
      context_parts.append(
          f"[Source {i} | Paper: {paper_id} | Page: {page}]\n{text}"
      )
    return "\n\n".join(context_parts)

  def generate_answer(
      self, query: str, top_k: int = 3, paper_id: str | None = None
  ) -> Dict[str, Any]:
    """Retrieves relevant chunks and generates a grounded response with page citations."""
    chunks = self.retriever.retrieve(query=query, top_k=top_k, paper_id=paper_id)

    if not chunks:
      return {
          "answer": "No relevant documents found in the database.",
          "sources": [],
      }

    context = self.format_context(chunks)

    system_prompt = (
        "You are an expert academic research assistant.\n"
        "Rules:\n"
        "1. Answer the question using ONLY the provided CONTEXT.\n"
        "2. If the context does not contain the answer, explicitly state: 'I cannot find sufficient information in the indexed document.'\n"
        "3. Always cite specific page numbers using [Page X] notation whenever stating a fact."
    )

    user_prompt = f"""CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

    response = self.client.chat.completions.create(
        model=self.model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": chunks,
    }


if __name__ == "__main__":
  generator = RAGGenerator()

  test_query = (
      "How does Multi-Head Attention work and what are its key components?"
  )
  print(f"Query: {test_query}\n" + "=" * 60)

  result = generator.generate_answer(test_query, top_k=3)

  print("\n--- Generated Answer ---\n")
  print(result["answer"])

  print("\n--- Retrieved Sources ---")
  for src in result["sources"]:
    print(
        f"• Page {src['metadata']['page_number']} (Score: {src['similarity_score']})"
    )