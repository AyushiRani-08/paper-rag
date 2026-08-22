# app.py
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from generation.generator import RAGGenerator
from ingestion.arxiv_fetcher import fetch_arxiv_paper
from indexing.vector_store import VectorStore

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Paper RAG Assistant",
    page_icon="📚",
    layout="wide",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.5rem; }
    .sub-text { font-size: 1.05rem; color: #666; margin-bottom: 2rem; }
    .citation-badge { background-color: #f0f2f6; padding: 2px 8px; border-radius: 4px; font-size: 0.85rem; font-weight: 600; }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_components():
  """Loads vector store and generator once and caches them in memory."""
  store = VectorStore(persist_directory="data/chroma_db")
  generator = RAGGenerator(retriever=None, model_name="openai/gpt-oss-20b")
  return store, generator


vector_store, rag_generator = get_components()

# --- SIDEBAR: Paper Management ---
with st.sidebar:
  st.title("📄 Paper Indexer")
  st.write("Add papers to your local vector store.")

  tab1, tab2 = st.tabs(["arXiv Fetcher", "PDF Upload"])

  with tab1:
    arxiv_id = st.text_input(
        "arXiv ID", placeholder="e.g. 1706.03762", key="arxiv_input"
    )
    if st.button("Fetch & Index Paper", use_container_width=True):
      if arxiv_id.strip():
        with st.spinner(f"Fetching arXiv:{arxiv_id}..."):
          try:
            paper_data = fetch_arxiv_paper(
                arxiv_id.strip(), out_dir="data/raw_pdfs"
            )
            pdf_path = paper_data["pdf_path"]

            st.info("Embedding and indexing into ChromaDB...")
            vector_store.add_pdf(pdf_path)
            st.success(f"✅ Successfully indexed `{arxiv_id}`!")
            st.rerun()
          except Exception as e:
            st.error(f"❌ Fetch/Index Failed: {e}")
            st.exception(e)
      else:
        st.warning("Please enter a valid arXiv ID.")

  with tab2:
    uploaded_file = st.file_uploader(
        "Upload custom PDF", type=["pdf"], key="pdf_uploader"
    )
    if uploaded_file and st.button(
        "Index Uploaded File", use_container_width=True
    ):
      raw_dir = Path("data/raw_pdfs")
      raw_dir.mkdir(parents=True, exist_ok=True)

      # Sanitize filename (removes spaces, parentheses, special characters)
      clean_name = "".join(
          c for c in uploaded_file.name if c.isalnum() or c in (".", "_", "-")
      )
      save_path = raw_dir / clean_name

      with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

      with st.spinner(f"Parsing, chunking, and embedding '{clean_name}'..."):
        try:
          added_count = vector_store.add_pdf(str(save_path))
          st.success(
              f"✅ Successfully indexed `{clean_name}` ({added_count} chunks"
              " added)!"
          )
          st.rerun()
        except Exception as e:
          st.error(f"❌ Indexing Failed: {e}")
          st.exception(e)

  st.divider()

  # Collection Stats
  total_chunks = vector_store.collection.count()
  st.metric(label="Total Chunks in Vector DB", value=total_chunks)

  # Filter query by specific paper
  filter_paper = st.text_input(
      "Optional: Filter by Paper ID",
      placeholder="e.g. 1706.03762",
      help="Leave empty to search across all indexed papers",
  )

  # Top-K Slider
  top_k = st.slider("Context Chunks (Top-K)", min_value=1, max_value=6, value=3)

# --- MAIN CHAT INTERFACE ---
st.markdown(
    '<div class="main-header">📚 Research Paper RAG Assistant</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-text">Ask deep technical questions grounded directly in'
    " your indexed research papers.</div>",
    unsafe_allow_html=True,
)

# Chat session state initialization
if "messages" not in st.session_state:
  st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
  with st.chat_message(msg["role"]):
    st.markdown(msg["content"])
    if "sources" in msg and msg["sources"]:
      with st.expander("🔍 View Retrieved Sources"):
        for i, src in enumerate(msg["sources"], start=1):
          meta = src["metadata"]
          st.markdown(
              f"**Source {i}:** Paper `{meta.get('paper_id')}` | Page"
              f" `{meta.get('page_number')}` | Similarity Score:"
              f" `{src.get('similarity_score')}`"
          )
          st.caption(src["text"])
          st.divider()

# User Input
if prompt := st.chat_input("Ask a question about the paper..."):
  st.session_state.messages.append({"role": "user", "content": prompt})
  with st.chat_message("user"):
    st.markdown(prompt)

  with st.chat_message("assistant"):
    with st.spinner("Retrieving relevant passages and reasoning..."):
      target_paper = filter_paper.strip() if filter_paper.strip() else None
      response_data = rag_generator.generate_answer(
          query=prompt, top_k=top_k, paper_id=target_paper
      )

      st.markdown(response_data["answer"])

      if response_data["sources"]:
        with st.expander("🔍 View Retrieved Sources"):
          for i, src in enumerate(response_data["sources"], start=1):
            meta = src["metadata"]
            st.markdown(
                f"**Source {i}:** Paper `{meta.get('paper_id')}` | Page"
                f" `{meta.get('page_number')}` | Similarity Score:"
                f" `{src.get('similarity_score')}`"
            )
            st.caption(src["text"])
            st.divider()

  st.session_state.messages.append({
      "role": "assistant",
      "content": response_data["answer"],
      "sources": response_data["sources"],
  })