# app.py
import streamlit as st
from ingestion.arxiv_fetcher import fetch_arxiv_paper
from ingestion.pdf_parser import extract_raw_text

st.set_page_config(page_title="Paper-RAG", page_icon="📄")

st.title("📄 Paper-RAG")
st.caption("Scaffold is live. Full pipeline coming online day by day.")

st.divider()
st.subheader("Day 1 sanity check")
st.write("Fetch a paper from arXiv and preview extracted text — proves ingestion works end to end.")

arxiv_id = st.text_input("arXiv ID", value="1706.03762")

if st.button("Fetch & Extract"):
    with st.spinner("Fetching from arXiv..."):
        try:
            meta = fetch_arxiv_paper(arxiv_id)
            st.success(f"Fetched: {meta['title']}")
            st.json(meta)

            pages = extract_raw_text(meta["pdf_path"])
            st.subheader(f"Extracted {len(pages)} pages")
            st.text_area("Page 1 preview", pages[0]["text"][:1000], height=200)
        except Exception as e:
            st.error(f"Something went wrong: {e}")