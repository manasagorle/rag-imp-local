# app.py
import streamlit as st
from retriever import Retriever
# from generator import answer_from_context   # use transformer backend
from generator_llama import answer_from_context  # or llama backend

retriever = Retriever()

st.title("Local RAG — Docs QA (no API key)")
q = st.text_input("Ask a question about the documents:")

if st.button("Ask") and q.strip():
    with st.spinner("Retrieving..."):
        hits = retriever.get_relevant(q, k=5)
    contexts = [h["text"] for h in hits]
    st.write("### Retrieved snippets")
    for h in hits:
        st.markdown(f"*Source:* {h['source']}, chunk {h['chunk_id']}")
        st.write(h['text'][:400] + ("..." if len(h['text'])>400 else ""))
    with st.spinner("Generating answer..."):
        ans = answer_from_context(q, contexts)
    st.write("### Answer")
    st.write(ans)
