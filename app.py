import streamlit as st
from rag.retriever import get_retriever

st.title("AstraRAG - Agentic RAG Chatbot")

query = st.text_input("Ask a question")

if query:

    retriever = get_retriever()

    docs = retriever.get_relevant_documents(query)

    context = "\n".join([doc.page_content for doc in docs])

    response = f"""
Context from documents:

{context}

User Question:
{query}

Answer:
Based on the retrieved information, here is the response.
"""

    st.write(response)