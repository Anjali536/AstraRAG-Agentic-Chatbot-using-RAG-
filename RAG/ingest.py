from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os

DATA_PATH = "data/documents"
DB_PATH = "vector_db"

def ingest_documents():
    docs = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()

    db = FAISS.from_documents(chunks, embeddings)

    db.save_local(DB_PATH)

    print("Vector DB created!")

if _name_ == "_main_":
    ingest_documents()