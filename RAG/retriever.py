from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

DB_PATH = "vector_db"

def get_retriever():

    embeddings = OpenAIEmbeddings()

    db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

    retriever = db.as_retriever(search_kwargs={"k":3})

    return retriever