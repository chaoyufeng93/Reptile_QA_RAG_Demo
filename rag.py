import os
import getpass
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from data_prep import load_doc
from settings import RAGConfig

load_dotenv()
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("enter your key: ")

def split_docs(
    doc: Document,
    separators: list = ["\n\n", "\n", ".", " ", ""],
    chunk_size: int = 1200,
    overlap: int = 100
):
    splitter = RecursiveCharacterTextSplitter(
        separators = separators,
        chunk_size = chunk_size,
        chunk_overlap = overlap
    )
    res = splitter.split_documents(doc)
    return res

def create_vec_store(
    embedding: str = "text-embedding-3-small"
):
    doc = load_doc()
    split_doc = split_docs(
        doc = doc,
        separators = RAGConfig.splitter.separators,
        chunk_size = RAGConfig.splitter.chunk_size,
        overlap = RAGConfig.splitter.overlap
    )
    embed_model = OpenAIEmbeddings(
        model = embedding
    )
    vec_store = FAISS.from_documents(
        split_doc,
        embedding = embed_model,
    )
    return vec_store

def load_retriever(
    path: str = "./sources/faiss_index",
    model: str = "text-embedding-3-small"
):
    # Embedding vectors
    vec_store = FAISS.load_local(
        path,
        OpenAIEmbeddings(model = model),
        allow_dangerous_deserialization=True
    )
    vec_retr = vec_store.as_retriever(
        search_type = RAGConfig.retriever.search_type,
        search_kwargs = RAGConfig.retriever.search_kwargs
    )
    # BM25
    doc = load_doc()
    split_doc = split_docs(
        doc = doc
    )
    bm_retr = BM25Retriever.from_documents(split_doc)
    bm_retr.k = RAGConfig.bm25.k
    return vec_retr, bm_retr

if __name__ == "__main__":
    query = "ackie monitor的温度要求是"
    create = True

    if create:
        vec_store = create_vec_store(
            embedding=RAGConfig.embedding.model
        )
        vec_store.save_local("./sources/faiss_index")

    else:
        vec_store = FAISS.load_local(
            "./sources/faiss_index",
            OpenAIEmbeddings(model = RAGConfig.embedding.model),
            allow_dangerous_deserialization=True
        )

        retriever = vec_store.as_retriever(
            search_type = RAGConfig.retriever.search_type,
            search_kwargs = RAGConfig.retriever.search_kwargs
        )
        res = retriever.invoke(query)
        for i in range(len(res)):
            print(f"query_{i}:\n {res[i].page_content}")
    