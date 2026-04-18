from langchain_core.vectorstores import (InMemoryVectorStore,
                                         VectorStoreRetriever)
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def build_retriever(doc_splits) -> VectorStoreRetriever:
    """Create an in-memory vector store from document chunks and return a retriever."""
    vector_store = InMemoryVectorStore.from_documents(
        documents=doc_splits,
        embedding=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview"),
    )

    return vector_store.as_retriever()
