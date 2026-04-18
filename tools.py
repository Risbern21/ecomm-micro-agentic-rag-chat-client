from langchain.tools import tool

from config import URLS
from ingestion import load_documents, split_documents
from retriever import build_retriever


def make_retriever_tool(retriever):
    """Return a LangChain tool"""

    @tool
    def retrieve(query: str) -> str:
        """Search and return information about the query"""
        docs = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])

    return retrieve


_docs = load_documents(urls=URLS)
_doc_splits = split_documents(docs=_docs)
_retriever = build_retriever(doc_splits=_doc_splits)
retriever_tool = make_retriever_tool(_retriever)
