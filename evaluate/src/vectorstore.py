"""Build and manage the Chroma vector store for the RAG pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    RETRIEVER_SEARCH_TYPE,
    RETRIEVER_TOP_K,
    VECTORSTORE_DIR,
)

logger = logging.getLogger(__name__)

COLLECTION_NAME = "hotpotqa"


def get_text_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """Return a configured RecursiveCharacterTextSplitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into smaller chunks."""
    splitter = get_text_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(documents)
    logger.info("Split %d documents into %d chunks.", len(documents), len(chunks))
    return chunks


def get_embedding_model(
    model: str = EMBEDDING_MODEL,
    api_key: str = OPENAI_API_KEY,
) -> OpenAIEmbeddings:
    """Initialise the OpenAI embedding model."""
    return OpenAIEmbeddings(model=model, openai_api_key=api_key)


def build_vectorstore(
    documents: list[Document],
    persist_directory: str | Path = VECTORSTORE_DIR,
    collection_name: str = COLLECTION_NAME,
    embedding_model: str = EMBEDDING_MODEL,
) -> Chroma:
    """Chunk documents, embed them, and persist to a Chroma vector store."""
    chunks = split_documents(documents)
    embeddings = get_embedding_model(model=embedding_model)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_directory),
    )
    logger.info("Built vector store: %d chunks saved to %s", len(chunks), persist_directory)
    return vectorstore


def load_vectorstore(
    persist_directory: str | Path = VECTORSTORE_DIR,
    collection_name: str = COLLECTION_NAME,
    embedding_model: str = EMBEDDING_MODEL,
) -> Chroma:
    """Load an existing persisted Chroma vector store."""
    embeddings = get_embedding_model(model=embedding_model)
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
    )
    logger.info("Loaded vector store from %s", persist_directory)
    return vectorstore


def get_retriever(
    vectorstore: Chroma,
    search_type: str = RETRIEVER_SEARCH_TYPE,
    top_k: int = RETRIEVER_TOP_K,
):
    """Create a LangChain retriever from the given vector store."""
    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": top_k},
    )
    logger.info("Retriever: search_type=%s, top_k=%d", search_type, top_k)
    return retriever
