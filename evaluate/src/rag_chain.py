"""Construct the RAG chain using standard LangChain components."""

from __future__ import annotations

import logging

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from config.settings import (
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    OPENAI_API_KEY,
)

logger = logging.getLogger(__name__)


# --- LLM ---

def get_llm(
    model: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
    api_key: str = OPENAI_API_KEY,
) -> ChatOpenAI:
    """Initialise the ChatOpenAI LLM."""
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=api_key,
    )


# --- Prompt templates ---

# Used by create_retrieval_chain (variables: {context}, {input})
RETRIEVAL_CHAIN_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful QA assistant. Answer using only evidence in context.\n\n"
        "Output style (balanced):\n"
        "- Prefer concise answers (entity/number/date/short phrase) when possible.\n"
        "- If needed for clarity, provide one short sentence only.\n"
        "- Do not add long explanations, chain-of-thought, or unrelated details.\n"
        "- For yes/no questions, start with Yes or No.\n"
        "- If answer is missing in context, output exactly: I don't know\n"
        "- If multiple candidates exist, choose the one best supported by context.\n\n"
        "Context:\n{context}",
    ),
    ("human", "{input}"),
])

# Used by the manual LCEL chain (variables: {context}, {question})
LCEL_RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful QA assistant. Answer using only evidence in context.\n\n"
        "Output style (balanced):\n"
        "- Prefer concise answers (entity/number/date/short phrase) when possible.\n"
        "- If needed for clarity, provide one short sentence only.\n"
        "- Do not add long explanations, chain-of-thought, or unrelated details.\n"
        "- For yes/no questions, start with Yes or No.\n"
        "- If answer is missing in context, output exactly: I don't know\n"
        "- If multiple candidates exist, choose the one best supported by context.\n\n"
        "Context:\n{context}",
    ),
    ("human", "{question}"),
])

# --- Alternative: strict concise prompts (may cause "I don't know" on reasoning questions) ---
# RETRIEVAL_CHAIN_PROMPT = ChatPromptTemplate.from_messages([
#     (
#         "system",
#         "You are a helpful question-answering assistant. Answer the question "
#         "based ONLY on the provided context.\n\n"
#         "Rules:\n"
#         "- Answer as concisely as possible — ideally just the entity name, "
#         "number, date, or a short phrase.\n"
#         "- For yes/no questions, answer with just 'Yes' or 'No'.\n"
#         "- Do NOT add explanations, reasoning, or extra details unless the "
#         "question explicitly asks for them.\n"
#         "- If the answer is not found in the context, say 'I don't know'.\n\n"
#         "Context:\n{context}",
#     ),
#     ("human", "{input}"),
# ])
#
# LCEL_RAG_PROMPT = ChatPromptTemplate.from_messages([
#     (
#         "system",
#         "You are a helpful question-answering assistant. Answer the question "
#         "based ONLY on the context passages below.\n\n"
#         "Rules:\n"
#         "- Answer as concisely as possible — ideally just the entity name, "
#         "number, date, or a short phrase.\n"
#         "- For yes/no questions, answer with just 'Yes' or 'No'.\n"
#         "- Do NOT add explanations, reasoning, or extra details unless the "
#         "question explicitly asks for them.\n"
#         "- If the context is insufficient, say 'I don't know'.\n\n"
#         "Context:\n{context}",
#     ),
#     ("human", "{question}"),
# ])


# --- RAG chain: create_retrieval_chain approach ---

def build_retrieval_chain(retriever, llm=None):
    """Build a standard LangChain retrieval chain.

    Input:  {"input": "<question>"}
    Output: {"input": ..., "context": [...], "answer": "..."}
    """
    if llm is None:
        llm = get_llm()
    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=RETRIEVAL_CHAIN_PROMPT)
    rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=combine_docs_chain)
    logger.info("Built retrieval chain (create_retrieval_chain).")
    return rag_chain


# --- RAG chain: LCEL approach ---

def _format_docs(docs: list[Document]) -> str:
    """Join document contents into a single context string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_lcel_rag_chain(retriever, llm=None):
    """Build a RAG chain using LCEL (LangChain Expression Language).

    Input:  {"question": "<question>"}
    Output: string answer
    """
    if llm is None:
        llm = get_llm()
    rag_chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | LCEL_RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    logger.info("Built LCEL RAG chain.")
    return rag_chain


# --- Query helper ---

def ask(chain, question: str) -> str:
    """Send a question to the RAG chain and return the answer string."""
    try:
        result = chain.invoke({"input": question})
        if isinstance(result, dict):
            return result.get("answer", str(result))
        return str(result)
    except Exception:
        # LCEL chain accepts a raw string
        result = chain.invoke(question)
        return str(result)
