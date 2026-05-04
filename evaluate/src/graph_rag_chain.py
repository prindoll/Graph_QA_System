"""GraphRAG chain — combine graph-based retrieval with vector retrieval and LLM.

Two retrieval modes:
  - graph_only:  Use only knowledge-graph context.
  - hybrid:      Merge graph context + vector-store context (default).

The chain is invoked with {"input": question} and returns
{"input": ..., "answer": ..., "context": [...], "graph_context": ...}.
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config.settings import (
    GRAPH_TOP_K_ENTITIES,
    GRAPH_TRAVERSAL_DEPTH,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    OPENAI_API_KEY,
)
from src.graph_retriever import graph_retrieve

logger = logging.getLogger(__name__)


# ── Prompt template ───────────────────────────────────────────────

GRAPH_RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful QA assistant. Answer using only evidence from graph and documents.\n\n"
        "Output style (balanced):\n"
        "- Prefer concise answers (entity/number/date/short phrase) when possible.\n"
        "- If needed for clarity, provide one short sentence only.\n"
        "- Do not add long explanations, chain-of-thought, or unrelated details.\n"
        "- For yes/no questions, start with Yes or No.\n"
        "- If answer is missing, output exactly: I don't know\n"
        "- If multiple candidates exist, choose the best-supported one.\n\n"
        "Graph Context:\n{graph_context}\n\n"
        "Document Context:\n{doc_context}",
    ),
    ("human", "{input}"),
])


GRAPH_ONLY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful QA assistant. Answer using only evidence from graph context.\n\n"
        "Output style (balanced):\n"
        "- Prefer concise answers (entity/number/date/short phrase) when possible.\n"
        "- If needed for clarity, provide one short sentence only.\n"
        "- Do not add long explanations, chain-of-thought, or unrelated details.\n"
        "- For yes/no questions, start with Yes or No.\n"
        "- If answer is missing, output exactly: I don't know\n"
        "- If multiple candidates exist, choose the best-supported one.\n\n"
        "Graph Context:\n{graph_context}",
    ),
    ("human", "{input}"),
])


# ── LLM ───────────────────────────────────────────────────────────

def get_llm(
    model: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
    api_key: str = OPENAI_API_KEY,
) -> ChatOpenAI:
    """Initialise the ChatOpenAI LLM for GraphRAG."""
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=api_key,
    )


# ── GraphRAG chain class ─────────────────────────────────────────

class GraphRAGChain:
    """A callable chain that performs graph-based (or hybrid) retrieval + LLM generation.

    Designed to be used interchangeably with the standard RAG chain:
        result = chain.invoke({"input": question})
        # result["answer"], result["context"], result["graph_context"]
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        llm: ChatOpenAI | None = None,
        retriever=None,
        mode: str = "hybrid",
        traversal_depth: int = GRAPH_TRAVERSAL_DEPTH,
        top_k_entities: int = GRAPH_TOP_K_ENTITIES,
    ):
        """
        Args:
            graph: The knowledge graph (NetworkX DiGraph).
            llm: ChatOpenAI instance (created automatically if None).
            retriever: LangChain vector-store retriever (required for hybrid mode).
            mode: "hybrid" (graph + vector) or "graph_only".
            traversal_depth: Max hops for graph traversal.
            top_k_entities: Max entities to match from query.
        """
        self.graph = graph
        self.llm = llm or get_llm()
        self.retriever = retriever
        self.mode = mode
        self.traversal_depth = traversal_depth
        self.top_k_entities = top_k_entities

        self._query_llm = ChatOpenAI(
            model=LLM_MODEL, temperature=0, max_tokens=256, openai_api_key=OPENAI_API_KEY,
        )

        logger.info(
            "GraphRAGChain initialised (mode=%s, depth=%d, top_k=%d).",
            mode, traversal_depth, top_k_entities,
        )

    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run the GraphRAG pipeline.

        Args:
            inputs: {"input": "<question>"}

        Returns:
            {"input": ..., "answer": ..., "context": [...], "graph_context": "..."}
        """
        question = inputs.get("input", "")

        # 1. Graph retrieval
        graph_result = graph_retrieve(
            question=question,
            G=self.graph,
            llm=self._query_llm,
            depth=self.traversal_depth,
            top_k_entities=self.top_k_entities,
        )
        graph_context = graph_result["context_text"]
        graph_docs = graph_result["context_docs"]

        # 2. Vector retrieval (hybrid mode)
        vector_docs: list[Document] = []
        doc_context = ""
        if self.mode == "hybrid" and self.retriever is not None:
            vector_docs = self.retriever.invoke(question)
            doc_context = "\n\n---\n\n".join(doc.page_content for doc in vector_docs)

        # 3. Combine context docs
        all_docs = vector_docs + graph_docs

        # 4. Generate answer
        if self.mode == "hybrid":
            prompt = GRAPH_RAG_PROMPT
            chain = prompt | self.llm
            response = chain.invoke({
                "input": question,
                "graph_context": graph_context[:4000],
                "doc_context": doc_context[:4000],
            })
        else:
            prompt = GRAPH_ONLY_PROMPT
            chain = prompt | self.llm
            response = chain.invoke({
                "input": question,
                "graph_context": graph_context[:6000],
            })

        answer = response.content.strip()

        return {
            "input": question,
            "answer": answer,
            "context": all_docs,
            "graph_context": graph_context,
            "entities_extracted": graph_result["entities"],
            "matched_nodes": graph_result["matched_nodes"],
        }


# ── Convenience builders ──────────────────────────────────────────

def build_graph_rag_chain(
    graph: nx.DiGraph,
    retriever=None,
    mode: str = "hybrid",
    llm: ChatOpenAI | None = None,
) -> GraphRAGChain:
    """Build a GraphRAG chain (hybrid or graph-only)."""
    return GraphRAGChain(
        graph=graph,
        llm=llm,
        retriever=retriever,
        mode=mode,
    )


def ask(chain: GraphRAGChain, question: str) -> str:
    """Send a question to the GraphRAG chain and return the answer string."""
    try:
        result = chain.invoke({"input": question})
        if isinstance(result, dict):
            return result.get("answer", str(result))
        return str(result)
    except Exception as e:
        logger.error("GraphRAG query error: %s", e)
        return f"Error: {e}"
