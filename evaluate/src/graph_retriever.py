"""Graph-aware retrieval for GraphRAG.

Retrieval strategy:
  1. Extract key entities from the user query using LLM.
  2. Fuzzy-match extracted entities to nodes in the knowledge graph.
  3. Traverse the graph up to *depth* hops to gather related entities and edge context.
  4. Optionally combine with vector-similarity results from Chroma.
  5. Return assembled context documents with graph provenance metadata.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import networkx as nx
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from config.settings import (
    GRAPH_EXTRACT_MODEL,
    GRAPH_EXTRACT_TEMPERATURE,
    GRAPH_TOP_K_ENTITIES,
    GRAPH_TRAVERSAL_DEPTH,
    OPENAI_API_KEY,
)

logger = logging.getLogger(__name__)


# ── Entity extraction from query ──────────────────────────────────

QUERY_ENTITY_PROMPT = (
    "You are an expert at identifying key entities in questions. "
    "Given the following question, extract ALL named entities "
    "(people, places, organisations, works of art, events, etc.).\n\n"
    "Return a JSON array of strings. Example: [\"Scott Derrickson\", \"Ed Wood\"]\n"
    "Return ONLY the JSON array, no markdown fences or extra text.\n"
    "If no entities are found, return []."
)


def _get_query_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=GRAPH_EXTRACT_MODEL,
        temperature=GRAPH_EXTRACT_TEMPERATURE,
        max_tokens=256,
        openai_api_key=OPENAI_API_KEY,
    )


def extract_query_entities(
    question: str,
    llm: ChatOpenAI | None = None,
) -> list[str]:
    """Extract named entities from a user question using LLM."""
    if llm is None:
        llm = _get_query_llm()

    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", QUERY_ENTITY_PROMPT),
        ("human", "{question}"),
    ])

    response = (prompt | llm).invoke({"question": question})
    raw = response.content.strip()

    # Strip markdown fences
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()

    try:
        entities = json.loads(raw)
        if isinstance(entities, list):
            return [str(e).strip() for e in entities if e]
    except json.JSONDecodeError:
        logger.warning("Failed to parse query entities: %.120s", raw)
    return []


# ── Graph node matching ───────────────────────────────────────────

def _normalise(s: str) -> str:
    """Lowercase and strip for fuzzy matching."""
    return s.lower().strip()


def match_entities_to_graph(
    entities: list[str],
    G: nx.DiGraph,
    top_k: int = GRAPH_TOP_K_ENTITIES,
) -> list[str]:
    """Match extracted entity names to graph node names (exact + substring match).

    Returns the matched node names (up to *top_k*).
    """
    node_names = list(G.nodes())
    node_names_lower = {_normalise(n): n for n in node_names}
    matched: list[str] = []

    for entity in entities:
        ent_lower = _normalise(entity)

        # Exact match
        if ent_lower in node_names_lower:
            matched.append(node_names_lower[ent_lower])
            continue

        # Substring match (entity in node name, or node name in entity)
        for nl, original in node_names_lower.items():
            if ent_lower in nl or nl in ent_lower:
                matched.append(original)
                break

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for m in matched:
        if m not in seen:
            seen.add(m)
            unique.append(m)

    return unique[:top_k]


# ── Graph traversal ───────────────────────────────────────────────

def traverse_graph(
    G: nx.DiGraph,
    seed_nodes: list[str],
    depth: int = GRAPH_TRAVERSAL_DEPTH,
) -> dict[str, Any]:
    """BFS traversal from seed nodes up to *depth* hops.

    Returns:
      {
        "nodes": [{"name": ..., "doc_sources": [...], "community": ...}, ...],
        "edges": [{"source": ..., "target": ..., "predicates": [...], "weight": ...}, ...],
        "subgraph_text": "formatted text summary of the subgraph",
      }
    """
    visited_nodes: set[str] = set()
    collected_edges: list[dict[str, Any]] = []
    frontier: set[str] = set(seed_nodes)

    for _ in range(depth):
        next_frontier: set[str] = set()
        for node in frontier:
            if node not in G:
                continue
            visited_nodes.add(node)
            # Outgoing edges
            for _, target, data in G.out_edges(node, data=True):
                collected_edges.append({
                    "source": node,
                    "target": target,
                    "predicates": list(data.get("predicates", set())),
                    "weight": data.get("weight", 1),
                })
                if target not in visited_nodes:
                    next_frontier.add(target)
            # Incoming edges
            for source, _, data in G.in_edges(node, data=True):
                collected_edges.append({
                    "source": source,
                    "target": node,
                    "predicates": list(data.get("predicates", set())),
                    "weight": data.get("weight", 1),
                })
                if source not in visited_nodes:
                    next_frontier.add(source)
        frontier = next_frontier

    # Deduplicate edges
    edge_set: set[tuple[str, str]] = set()
    unique_edges: list[dict[str, Any]] = []
    for e in collected_edges:
        key = (e["source"], e["target"])
        if key not in edge_set:
            edge_set.add(key)
            unique_edges.append(e)

    # Build node info
    node_info = []
    for n in visited_nodes:
        data = G.nodes.get(n, {})
        node_info.append({
            "name": n,
            "doc_sources": list(data.get("doc_sources", set())),
            "community": data.get("community", -1),
        })

    # Build a text summary of the subgraph
    lines = []
    for e in unique_edges:
        preds = ", ".join(e["predicates"])
        lines.append(f"{e['source']} --[{preds}]--> {e['target']}")
    subgraph_text = "\n".join(lines) if lines else "(no graph relationships found)"

    return {
        "nodes": node_info,
        "edges": unique_edges,
        "subgraph_text": subgraph_text,
    }


# ── Context assembly ──────────────────────────────────────────────

def gather_graph_context(
    G: nx.DiGraph,
    seed_nodes: list[str],
    depth: int = GRAPH_TRAVERSAL_DEPTH,
) -> tuple[str, list[Document]]:
    """Gather context from the graph for the given seed entities.

    Returns:
      - context_text: formatted string with graph relationships + node source texts
      - context_docs: list of synthetic Documents (for compatibility with evaluation)
    """
    traversal = traverse_graph(G, seed_nodes, depth=depth)

    # Build rich context text
    parts: list[str] = []

    # 1. Relationship triples
    parts.append("=== Graph Relationships ===")
    parts.append(traversal["subgraph_text"])

    # 2. Source text snippets from visited nodes
    parts.append("\n=== Related Document Excerpts ===")
    seen_texts: set[str] = set()
    for node_info in traversal["nodes"]:
        node_name = node_info["name"]
        node_data = G.nodes.get(node_name, {})
        for text in node_data.get("doc_texts", []):
            text_key = text[:200]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                parts.append(f"\n[{node_name}]\n{text}")

    context_text = "\n".join(parts)

    # Build synthetic Documents for evaluation framework compatibility
    context_docs: list[Document] = []
    for node_info in traversal["nodes"]:
        node_name = node_info["name"]
        node_data = G.nodes.get(node_name, {})
        for src in node_info.get("doc_sources", []):
            doc = Document(
                page_content=f"[{node_name}] Entity from source: {src}",
                metadata={
                    "source": src,
                    "graph_entity": node_name,
                    "community": node_info.get("community", -1),
                },
            )
            context_docs.append(doc)

    return context_text, context_docs


def graph_retrieve(
    question: str,
    G: nx.DiGraph,
    llm: ChatOpenAI | None = None,
    depth: int = GRAPH_TRAVERSAL_DEPTH,
    top_k_entities: int = GRAPH_TOP_K_ENTITIES,
) -> dict[str, Any]:
    """Full graph retrieval pipeline: extract entities → match → traverse → assemble.

    Returns dict with keys:
      - entities: extracted entity names
      - matched_nodes: nodes found in graph
      - context_text: assembled context string
      - context_docs: list of Documents
      - graph_info: traversal details (nodes, edges)
    """
    # Step 1: Extract entities from question
    entities = extract_query_entities(question, llm=llm)
    logger.debug("Extracted entities: %s", entities)

    # Step 2: Match to graph nodes
    matched = match_entities_to_graph(entities, G, top_k=top_k_entities)
    logger.debug("Matched graph nodes: %s", matched)

    # Step 3: Traverse and gather context
    if matched:
        context_text, context_docs = gather_graph_context(G, matched, depth=depth)
        traversal = traverse_graph(G, matched, depth=depth)
    else:
        context_text = "(No matching entities found in the knowledge graph)"
        context_docs = []
        traversal = {"nodes": [], "edges": [], "subgraph_text": ""}

    return {
        "entities": entities,
        "matched_nodes": matched,
        "context_text": context_text,
        "context_docs": context_docs,
        "graph_info": traversal,
    }
