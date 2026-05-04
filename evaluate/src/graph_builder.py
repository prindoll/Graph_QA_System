"""Build a knowledge graph from HotPotQA documents using LLM-based extraction.

Pipeline:
  1. Deduplicate documents by source title (avoid redundant LLM calls).
  2. Extract (subject, predicate, object) triples via LLM with concurrent workers.
  3. Store triples in a NetworkX graph (entities = nodes, relationships = edges).
  4. Detect communities using the Louvain algorithm for global context summaries.
  5. Persist / load the graph to/from disk.
"""

from __future__ import annotations

import json
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import networkx as nx
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from config.settings import (
    GRAPH_DIR,
    GRAPH_EXTRACT_MODEL,
    GRAPH_EXTRACT_TEMPERATURE,
    OPENAI_API_KEY,
)

logger = logging.getLogger(__name__)

# ── Entity / relationship extraction ──────────────────────────────

EXTRACTION_SYSTEM_PROMPT = (
    "You are an expert knowledge-graph builder. Given a text passage, "
    "extract all meaningful entities and their relationships as structured triples.\n\n"
    "Return a JSON array of objects, each with keys:\n"
    '  "subject": entity name (string),\n'
    '  "predicate": relationship type (string, e.g. "is_a", "directed", "born_in", "nationality"),\n'
    '  "object": related entity name (string)\n\n'
    "Rules:\n"
    "- Normalise entity names to Title Case.\n"
    "- Use concise, consistent predicate labels (snake_case).\n"
    "- Extract ALL factual relationships, including nationality, profession, dates, locations.\n"
    "- If a person's nationality is mentioned, always include a triple like "
    '("Person Name", "nationality", "Country").\n'
    "- Return ONLY the JSON array, no markdown fences or extra text.\n"
    "- If no entities can be extracted, return an empty array []."
)


def _get_extraction_llm() -> ChatOpenAI:
    """Return a ChatOpenAI instance configured for entity extraction."""
    return ChatOpenAI(
        model=GRAPH_EXTRACT_MODEL,
        temperature=GRAPH_EXTRACT_TEMPERATURE,
        max_tokens=2048,
        openai_api_key=OPENAI_API_KEY,
    )


def extract_triples_from_text(
    text: str,
    llm: ChatOpenAI | None = None,
) -> list[dict[str, str]]:
    """Extract (subject, predicate, object) triples from a text passage using the LLM.

    Returns a list of dicts: [{"subject": ..., "predicate": ..., "object": ...}, ...]
    """
    if llm is None:
        llm = _get_extraction_llm()

    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", EXTRACTION_SYSTEM_PROMPT),
        ("human", "Text:\n{text}"),
    ])

    response = (prompt | llm).invoke({"text": text})
    raw = response.content.strip()

    # Robust extraction: find the JSON array in the response
    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    import re
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", raw, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1).strip()

    # Fallback: find first '[' to last ']'
    if not raw.startswith("["):
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            raw = raw[start : end + 1]

    try:
        triples = json.loads(raw)
        if not isinstance(triples, list):
            return []
        valid = []
        for t in triples:
            if all(k in t for k in ("subject", "predicate", "object")):
                valid.append({
                    "subject": str(t["subject"]).strip(),
                    "predicate": str(t["predicate"]).strip(),
                    "object": str(t["object"]).strip(),
                })
        return valid
    except json.JSONDecodeError:
        # Attempt to recover truncated JSON: find last complete object
        last_brace = raw.rfind("}")
        if last_brace > 0:
            truncated = raw[: last_brace + 1] + "]"
            try:
                triples = json.loads(truncated)
                if isinstance(triples, list):
                    valid = []
                    for t in triples:
                        if all(k in t for k in ("subject", "predicate", "object")):
                            valid.append({
                                "subject": str(t["subject"]).strip(),
                                "predicate": str(t["predicate"]).strip(),
                                "object": str(t["object"]).strip(),
                            })
                    if valid:
                        logger.debug("Recovered %d triples from truncated response.", len(valid))
                        return valid
            except json.JSONDecodeError:
                pass
        logger.warning("Failed to parse extraction result: %.200s", raw)
        return []


# ── Knowledge graph construction ──────────────────────────────────

def _deduplicate_documents(documents: list[Document]) -> list[Document]:
    """Keep only one document per unique source title (first occurrence)."""
    seen: set[str] = set()
    unique: list[Document] = []
    for doc in documents:
        source = doc.metadata.get("source", "")
        if source and source not in seen:
            seen.add(source)
            unique.append(doc)
        elif not source:
            unique.append(doc)
    logger.info("Deduplicated %d documents → %d unique sources.", len(documents), len(unique))
    return unique


def _extract_worker(
    doc: Document,
    llm: ChatOpenAI,
) -> tuple[Document, list[dict[str, str]]]:
    """Worker function for concurrent extraction (one doc at a time)."""
    triples = extract_triples_from_text(doc.page_content, llm=llm)
    return doc, triples


def _add_triples_to_graph(
    G: nx.DiGraph,
    triples: list[dict[str, str]],
    source: str,
    text: str,
):
    """Insert extracted triples into the graph (thread-safe if called from main thread)."""
    for t in triples:
        subj = t["subject"]
        obj = t["object"]
        pred = t["predicate"]

        # Add / update subject node
        if G.has_node(subj):
            G.nodes[subj]["doc_sources"].add(source)
            G.nodes[subj]["doc_texts"].append(text[:500])
        else:
            G.add_node(subj, label=subj, doc_sources={source}, doc_texts=[text[:500]])

        # Add / update object node
        if G.has_node(obj):
            G.nodes[obj]["doc_sources"].add(source)
        else:
            G.add_node(obj, label=obj, doc_sources={source}, doc_texts=[])

        # Add / update edge
        if G.has_edge(subj, obj):
            edge_data = G.edges[subj, obj]
            edge_data["predicates"].add(pred)
            edge_data["doc_sources"].add(source)
            edge_data["weight"] += 1
        else:
            G.add_edge(subj, obj, predicates={pred}, doc_sources={source}, weight=1)


def build_knowledge_graph(
    documents: list[Document],
    llm: ChatOpenAI | None = None,
    max_workers: int = 8,
    deduplicate: bool = True,
) -> nx.DiGraph:
    """Build a directed knowledge graph from a list of LangChain Documents.

    Optimisations:
      - Deduplicate documents by source title (avoids redundant LLM calls).
      - Concurrent LLM calls using ThreadPoolExecutor (default 8 workers).

    Each node stores:
      - label: entity name
      - doc_sources: set of source titles where it appeared
      - doc_texts: list of source text snippets

    Each edge stores:
      - predicate: relationship label
      - doc_sources: set of source titles
      - weight: count of times this relationship was extracted
    """
    if llm is None:
        llm = _get_extraction_llm()

    # Deduplicate to reduce LLM calls (e.g. 4970 docs → ~500 unique sources)
    if deduplicate:
        docs_to_process = _deduplicate_documents(documents)
    else:
        docs_to_process = documents

    G = nx.DiGraph()
    total_triples = 0

    # Concurrent extraction with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_extract_worker, doc, llm): doc
            for doc in docs_to_process
        }

        with tqdm(total=len(futures), desc="Extracting entities") as pbar:
            for future in as_completed(futures):
                try:
                    doc, triples = future.result()
                    source = doc.metadata.get("source", "unknown")
                    text = doc.page_content
                    total_triples += len(triples)
                    _add_triples_to_graph(G, triples, source, text)
                except Exception as e:
                    logger.warning("Extraction failed for a document: %s", e)
                pbar.update(1)

    logger.info(
        "Knowledge graph built: %d nodes, %d edges, %d triples from %d docs (of %d total).",
        G.number_of_nodes(), G.number_of_edges(), total_triples,
        len(docs_to_process), len(documents),
    )
    return G


# ── Community detection ───────────────────────────────────────────

def detect_communities(
    G: nx.DiGraph,
    resolution: float = 1.0,
) -> dict[str, int]:
    """Detect communities in the knowledge graph using the Louvain algorithm.

    Returns a dict mapping each node to its community id.
    """
    undirected = G.to_undirected()
    try:
        communities = nx.community.louvain_communities(undirected, resolution=resolution, seed=42)
    except AttributeError:
        # Fallback for older NetworkX versions
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(undirected))

    node_to_community: dict[str, int] = {}
    for cid, members in enumerate(communities):
        for node in members:
            node_to_community[node] = cid
            G.nodes[node]["community"] = cid

    logger.info("Detected %d communities across %d nodes.", len(communities), G.number_of_nodes())
    return node_to_community


def get_community_summaries(
    G: nx.DiGraph,
    node_to_community: dict[str, int],
) -> dict[int, dict[str, Any]]:
    """Build a summary for each community: member entities, internal edges, key relationships."""
    summaries: dict[int, dict[str, Any]] = {}

    for node, cid in node_to_community.items():
        if cid not in summaries:
            summaries[cid] = {"entities": [], "relationships": [], "sources": set()}
        summaries[cid]["entities"].append(node)
        summaries[cid]["sources"].update(G.nodes[node].get("doc_sources", set()))

    for u, v, data in G.edges(data=True):
        cu = node_to_community.get(u, -1)
        cv = node_to_community.get(v, -1)
        if cu == cv and cu >= 0:
            preds = ", ".join(data.get("predicates", set()))
            summaries[cu]["relationships"].append(f"{u} --[{preds}]--> {v}")

    # Convert sets to lists for serialisation
    for cid in summaries:
        summaries[cid]["sources"] = list(summaries[cid]["sources"])

    return summaries


# ── Persistence ───────────────────────────────────────────────────

def _prepare_for_pickle(G: nx.DiGraph) -> nx.DiGraph:
    """Convert sets in node/edge data to lists for pickle serialisation."""
    Gc = G.copy()
    for _, data in Gc.nodes(data=True):
        if "doc_sources" in data and isinstance(data["doc_sources"], set):
            data["doc_sources"] = list(data["doc_sources"])
    for _, _, data in Gc.edges(data=True):
        for key in ("predicates", "doc_sources"):
            if key in data and isinstance(data[key], set):
                data[key] = list(data[key])
    return Gc


def _restore_from_pickle(G: nx.DiGraph) -> nx.DiGraph:
    """Convert lists back to sets after loading from pickle."""
    for _, data in G.nodes(data=True):
        if "doc_sources" in data and isinstance(data["doc_sources"], list):
            data["doc_sources"] = set(data["doc_sources"])
    for _, _, data in G.edges(data=True):
        for key in ("predicates", "doc_sources"):
            if key in data and isinstance(data[key], list):
                data[key] = set(data[key])
    return G


def save_graph(G: nx.DiGraph, path: Path | str = GRAPH_DIR / "knowledge_graph.pkl"):
    """Persist the knowledge graph to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Gc = _prepare_for_pickle(G)
    with open(path, "wb") as f:
        pickle.dump(Gc, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved knowledge graph to %s (%d nodes, %d edges).", path, G.number_of_nodes(), G.number_of_edges())


def load_graph(path: Path | str = GRAPH_DIR / "knowledge_graph.pkl") -> nx.DiGraph:
    """Load a persisted knowledge graph from disk."""
    path = Path(path)
    with open(path, "rb") as f:
        G = pickle.load(f)
    G = _restore_from_pickle(G)
    logger.info("Loaded knowledge graph from %s (%d nodes, %d edges).", path, G.number_of_nodes(), G.number_of_edges())
    return G


def graph_stats(G: nx.DiGraph) -> dict[str, Any]:
    """Return basic statistics about the knowledge graph."""
    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
        "num_connected_components": nx.number_weakly_connected_components(G),
        "avg_degree": sum(dict(G.degree()).values()) / max(G.number_of_nodes(), 1),
        "top_nodes_by_degree": sorted(G.degree(), key=lambda x: x[1], reverse=True)[:10],
    }
