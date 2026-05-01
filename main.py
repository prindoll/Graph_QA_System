#!/usr/bin/env python
"""GraphRAG command line interface."""

import argparse
import asyncio
from pathlib import Path
from typing import Optional

from src.core.graphrag import GraphRAG
from src.graph.kg_builder import KnowledgeGraphBuilder
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


async def index_pdf(
    pdf_path: str,
    start_page: int = 0,
    end_page: Optional[int] = None,
    clear: bool = False,
) -> dict:
    path = Path(pdf_path)
    if not path.exists():
        return {"status": "error", "message": f"File not found: {pdf_path}"}

    rag = GraphRAG()
    builder = KnowledgeGraphBuilder(
        save_intermediates=True,
        output_dir="data/processing",
        llm_manager=rag.llm_manager,
        embedding_manager=rag.embedding_manager,
    )
    result = await builder.index_pdf(
        str(path),
        rag.graph_manager,
        start_page=start_page,
        end_page=end_page,
        clear=clear,
    )
    stats = await rag.graph_manager.get_stats()
    result["total_nodes"] = stats.get("nodes", 0)
    result["total_edges"] = stats.get("edges", 0)
    return result


async def index_path(
    file_path: str,
    start_page: int = 0,
    end_page: Optional[int] = None,
    clear: bool = False,
) -> dict:
    path = Path(file_path)
    if not path.exists():
        return {"status": "error", "message": f"File not found: {file_path}"}

    rag = GraphRAG()
    builder = KnowledgeGraphBuilder(
        save_intermediates=True,
        output_dir="data/processing",
        llm_manager=rag.llm_manager,
        embedding_manager=rag.embedding_manager,
    )
    result = await builder.index_path(
        str(path),
        rag.graph_manager,
        start_page=start_page,
        end_page=end_page,
        clear=clear,
    )
    stats = await rag.graph_manager.get_stats()
    result["total_nodes"] = stats.get("nodes", 0)
    result["total_edges"] = stats.get("edges", 0)
    return result


async def query_graph(text: str, top_k: int = 5, mode: str = "auto", max_hops: int = 2) -> dict:
    rag = GraphRAG()
    result = await rag.query(
        text,
        top_k=top_k,
        retrieval_mode=mode,
        max_hops=max_hops,
        include_sources=True,
    )
    return {
        "status": "success",
        "query": text,
        "answer": result.get("answer", ""),
        "context": result.get("context", [])[:top_k],
        "retrieval_mode": result.get("retrieval_mode", mode),
    }


async def stats() -> dict:
    rag = GraphRAG()
    graph_stats = await rag.graph_manager.get_stats()
    return {"status": "success", **graph_stats}


async def clear_db() -> dict:
    rag = GraphRAG()
    await rag.graph_manager.clear()
    return {"status": "success", "message": "Knowledge graph cleared"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GraphRAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", aliases=["extract"], help="Index a PDF or Markdown file into Neo4j GraphRAG schema")
    index_parser.add_argument("path")
    index_parser.add_argument("--start-page", type=int, default=0)
    index_parser.add_argument("--end-page", type=int, default=None)
    index_parser.add_argument("--batch", type=int, default=None, help="Accepted for compatibility; indexing uses text-unit chunking.")
    index_parser.add_argument("--clear", action="store_true")

    extract_pages_parser = subparsers.add_parser("extract-pages", help="Index a specific 0-based page range")
    extract_pages_parser.add_argument("pdf")
    extract_pages_parser.add_argument("start_page", type=int)
    extract_pages_parser.add_argument("end_page", type=int)
    extract_pages_parser.add_argument("--clear", action="store_true")

    query_parser = subparsers.add_parser("query", help="Query the knowledge graph")
    query_parser.add_argument("text")
    query_parser.add_argument("--top-k", type=int, default=5)
    query_parser.add_argument("--mode", choices=["auto", "basic", "local", "global", "drift"], default="auto")
    query_parser.add_argument("--max-hops", type=int, default=2)

    subparsers.add_parser("stats", help="Show graph statistics")
    subparsers.add_parser("clear", help="Clear graph data")
    return parser


def print_index_result(result: dict) -> None:
    print("\n" + "=" * 60)
    if result.get("status") == "success":
        print("SUCCESS")
        print(f"  Documents: {result.get('documents', 0)}")
        print(f"  Text units: {result.get('text_units', 0)}")
        print(f"  Entities: {result.get('entities', 0)}")
        print(f"  Relationships: {result.get('relationships', 0)}")
        print(f"  Communities: {result.get('communities', 0)}")
        print(f"  Community reports: {result.get('community_reports', 0)}")
        print(f"  Persisted nodes: {result.get('nodes_added', 0)}")
        print(f"  Persisted edges: {result.get('edges_added', 0)}")
        print(f"  Total graph: {result.get('total_nodes', 0)} nodes, {result.get('total_edges', 0)} edges")
    else:
        print(f"ERROR: {result.get('message', 'Unknown error')}")
    print("=" * 60)


def print_query_result(result: dict) -> None:
    print("\n" + "=" * 60)
    if result.get("status") == "success":
        print(f"Mode: {result.get('retrieval_mode')}")
        print(f"Q: {result['query']}")
        print(f"\nA: {result['answer']}")
        if result.get("context"):
            print(f"\nContext ({len(result['context'])} items):")
            for index, item in enumerate(result["context"], 1):
                title = item.get("title") or item.get("id")
                text = item.get("text", "")
                preview = text[:300] + "..." if len(text) > 300 else text
                print(f"  {index}. [{item.get('type')}] {title}: {preview}")
    else:
        print(f"ERROR: {result.get('message', 'Unknown error')}")
    print("=" * 60)


async def main() -> None:
    args = build_parser().parse_args()

    if args.command in {"index", "extract"}:
        result = await index_path(args.path, start_page=args.start_page, end_page=args.end_page, clear=args.clear)
        print_index_result(result)
    elif args.command == "extract-pages":
        result = await index_pdf(args.pdf, start_page=args.start_page, end_page=args.end_page, clear=args.clear)
        print_index_result(result)
    elif args.command == "query":
        result = await query_graph(args.text, top_k=args.top_k, mode=args.mode, max_hops=args.max_hops)
        print_query_result(result)
    elif args.command == "stats":
        result = await stats()
        print("\n" + "=" * 60)
        if result.get("status") == "success":
            print(f"Nodes: {result.get('nodes', 0)}")
            print(f"Edges: {result.get('edges', 0)}")
            labels = result.get("labels", {})
            if labels:
                print("Labels:")
                for label, count in labels.items():
                    print(f"  {label}: {count}")
        else:
            print(f"ERROR: {result.get('message', 'Unknown error')}")
        print("=" * 60)
    elif args.command == "clear":
        result = await clear_db()
        print(result.get("message", result))


if __name__ == "__main__":
    asyncio.run(main())
