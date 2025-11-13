#!/usr/bin/env python
"""
GraphRAG - Command Line Interface
"""
import asyncio
import sys
from pathlib import Path
from typing import Optional

from src.core.graphrag import GraphRAG
from src.utils.pdf_processor import PDFProcessor
from src.graph.kg_builder import KnowledgeGraphBuilder
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


async def extract_pages(pdf_path: str, start_page: int, end_page: int, clear: bool = False) -> dict:
    """Extract specific page range from PDF (fast extraction for testing)"""
    try:
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            return {"status": "error", "message": f"File not found: {pdf_path}"}
        
        rag = GraphRAG()
        processor = PDFProcessor()
        kg_builder = KnowledgeGraphBuilder(save_intermediates=True, output_dir="data/processing")
        
        if clear:
            await rag.graph_manager.clear()
        
        logger.info(f"Processing: {pdf_file.name}")
        page_count = processor.get_pdf_page_count(str(pdf_file))
        logger.info(f"Total pages: {page_count}")
        
        # Validate page range
        if start_page < 0 or end_page > page_count or start_page >= end_page:
            return {
                "status": "error", 
                "message": f"Invalid page range: {start_page}-{end_page} (valid: 0-{page_count-1})"
            }
        
        logger.info(f"Extracting pages {start_page+1}-{end_page} ({end_page - start_page} pages)...")
        
        # Extract the page range
        batch_text = processor.extract_pdf_batch(str(pdf_file), start_page, end_page)
        
        if not batch_text:
            return {"status": "error", "message": "No text extracted from page range"}
        
        # Build knowledge graph for this range
        docs = [{
            "id": f"{pdf_file.stem}_p{start_page+1}_{end_page}",
            "content": batch_text,
            "metadata": {
                "pages": f"{start_page+1}-{end_page}",
                "file": pdf_file.name
            }
        }]
        
        result = await kg_builder.build_and_persist(docs, rag.graph_manager)
        
        stats = await rag.graph_manager.get_stats()
        return {
            "status": "success",
            "file": pdf_file.name,
            "pages_extracted": f"{start_page+1}-{end_page}",
            "page_count": end_page - start_page,
            "nodes_extracted": result.get("nodes_added", 0),
            "edges_extracted": result.get("edges_added", 0),
            "total_nodes": stats.get("nodes", 0),
            "total_edges": stats.get("edges", 0)
        }
    
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return {"status": "error", "message": str(e)}


async def process_pdf(pdf_path: str, batch_size: Optional[int] = None, clear: bool = False) -> dict:
    """Process PDF and extract knowledge graph (using LLM extraction)"""
    try:
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            return {"status": "error", "message": f"File not found: {pdf_path}"}
        
        rag = GraphRAG()
        processor = PDFProcessor()
        kg_builder = KnowledgeGraphBuilder(save_intermediates=True, output_dir="data/processing")
        
        if clear:
            await rag.graph_manager.clear()
        
        logger.info(f"Processing: {pdf_file.name}")
        page_count = processor.get_pdf_page_count(str(pdf_file))
        logger.info(f"Total pages: {page_count}")
        
        total_nodes = 0
        total_edges = 0
        
        if batch_size and page_count > batch_size:
            logger.info(f"Processing in {batch_size}-page batches...")
            ranges = processor.generate_batch_ranges(str(pdf_file), batch_size)
            
            for i, (start, end) in enumerate(ranges, 1):
                logger.info(f"Batch {i}/{len(ranges)}: Pages {start+1}-{end}")
                batch_text = processor.extract_pdf_batch(str(pdf_file), start, end)
                
                if batch_text:
                    docs = [{
                        "id": f"{pdf_file.stem}_b{i}",
                        "content": batch_text,
                        "metadata": {"batch": i, "pages": f"{start+1}-{end}"}
                    }]
                    result = await kg_builder.build_and_persist(docs, rag.graph_manager)
                    total_nodes += result.get("nodes_added", 0)
                    total_edges += result.get("edges_added", 0)
        else:
            logger.info("Processing entire PDF...")
            text = processor.extract_pdf(str(pdf_file))
            
            if text:
                docs = [{
                    "id": pdf_file.stem,
                    "content": text,
                    "metadata": {"file": pdf_file.name}
                }]
                result = await kg_builder.build_and_persist(docs, rag.graph_manager)
                total_nodes = result.get("nodes_added", 0)
                total_edges = result.get("edges_added", 0)
        
        stats = await rag.graph_manager.get_stats()
        return {
            "status": "success",
            "file": pdf_file.name,
            "pages": page_count,
            "nodes_extracted": total_nodes,
            "edges_extracted": total_edges,
            "total_nodes": stats.get("nodes", 0),
            "total_edges": stats.get("edges", 0)
        }
    
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return {"status": "error", "message": str(e)}


async def query(text: str, top_k: int = 5) -> dict:
    """Query knowledge graph"""
    try:
        rag = GraphRAG()
        result = await rag.query(text, top_k=top_k)
        return {
            "status": "success",
            "query": text,
            "answer": result.get("answer", ""),
            "context": result.get("context", [])[:top_k]
        }
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"status": "error", "message": str(e)}


async def stats() -> dict:
    """Get graph statistics"""
    try:
        rag = GraphRAG()
        s = await rag.graph_manager.get_stats()
        return {"status": "success", "nodes": s.get("nodes", 0), "edges": s.get("edges", 0)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def clear_db() -> dict:
    """Clear knowledge graph"""
    try:
        rag = GraphRAG()
        await rag.graph_manager.clear()
        return {"status": "success", "message": "Knowledge graph cleared"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def show_help():
    """Show help message"""
    print("""
Usage:
  python main.py extract <pdf>                      Extract entire PDF
  python main.py extract <pdf> --batch 50           Process in 50-page batches
  python main.py extract <pdf> --batch 50 --clear   Clear & re-extract
  python main.py extract-pages <pdf> 0 5            Extract pages 1-5 (0-indexed)
  python main.py extract-pages <pdf> 10 15 --clear  Extract pages 11-15, clear first
  python main.py query "<question>"                 Query knowledge graph
  python main.py query "<question>" --top-k 10      Get more results
  python main.py stats                              Show graph statistics
  python main.py clear                              Clear knowledge graph

Examples:
  python main.py extract document.pdf
  python main.py extract book.pdf --batch 50
  python main.py extract-pages book.pdf 0 5         # Extract first 5 pages
  python main.py extract-pages book.pdf 20 25       # Extract pages 21-25
  python main.py query "What is Binary Search?"
  python main.py stats
""")


async def main():
    """Main CLI entry point"""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == "extract" and len(sys.argv) > 2:
        pdf_path = sys.argv[2]
        batch_size = None
        clear = False
        
        for i, arg in enumerate(sys.argv[3:], 3):
            if arg == "--batch" and i + 1 < len(sys.argv):
                batch_size = int(sys.argv[i + 1])
            elif arg == "--clear":
                clear = True
        
        result = await process_pdf(pdf_path, batch_size, clear)
        print("\n" + "="*60)
        if result["status"] == "success":
            print(f"SUCCESS")
            print(f"  File: {result['file']}")
            print(f"  Pages: {result['pages']}")
            print(f"  Extracted - Nodes: {result['nodes_extracted']}, Edges: {result['edges_extracted']}")
            print(f"  Total Graph - Nodes: {result['total_nodes']}, Edges: {result['total_edges']}")
        else:
            print(f"ERROR: {result['message']}")
        print("="*60)
    
    elif command == "extract-pages" and len(sys.argv) > 4:
        pdf_path = sys.argv[2]
        try:
            start_page = int(sys.argv[3])
            end_page = int(sys.argv[4])
        except ValueError:
            print("ERROR: start_page and end_page must be integers (0-indexed)")
            return
        
        clear = False
        for arg in sys.argv[5:]:
            if arg == "--clear":
                clear = True
        
        result = await extract_pages(pdf_path, start_page, end_page, clear)
        print("\n" + "="*60)
        if result["status"] == "success":
            print(f"SUCCESS")
            print(f"  File: {result['file']}")
            print(f"  Pages extracted: {result['pages_extracted']} ({result['page_count']} pages)")
            print(f"  Extracted - Nodes: {result['nodes_extracted']}, Edges: {result['edges_extracted']}")
            print(f"  Total Graph - Nodes: {result['total_nodes']}, Edges: {result['total_edges']}")
        else:
            print(f"ERROR: {result['message']}")
        print("="*60)
    
    elif command == "query" and len(sys.argv) > 2:
        query_text = sys.argv[2]
        top_k = 5
        
        for i, arg in enumerate(sys.argv[3:], 3):
            if arg == "--top-k" and i + 1 < len(sys.argv):
                top_k = int(sys.argv[i + 1])
        
        result = await query(query_text, top_k)
        print("\n" + "="*60)
        if result["status"] == "success":
            print(f"Q: {result['query']}")
            print(f"\nA: {result['answer']}")
            if result['context']:
                print(f"\nContext ({len(result['context'])} items):")
                for i, ctx in enumerate(result['context'], 1):
                    # Show more context (300 chars instead of 80)
                    if len(ctx) > 300:
                        print(f"  {i}. {ctx[:300]}...")
                    else:
                        print(f"  {i}. {ctx}")
        else:
            print(f"ERROR: {result['message']}")
        print("="*60)
    
    elif command == "stats":
        result = await stats()
        print("\n" + "="*60)
        if result["status"] == "success":
            print(f"Graph Statistics:")
            print(f"  Nodes: {result['nodes']}")
            print(f"  Edges: {result['edges']}")
        else:
            print(f"ERROR: {result['message']}")
        print("="*60)
    
    elif command == "clear":
        result = await clear_db()
        print("\n" + "="*60)
        if result["status"] == "success":
            print(f"{result['message']}")
        else:
            print(f"ERROR: {result['message']}")
        print("="*60)
    
    else:
        print(f"Unknown command: {command}")
        show_help()


if __name__ == "__main__":
    asyncio.run(main())
