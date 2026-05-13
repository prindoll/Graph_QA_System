"""CLI for the HotPotQA GraphRAG pipeline."""

from __future__ import annotations

import argparse
import logging
import sys

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
# Keep third-party clients quiet.
for _noisy in ("httpx", "httpcore", "huggingface_hub", "openai"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
console = Console()


def setup_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="GraphRAG Pipeline for HotPotQA")
    parser.add_argument(
        "--mode", choices=["full", "build", "query", "evaluate"], default="full",
        help="Run mode (default: full)",
    )
    parser.add_argument(
        "--graph-mode", choices=["hybrid", "graph_only"], default="hybrid",
        help="GraphRAG retrieval mode: hybrid (graph+vector) or graph_only",
    )
    parser.add_argument("--sample-size", type=int, default=None, help="HotPotQA sample count")
    parser.add_argument("--eval-size", type=int, default=None, help="Evaluation sample count")
    parser.add_argument("--use-llm-eval", action="store_true", help="Enable LLM-based metrics")
    return parser.parse_args()


def step_build_graph(sample_size: int | None = None):
    """Load HotPotQA data, build vector store, and construct the knowledge graph."""
    from config.settings import HOTPOTQA_SAMPLE_SIZE
    from src.data_loader import load_and_prepare
    from src.vectorstore import build_vectorstore
    from src.graph_builder import build_knowledge_graph, detect_communities, save_graph, graph_stats

    n = sample_size or HOTPOTQA_SAMPLE_SIZE

    console.print(Panel(f"[bold cyan]Step 1:[/] Load HotPotQA ({n} samples)"))
    documents, qa_pairs = load_and_prepare(sample_size=n)
    console.print(f"[green]OK[/] {len(documents)} documents, {len(qa_pairs)} QA pairs.")

    console.print(Panel("[bold cyan]Step 2:[/] Build Vector Store (Chroma)"))
    vectorstore = build_vectorstore(documents)
    console.print(f"[green]OK[/] Indexed {len(documents)} documents.")

    console.print(Panel("[bold cyan]Step 3:[/] Build Knowledge Graph (LLM extraction)"))
    console.print("[dim]Extracting entities and relationships from documents...[/dim]")
    G = build_knowledge_graph(documents)

    console.print(Panel("[bold cyan]Step 4:[/] Detect Communities"))
    node_to_community = detect_communities(G)

    stats = graph_stats(G)
    table = Table(title="Knowledge Graph Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_row("Nodes", str(stats["num_nodes"]))
    table.add_row("Edges", str(stats["num_edges"]))
    table.add_row("Density", f"{stats['density']:.4f}")
    table.add_row("Weakly Connected Components", str(stats["num_connected_components"]))
    table.add_row("Avg Degree", f"{stats['avg_degree']:.2f}")
    table.add_row("Communities", str(len(set(node_to_community.values()))))
    console.print(table)

    console.print("[bold]Top 10 nodes by degree:[/]")
    for name, deg in stats["top_nodes_by_degree"]:
        console.print(f"  {name}: {deg}")

    save_graph(G)
    console.print("[green]OK[/] Knowledge graph saved.")

    return G, vectorstore, qa_pairs


def step_evaluate(
    G=None, retriever=None, qa_pairs=None,
    eval_size: int | None = None,
    graph_mode: str = "hybrid",
    use_llm_eval: bool = False,
):
    """Evaluate GraphRAG performance on HotPotQA."""
    from config.settings import EVAL_SAMPLE_SIZE
    from src.evaluation import evaluate_rag, save_results, analyze_by_type, analyze_by_level
    from src.graph_rag_chain import build_graph_rag_chain

    n = eval_size or EVAL_SAMPLE_SIZE

    console.print(Panel(
        f"[bold cyan]Evaluate GraphRAG:[/] {n} samples, mode={graph_mode}"
        f"{', + LLM eval' if use_llm_eval else ''}"
    ))

    # Reuse graph/vector objects from the full pipeline when available.
    if G is None:
        from src.graph_builder import load_graph
        console.print("[dim]Loading knowledge graph from disk...[/dim]")
        G = load_graph()

    if retriever is None and graph_mode == "hybrid":
        from src.vectorstore import load_vectorstore, get_retriever
        vectorstore = load_vectorstore()
        retriever = get_retriever(vectorstore)

    if qa_pairs is None:
        from src.data_loader import load_and_prepare
        _, qa_pairs = load_and_prepare()

    chain = build_graph_rag_chain(
        graph=G,
        retriever=retriever,
        mode=graph_mode,
    )

    df = evaluate_rag(chain, retriever, qa_pairs, sample_size=n, use_llm_eval=use_llm_eval)

    table1 = Table(title="GraphRAG — Text Metrics")
    table1.add_column("Metric", style="cyan")
    table1.add_column("Score", style="green", justify="right")
    for label, col in [
        ("Exact Match", "exact_match"), ("Token Precision", "token_precision"),
        ("Token Recall", "token_recall"), ("F1 Score", "f1_score"),
        ("ROUGE-1", "rouge1"), ("ROUGE-2", "rouge2"),
        ("ROUGE-L", "rougeL"), ("BLEU", "bleu"),
    ]:
        if col in df.columns:
            table1.add_row(label, f"{df[col].mean():.4f}")
    console.print(table1)

    table2 = Table(title="GraphRAG — Retrieval Metrics")
    table2.add_column("Metric", style="cyan")
    table2.add_column("Score", style="green", justify="right")
    for label, col in [
        ("Retrieval Recall", "retrieval_recall"), ("Retrieval Precision", "retrieval_precision"),
        ("Retrieval F1", "retrieval_f1"), ("MRR", "mrr"), ("Hit Rate@5", "hit_rate_at_5"),
    ]:
        if col in df.columns:
            table2.add_row(label, f"{df[col].mean():.4f}")
    table2.add_row("Samples", str(len(df)))
    console.print(table2)

    if use_llm_eval and "llm_correctness" in df.columns:
        table3 = Table(title="GraphRAG — LLM Metrics (0-5)")
        table3.add_column("Metric", style="cyan")
        table3.add_column("Score", style="green", justify="right")
        for label, col in [
            ("Correctness", "llm_correctness"), ("Faithfulness", "llm_faithfulness"),
            ("Answer Relevancy", "llm_answer_relevancy"),
            ("Context Relevancy", "llm_context_relevancy"),
            ("Answer Similarity", "llm_answer_similarity"),
        ]:
            if col in df.columns:
                valid = df[df[col] >= 0][col]
                if not valid.empty:
                    table3.add_row(label, f"{valid.mean():.2f} / 5.00")
        console.print(table3)

    by_type = analyze_by_type(df)
    if not by_type.empty:
        table_type = Table(title="GraphRAG — By Question Type")
        table_type.add_column("Type", style="magenta")
        table_type.add_column("Count", style="white", justify="right")
        table_type.add_column("EM", style="green", justify="right")
        table_type.add_column("F1", style="green", justify="right")
        table_type.add_column("ROUGE-L", style="green", justify="right")
        table_type.add_column("Ret.Recall", style="cyan", justify="right")
        table_type.add_column("MRR", style="cyan", justify="right")
        for q_type in by_type.index:
            row = by_type.loc[q_type]
            table_type.add_row(
                str(q_type),
                str(int(row.get("count", 0))),
                f"{row.get('exact_match', 0):.4f}",
                f"{row.get('f1_score', 0):.4f}",
                f"{row.get('rougeL', 0):.4f}",
                f"{row.get('retrieval_recall', 0):.4f}",
                f"{row.get('mrr', 0):.4f}",
            )
        console.print(table_type)

    by_level = analyze_by_level(df)
    if not by_level.empty:
        table_level = Table(title="GraphRAG — By Difficulty Level")
        table_level.add_column("Level", style="magenta")
        table_level.add_column("Count", style="white", justify="right")
        table_level.add_column("EM", style="green", justify="right")
        table_level.add_column("F1", style="green", justify="right")
        table_level.add_column("ROUGE-L", style="green", justify="right")
        table_level.add_column("Ret.Recall", style="cyan", justify="right")
        table_level.add_column("MRR", style="cyan", justify="right")
        for level in by_level.index:
            row = by_level.loc[level]
            table_level.add_row(
                str(level),
                str(int(row.get("count", 0))),
                f"{row.get('exact_match', 0):.4f}",
                f"{row.get('f1_score', 0):.4f}",
                f"{row.get('rougeL', 0):.4f}",
                f"{row.get('retrieval_recall', 0):.4f}",
                f"{row.get('mrr', 0):.4f}",
            )
        console.print(table_level)

    paths = save_results(df, prefix="graph_eval")
    for key, path in paths.items():
        console.print(f"[green]OK[/] {key}: {path}")

    return df


def step_query(G=None, retriever=None, graph_mode: str = "hybrid"):
    """Interactive QA with GraphRAG."""
    from src.graph_rag_chain import build_graph_rag_chain, ask

    if G is None:
        from src.graph_builder import load_graph
        G = load_graph()

    if retriever is None and graph_mode == "hybrid":
        from src.vectorstore import load_vectorstore, get_retriever
        vectorstore = load_vectorstore()
        retriever = get_retriever(vectorstore)

    chain = build_graph_rag_chain(graph=G, retriever=retriever, mode=graph_mode)

    console.print(Panel(f"[bold cyan]Interactive QA[/] (GraphRAG, mode={graph_mode})"))
    console.print("[dim]Enter a question (type 'quit' to exit):[/dim]")
    while True:
        question = console.input("[bold yellow]Q: [/]")
        if question.strip().lower() in ("quit", "exit", "q"):
            break

        result = chain.invoke({"input": question})
        console.print(f"[bold green]A:[/] {result.get('answer', '')}")

        # Print matched graph clues under the answer.
        entities = result.get("entities_extracted", [])
        matched = result.get("matched_nodes", [])
        if entities:
            console.print(f"[dim]Entities: {entities}[/dim]")
        if matched:
            console.print(f"[dim]Graph nodes: {matched}[/dim]")
        console.print()

    return chain


def main():
    args = setup_args()

    console.print(Panel.fit(
        "[bold magenta]GraphRAG Pipeline for HotPotQA[/]",
        border_style="magenta",
    ))

    try:
        if args.mode == "build":
            step_build_graph(sample_size=args.sample_size)

        elif args.mode == "query":
            step_query(graph_mode=args.graph_mode)

        elif args.mode == "evaluate":
            step_evaluate(
                eval_size=args.eval_size,
                graph_mode=args.graph_mode,
                use_llm_eval=args.use_llm_eval,
            )

        elif args.mode == "full":
            G, vectorstore, qa_pairs = step_build_graph(sample_size=args.sample_size)

            from src.vectorstore import get_retriever
            retriever = get_retriever(vectorstore)

            step_evaluate(
                G=G, retriever=retriever, qa_pairs=qa_pairs,
                eval_size=args.eval_size,
                graph_mode=args.graph_mode,
                use_llm_eval=args.use_llm_eval,
            )

            console.print("\n[dim]Enter a question (type 'quit' to exit):[/dim]")
            from src.graph_rag_chain import build_graph_rag_chain, ask as graph_ask
            chain = build_graph_rag_chain(graph=G, retriever=retriever, mode=args.graph_mode)
            while True:
                question = console.input("[bold yellow]Q: [/]")
                if question.strip().lower() in ("quit", "exit", "q"):
                    break
                console.print(f"[bold green]A:[/] {graph_ask(chain, question)}\n")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        logger.exception("GraphRAG pipeline error")
        sys.exit(1)

    console.print(Panel("[bold green]Done.[/]", border_style="green"))


if __name__ == "__main__":
    main()
