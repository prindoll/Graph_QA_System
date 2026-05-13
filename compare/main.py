"""CLI for the HotPotQA RAG pipeline."""

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
    parser = argparse.ArgumentParser(description="RAG Pipeline for HotPotQA with LangChain")
    parser.add_argument("--mode", choices=["full", "index", "query", "evaluate"], default="full",
                        help="Run mode (default: full)")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Override HotPotQA sample count")
    parser.add_argument("--eval-size", type=int, default=None,
                        help="Number of samples for evaluation")
    parser.add_argument("--chain-type", choices=["retrieval", "lcel"], default="retrieval",
                        help="RAG chain type (default: retrieval)")
    parser.add_argument("--use-llm-eval", action="store_true",
                        help="Enable LLM-based evaluation metrics")
    return parser.parse_args()


def step_index(sample_size: int | None = None):
    """Load data and build the vector store."""
    from src.data_loader import load_and_prepare
    from src.vectorstore import build_vectorstore
    from config.settings import HOTPOTQA_SAMPLE_SIZE

    n = sample_size or HOTPOTQA_SAMPLE_SIZE
    console.print(Panel(f"[bold cyan]Step 1:[/] Load HotPotQA ({n} samples)"))
    documents, qa_pairs = load_and_prepare(sample_size=n)

    console.print(Panel("[bold cyan]Step 2:[/] Build Vector Store (Chroma)"))
    vectorstore = build_vectorstore(documents)
    console.print(f"[green]OK[/] Indexed {len(documents)} documents.")
    return vectorstore, qa_pairs


def step_query(chain_type: str = "retrieval"):
    """Load vector store and start interactive QA."""
    from src.vectorstore import load_vectorstore, get_retriever
    from src.rag_chain import build_retrieval_chain, build_lcel_rag_chain, ask

    console.print(Panel("[bold cyan]Step 3:[/] Load Vector Store + Build RAG Chain"))
    vectorstore = load_vectorstore()
    retriever = get_retriever(vectorstore)

    if chain_type == "retrieval":
        chain = build_retrieval_chain(retriever)
    else:
        chain = build_lcel_rag_chain(retriever)

    console.print(Panel("[bold cyan]Step 4:[/] Interactive QA"))
    console.print("[dim]Enter a question (type 'quit' to exit):[/dim]")
    while True:
        question = console.input("[bold yellow]Q: [/]")
        if question.strip().lower() in ("quit", "exit", "q"):
            break
        console.print(f"[bold green]A:[/] {ask(chain, question)}\n")
    return chain, retriever


def step_evaluate(
    chain=None, retriever=None, qa_pairs=None,
    eval_size: int | None = None,
    chain_type: str = "retrieval",
    use_llm_eval: bool = False,
):
    """Evaluate RAG performance."""
    from src.vectorstore import load_vectorstore, get_retriever
    from src.rag_chain import build_retrieval_chain, build_lcel_rag_chain
    from src.evaluation import evaluate_rag, save_results, analyze_by_type, analyze_by_level
    from config.settings import EVAL_SAMPLE_SIZE

    n = eval_size or EVAL_SAMPLE_SIZE
    console.print(Panel(
        f"[bold cyan]Evaluate:[/] {n} samples"
        f"{', + LLM eval' if use_llm_eval else ''}"
    ))

    # Reuse objects from the full pipeline when available.
    if retriever is None or chain is None:
        vectorstore = load_vectorstore()
        retriever = get_retriever(vectorstore)
        chain = (build_retrieval_chain(retriever) if chain_type == "retrieval"
                 else build_lcel_rag_chain(retriever))

    if qa_pairs is None:
        from src.data_loader import load_and_prepare
        _, qa_pairs = load_and_prepare()

    df = evaluate_rag(chain, retriever, qa_pairs, sample_size=n, use_llm_eval=use_llm_eval)

    table1 = Table(title="Text Metrics")
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

    table2 = Table(title="Retrieval Metrics")
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
        table3 = Table(title="LLM Metrics (0-5)")
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
        table_type = Table(title="By Question Type")
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
        table_level = Table(title="By Difficulty Level")
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

    paths = save_results(df)
    for key, path in paths.items():
        console.print(f"[green]OK[/] {key}: {path}")
    return df


def main():
    args = setup_args()

    console.print(Panel.fit(
        "[bold magenta]RAG Pipeline for HotPotQA — LangChain[/]",
        border_style="magenta",
    ))

    try:
        if args.mode == "index":
            step_index(sample_size=args.sample_size)

        elif args.mode == "query":
            step_query(chain_type=args.chain_type)

        elif args.mode == "evaluate":
            step_evaluate(
                eval_size=args.eval_size, chain_type=args.chain_type,
                use_llm_eval=args.use_llm_eval,
            )

        elif args.mode == "full":
            vectorstore, qa_pairs = step_index(sample_size=args.sample_size)

            from src.vectorstore import get_retriever
            from src.rag_chain import build_retrieval_chain, build_lcel_rag_chain

            retriever = get_retriever(vectorstore)
            chain = (build_retrieval_chain(retriever) if args.chain_type == "retrieval"
                     else build_lcel_rag_chain(retriever))

            step_evaluate(
                chain=chain, retriever=retriever, qa_pairs=qa_pairs,
                eval_size=args.eval_size, chain_type=args.chain_type,
                use_llm_eval=args.use_llm_eval,
            )

            console.print("\n[dim]Enter a question (type 'quit' to exit):[/dim]")
            from src.rag_chain import ask
            while True:
                question = console.input("[bold yellow]Q: [/]")
                if question.strip().lower() in ("quit", "exit", "q"):
                    break
                console.print(f"[bold green]A:[/] {ask(chain, question)}\n")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        logger.exception("Pipeline error")
        sys.exit(1)

    console.print(Panel("[bold green]Done.[/]", border_style="green"))


if __name__ == "__main__":
    main()
