"""Standalone evaluation script for the RAG pipeline.

Usage:
  python evaluate.py                                  # Basic evaluation
  python evaluate.py --use-llm-eval                   # + LLM-based metrics
  python evaluate.py --eval-size 100 --chain-type lcel
  python evaluate.py --analyze-only data/eval_detail_*.csv

Results are saved to data/:
  eval_detail_<ts>.csv, eval_summary_<ts>.json,
  eval_by_type_<ts>.csv, eval_by_level_<ts>.csv, eval_errors_<ts>.csv
"""

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
logger = logging.getLogger(__name__)
console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG Pipeline on HotPotQA")
    parser.add_argument("--eval-size", type=int, default=None, help="Evaluation sample count")
    parser.add_argument("--chain-type", choices=["retrieval", "lcel"], default="retrieval",
                        help="RAG chain type")
    parser.add_argument("--use-llm-eval", action="store_true",
                        help="Enable LLM-based metrics (correctness, faithfulness, relevancy, similarity)")
    parser.add_argument("--analyze-only", type=str, default=None,
                        help="Re-analyze an existing CSV result file (skip RAG execution)")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="HotPotQA sample count to load")
    return parser.parse_args()


def display_summary_table(df, use_llm_eval: bool = False):
    """Print summary metric tables (text, retrieval, LLM)."""
    # Text metrics
    table1 = Table(title="Text Metrics", show_header=True, header_style="bold cyan")
    table1.add_column("Metric", style="cyan", min_width=25)
    table1.add_column("Mean", style="green", justify="right")
    table1.add_column("Std", style="yellow", justify="right")
    table1.add_column("Min", style="red", justify="right")
    table1.add_column("Max", style="bright_green", justify="right")

    for label, col in [
        ("Exact Match", "exact_match"), ("Token Precision", "token_precision"),
        ("Token Recall", "token_recall"), ("F1 Score", "f1_score"),
        ("ROUGE-1", "rouge1"), ("ROUGE-2", "rouge2"),
        ("ROUGE-L", "rougeL"), ("BLEU", "bleu"),
    ]:
        if col in df.columns:
            table1.add_row(label, f"{df[col].mean():.4f}", f"{df[col].std():.4f}",
                           f"{df[col].min():.4f}", f"{df[col].max():.4f}")
    console.print(table1)

    # Retrieval metrics
    table2 = Table(title="Retrieval Metrics", show_header=True, header_style="bold cyan")
    table2.add_column("Metric", style="cyan", min_width=25)
    table2.add_column("Mean", style="green", justify="right")
    table2.add_column("Std", style="yellow", justify="right")
    table2.add_column("Min", style="red", justify="right")
    table2.add_column("Max", style="bright_green", justify="right")

    for label, col in [
        ("Retrieval Recall", "retrieval_recall"), ("Retrieval Precision", "retrieval_precision"),
        ("Retrieval F1", "retrieval_f1"), ("MRR", "mrr"), ("Hit Rate@5", "hit_rate_at_5"),
    ]:
        if col in df.columns:
            table2.add_row(label, f"{df[col].mean():.4f}", f"{df[col].std():.4f}",
                           f"{df[col].min():.4f}", f"{df[col].max():.4f}")
    console.print(table2)

    # LLM metrics (optional)
    if use_llm_eval and "llm_correctness" in df.columns:
        table3 = Table(title="LLM Metrics (0-5)", show_header=True, header_style="bold cyan")
        table3.add_column("Metric", style="cyan", min_width=25)
        table3.add_column("Mean", style="green", justify="right")
        table3.add_column("Std", style="yellow", justify="right")
        for label, col in [
            ("Correctness", "llm_correctness"), ("Faithfulness", "llm_faithfulness"),
            ("Answer Relevancy", "llm_answer_relevancy"),
            ("Context Relevancy", "llm_context_relevancy"),
            ("Answer Similarity", "llm_answer_similarity"),
        ]:
            if col in df.columns:
                valid = df[df[col] >= 0][col]
                if not valid.empty:
                    table3.add_row(label, f"{valid.mean():.2f}", f"{valid.std():.2f}")
        console.print(table3)


def display_breakdown_tables(df):
    """Print breakdown tables by question type and difficulty level."""
    from src.evaluation import analyze_by_type, analyze_by_level

    by_type = analyze_by_type(df)
    if not by_type.empty:
        table = Table(title="By Question Type", show_header=True, header_style="bold magenta")
        table.add_column("Type", style="magenta")
        table.add_column("Count", style="white", justify="right")
        table.add_column("EM", style="green", justify="right")
        table.add_column("F1", style="green", justify="right")
        table.add_column("ROUGE-L", style="green", justify="right")
        table.add_column("Ret.Recall", style="cyan", justify="right")
        table.add_column("MRR", style="cyan", justify="right")
        for q_type in by_type.index:
            row = by_type.loc[q_type]
            table.add_row(
                str(q_type), str(int(row.get("count", 0))),
                f"{row.get('exact_match', 0):.4f}", f"{row.get('f1_score', 0):.4f}",
                f"{row.get('rougeL', 0):.4f}", f"{row.get('retrieval_recall', 0):.4f}",
                f"{row.get('mrr', 0):.4f}",
            )
        console.print(table)

    by_level = analyze_by_level(df)
    if not by_level.empty:
        table = Table(title="By Difficulty Level", show_header=True, header_style="bold magenta")
        table.add_column("Level", style="magenta")
        table.add_column("Count", style="white", justify="right")
        table.add_column("EM", style="green", justify="right")
        table.add_column("F1", style="green", justify="right")
        table.add_column("ROUGE-L", style="green", justify="right")
        table.add_column("Ret.Recall", style="cyan", justify="right")
        table.add_column("MRR", style="cyan", justify="right")
        for level in by_level.index:
            row = by_level.loc[level]
            table.add_row(
                str(level), str(int(row.get("count", 0))),
                f"{row.get('exact_match', 0):.4f}", f"{row.get('f1_score', 0):.4f}",
                f"{row.get('rougeL', 0):.4f}", f"{row.get('retrieval_recall', 0):.4f}",
                f"{row.get('mrr', 0):.4f}",
            )
        console.print(table)


def display_error_summary(df):
    """Print error analysis summary (questions with F1 < 0.3)."""
    from src.evaluation import error_analysis

    errors = error_analysis(df)
    if errors.empty:
        console.print("[green]OK[/] No questions with F1 < 0.3")
        return

    table = Table(title=f"Error Analysis (F1 < 0.3): {len(errors)} questions",
                  show_header=True, header_style="bold red")
    table.add_column("Error Type", style="red")
    table.add_column("Count", style="white", justify="right")
    table.add_column("Ratio", style="yellow", justify="right")

    if "error_type" in errors.columns:
        for error_type, count in errors["error_type"].value_counts().items():
            table.add_row(str(error_type), str(count), f"{count / len(errors) * 100:.1f}%")
    console.print(table)


def main():
    args = parse_args()

    console.print(Panel.fit(
        "[bold magenta]RAG Evaluation — HotPotQA[/]", border_style="magenta",
    ))

    try:
        # Analyze-only mode
        if args.analyze_only:
            import pandas as pd
            console.print(f"Loading results from: {args.analyze_only}")
            df = pd.read_csv(args.analyze_only)
            display_summary_table(df, use_llm_eval="llm_correctness" in df.columns)
            display_breakdown_tables(df)
            display_error_summary(df)
            return

        # Full evaluation
        from config.settings import EVAL_SAMPLE_SIZE, HOTPOTQA_SAMPLE_SIZE
        from src.data_loader import load_and_prepare
        from src.vectorstore import load_vectorstore, get_retriever
        from src.rag_chain import build_retrieval_chain, build_lcel_rag_chain
        from src.evaluation import evaluate_rag, save_results

        eval_size = args.eval_size or EVAL_SAMPLE_SIZE
        sample_size = args.sample_size or HOTPOTQA_SAMPLE_SIZE

        console.print(Panel("[bold cyan]1/4[/] Load HotPotQA (QA pairs)"))
        _, qa_pairs = load_and_prepare(sample_size=sample_size)
        console.print(f"[green]OK[/] Loaded {len(qa_pairs)} QA pairs.")

        console.print(Panel("[bold cyan]2/4[/] Load Vector Store + Retriever"))
        vectorstore = load_vectorstore()
        retriever = get_retriever(vectorstore)

        console.print(Panel("[bold cyan]3/4[/] Build RAG Chain"))
        if args.chain_type == "retrieval":
            chain = build_retrieval_chain(retriever)
        else:
            chain = build_lcel_rag_chain(retriever)

        console.print(Panel(
            f"[bold cyan]4/4[/] Evaluate ({eval_size} samples"
            f"{', + LLM eval' if args.use_llm_eval else ''})"
        ))
        df = evaluate_rag(
            rag_chain=chain, retriever=retriever, qa_pairs=qa_pairs,
            sample_size=eval_size, use_llm_eval=args.use_llm_eval,
        )

        console.print()
        display_summary_table(df, use_llm_eval=args.use_llm_eval)
        display_breakdown_tables(df)
        display_error_summary(df)

        console.print()
        save_results(df)
        console.print(Panel("[bold green]Evaluation complete.[/]", border_style="green"))

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        logger.exception("Evaluation error")
        sys.exit(1)


if __name__ == "__main__":
    main()
