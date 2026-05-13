#!/usr/bin/env python
"""Compare plain RAG and GraphRAG on the Fusarium QA set with local LLMs."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import settings
from evaluate_fusarium import DEFAULT_INPUT, VALID_RETRIEVAL_MODES, evaluate_fusarium_qa


DEFAULT_MODELS = ("gemma2.5", "qwen3.5")
DEFAULT_COMPARE_OUTPUT_DIR = Path("outputs") / "comparisons"


async def compare_fusarium_rag_graphrag(
    models: list[str] | tuple[str, ...] = DEFAULT_MODELS,
    input_path: str | Path = DEFAULT_INPUT,
    output_dir: str | Path = DEFAULT_COMPARE_OUTPUT_DIR,
    ollama_base_url: str | None = None,
    limit: int | None = None,
    top_k: int = 5,
    rag_mode: str = "basic",
    graph_mode: str = "auto",
    max_hops: int = 2,
    include_sources: bool = True,
    continue_on_error: bool = True,
    progress: bool = True,
) -> dict[str, Any]:
    """Evaluate RAG and GraphRAG for each local model and write a comparison report."""
    if not models:
        raise ValueError("At least one model is required")
    if rag_mode not in VALID_RETRIEVAL_MODES:
        raise ValueError(f"rag_mode must be one of: {', '.join(VALID_RETRIEVAL_MODES)}")
    if graph_mode not in VALID_RETRIEVAL_MODES:
        raise ValueError(f"graph_mode must be one of: {', '.join(VALID_RETRIEVAL_MODES)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"fusarium_compare_local_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    old_settings = {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "ollama_base_url": settings.ollama_base_url,
    }

    comparison_rows: list[dict[str, Any]] = []
    run_reports: list[dict[str, Any]] = []
    try:
        settings.llm_provider = "ollama"
        if ollama_base_url:
            settings.ollama_base_url = ollama_base_url

        for model in models:
            settings.llm_model = model
            for approach in _approaches(rag_mode=rag_mode, graph_mode=graph_mode):
                approach_name = approach["name"]
                if progress:
                    print(f"\n=== {model} | {approach_name} ===")

                stem = f"{_slug(model)}_{approach_name}"
                report = await evaluate_fusarium_qa(
                    input_path=input_path,
                    output_dir=run_dir,
                    limit=limit,
                    top_k=top_k,
                    retrieval_mode=approach["retrieval_mode"],
                    max_hops=max_hops,
                    use_graph=approach["use_graph"],
                    include_sources=include_sources,
                    csv_path=run_dir / f"{stem}.csv",
                    json_path=run_dir / f"{stem}.json",
                    continue_on_error=continue_on_error,
                    progress=progress,
                )

                row = _comparison_row(
                    model=model,
                    approach=approach_name,
                    retrieval_mode=approach["retrieval_mode"],
                    use_graph=approach["use_graph"],
                    report=report,
                )
                comparison_rows.append(row)
                run_reports.append(
                    {
                        "model": model,
                        "approach": approach_name,
                        "retrieval_mode": approach["retrieval_mode"],
                        "use_graph": approach["use_graph"],
                        "summary": report["summary"],
                        "csv_path": report["csv_path"],
                        "json_path": report["json_path"],
                    }
                )
    finally:
        settings.llm_provider = old_settings["llm_provider"]
        settings.llm_model = old_settings["llm_model"]
        settings.ollama_base_url = old_settings["ollama_base_url"]

    comparison_csv = run_dir / "comparison_summary.csv"
    comparison_json = run_dir / "comparison_summary.json"
    _write_comparison_csv(comparison_rows, comparison_csv)
    _write_json(
        {
            "config": {
                "input_path": str(input_path),
                "models": list(models),
                "llm_provider": "ollama",
                "ollama_base_url": ollama_base_url or settings.ollama_base_url,
                "limit": limit,
                "top_k": top_k,
                "rag_mode": rag_mode,
                "graph_mode": graph_mode,
                "max_hops": max_hops,
                "include_sources": include_sources,
            },
            "summary_rows": comparison_rows,
            "runs": run_reports,
        },
        comparison_json,
    )

    return {
        "run_dir": str(run_dir),
        "comparison_csv": str(comparison_csv),
        "comparison_json": str(comparison_json),
        "rows": comparison_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare plain RAG and GraphRAG on Fusarium QA using local Ollama models."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to Fusarium QA JSONL")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_COMPARE_OUTPUT_DIR),
        help="Directory for comparison CSV/JSON outputs",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Ollama model names/tags, e.g. gemma2.5 qwen3.5 or gemma2:9b qwen3:8b",
    )
    parser.add_argument("--ollama-base-url", default=None, help="Ollama base URL")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved contexts")
    parser.add_argument(
        "--rag-mode",
        choices=VALID_RETRIEVAL_MODES,
        default="basic",
        help="Retrieval mode for plain RAG. Defaults to basic with graph disabled.",
    )
    parser.add_argument(
        "--graph-mode",
        choices=VALID_RETRIEVAL_MODES,
        default="auto",
        help="Retrieval mode for GraphRAG. Defaults to auto with graph enabled.",
    )
    parser.add_argument("--max-hops", type=int, default=2, help="Graph expansion depth")
    parser.add_argument("--no-sources", action="store_true", help="Do not request expanded source contexts")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop at the first failed question")
    parser.add_argument("--quiet", action="store_true", help="Hide per-question progress")
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    report = await compare_fusarium_rag_graphrag(
        models=args.models,
        input_path=args.input,
        output_dir=args.output_dir,
        ollama_base_url=args.ollama_base_url,
        limit=args.limit,
        top_k=args.top_k,
        rag_mode=args.rag_mode,
        graph_mode=args.graph_mode,
        max_hops=args.max_hops,
        include_sources=not args.no_sources,
        continue_on_error=not args.stop_on_error,
        progress=not args.quiet,
    )
    print_comparison(report)


def print_comparison(report: dict[str, Any]) -> None:
    print("\nLocal LLM RAG vs GraphRAG comparison")
    print(f"Run dir: {report['run_dir']}")
    print("model,approach,count,errors,EM,F1,ROUGE-L,evidence_hit")
    for row in report["rows"]:
        print(
            f"{row['model']},{row['approach']},{row['count']},{row['errors']},"
            f"{row['exact_match']:.4f},{row['f1_score']:.4f},"
            f"{row['rouge_l']:.4f},{row['evidence_hit']:.4f}"
        )
    print(f"Summary CSV: {report['comparison_csv']}")
    print(f"Summary JSON: {report['comparison_json']}")


def _approaches(rag_mode: str, graph_mode: str) -> list[dict[str, Any]]:
    return [
        {"name": "rag", "retrieval_mode": rag_mode, "use_graph": False},
        {"name": "graphrag", "retrieval_mode": graph_mode, "use_graph": True},
    ]


def _comparison_row(
    model: str,
    approach: str,
    retrieval_mode: str,
    use_graph: bool,
    report: dict[str, Any],
) -> dict[str, Any]:
    summary = report["summary"]
    overall = summary["overall"]
    return {
        "model": model,
        "approach": approach,
        "retrieval_mode": retrieval_mode,
        "use_graph": use_graph,
        "count": summary["count"],
        "errors": summary["errors"],
        "exact_match": overall["exact_match"],
        "token_precision": overall["token_precision"],
        "token_recall": overall["token_recall"],
        "f1_score": overall["f1_score"],
        "rouge_l": overall["rouge_l"],
        "bleu": overall["bleu"],
        "evidence_hit": overall["evidence_hit"],
        "detail_csv": report["csv_path"],
        "detail_json": report["json_path"],
    }


def _write_comparison_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "model",
        "approach",
        "retrieval_mode",
        "use_graph",
        "count",
        "errors",
        "exact_match",
        "token_precision",
        "token_recall",
        "f1_score",
        "rouge_l",
        "bleu",
        "evidence_hit",
        "detail_csv",
        "detail_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_json(payload: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("._-") or "model"


if __name__ == "__main__":
    asyncio.run(async_main())
