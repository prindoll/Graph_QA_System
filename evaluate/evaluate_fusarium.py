#!/usr/bin/env python
"""Evaluate the Fusarium GraphRAG QA set."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import re
import string
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


VALID_RETRIEVAL_MODES = ("auto", "basic", "local", "global", "drift")
DEFAULT_INPUT = Path("data/qa_eval_fusarium.jsonl")
DEFAULT_OUTPUT_DIR = Path("evaluate")


def normalize_answer(text: str) -> str:
    """Normalize text for exact match and token overlap metrics."""
    text = (text or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def answer_tokens(text: str) -> list[str]:
    return normalize_answer(text).split()


def exact_match_score(prediction: str, expected: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(expected))


def token_precision_recall_f1(prediction: str, expected: str) -> dict[str, float]:
    pred_tokens = answer_tokens(prediction)
    expected_tokens = answer_tokens(expected)

    if not pred_tokens and not expected_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not expected_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    overlap = Counter(pred_tokens) & Counter(expected_tokens)
    same = sum(overlap.values())
    if same == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = same / len(pred_tokens)
    recall = same / len(expected_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def rouge_l_score(prediction: str, expected: str) -> float:
    pred_tokens = answer_tokens(prediction)
    expected_tokens = answer_tokens(expected)
    if not pred_tokens and not expected_tokens:
        return 1.0
    if not pred_tokens or not expected_tokens:
        return 0.0

    lcs = _lcs_length(pred_tokens, expected_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(expected_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu_score(prediction: str, expected: str, max_n: int = 4) -> float:
    pred_tokens = answer_tokens(prediction)
    expected_tokens = answer_tokens(expected)
    if not pred_tokens or not expected_tokens:
        return 0.0

    brevity_penalty = min(1.0, math.exp(1 - len(expected_tokens) / len(pred_tokens)))
    log_precision_sum = 0.0
    effective_n = 0

    for n in range(1, max_n + 1):
        pred_ngrams = _ngrams(pred_tokens, n)
        expected_ngrams = _ngrams(expected_tokens, n)
        if not pred_ngrams:
            continue
        clipped = sum(min(pred_ngrams[ngram], expected_ngrams[ngram]) for ngram in pred_ngrams)
        total = sum(pred_ngrams.values())
        log_precision_sum += math.log((clipped + 1) / (total + 1))
        effective_n += 1

    if effective_n == 0:
        return 0.0
    return brevity_penalty * math.exp(log_precision_sum / effective_n)


def load_fusarium_qa(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    qa_path = Path(path)
    if not qa_path.exists():
        raise FileNotFoundError(f"QA file not found: {qa_path}")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be a positive integer")

    examples: list[dict[str, Any]] = []
    with qa_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if limit is not None and len(examples) >= limit:
                break
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            _validate_qa_item(item, line_number)
            examples.append(item)
    return examples


async def evaluate_fusarium_qa(
    input_path: str | Path = DEFAULT_INPUT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    limit: int | None = None,
    top_k: int = 5,
    retrieval_mode: str = "auto",
    max_hops: int = 2,
    use_graph: bool = True,
    include_sources: bool = True,
    csv_path: str | Path | None = None,
    json_path: str | Path | None = None,
    continue_on_error: bool = True,
    progress: bool = True,
) -> dict[str, Any]:
    """Run GraphRAG over the Fusarium JSONL QA set and persist metrics."""
    if retrieval_mode not in VALID_RETRIEVAL_MODES:
        raise ValueError(f"retrieval_mode must be one of: {', '.join(VALID_RETRIEVAL_MODES)}")
    if top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if max_hops < 1:
        raise ValueError("max_hops must be a positive integer")

    examples = load_fusarium_qa(input_path, limit=limit)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.core.graphrag import GraphRAG

    rag = GraphRAG()
    rows: list[dict[str, Any]] = []
    try:
        for index, item in enumerate(examples, start=1):
            question = item["question"]
            if progress:
                print(f"[{index}/{len(examples)}] {item.get('id', '')}: {question}")

            try:
                rag_result = await rag.query(
                    query=question,
                    top_k=top_k,
                    use_graph=use_graph,
                    retrieval_mode=retrieval_mode,
                    max_hops=max_hops,
                    include_sources=include_sources,
                )
                row = build_result_row(item, rag_result)
            except Exception as exc:
                if not continue_on_error:
                    raise
                row = build_error_row(item, exc, retrieval_mode)
            rows.append(row)
    finally:
        await rag.graph_manager.disconnect()

    summary = summarize_results(rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_output = Path(csv_path) if csv_path else output_dir / f"fusarium_eval_{timestamp}.csv"
    json_output = Path(json_path) if json_path else output_dir / f"fusarium_eval_{timestamp}.json"

    write_csv(rows, csv_output)
    write_json(
        {
            "summary": summary,
            "config": {
                "input_path": str(input_path),
                "limit": limit,
                "top_k": top_k,
                "retrieval_mode": retrieval_mode,
                "max_hops": max_hops,
                "use_graph": use_graph,
                "include_sources": include_sources,
            },
            "results": rows,
        },
        json_output,
    )

    return {
        "summary": summary,
        "results": rows,
        "csv_path": str(csv_output),
        "json_path": str(json_output),
    }


def build_result_row(item: dict[str, Any], rag_result: dict[str, Any]) -> dict[str, Any]:
    expected = item["expected_answer"]
    prediction = rag_result.get("answer", "")
    prf = token_precision_recall_f1(prediction, expected)
    sources = summarize_sources(rag_result.get("context", []))
    top_source = sources[0] if sources else {}

    return {
        "id": item.get("id", ""),
        "question": item["question"],
        "expected_answer": expected,
        "prediction": prediction,
        "question_type": item.get("question_type", ""),
        "difficulty": item.get("difficulty", ""),
        "evidence": item.get("evidence", ""),
        "retrieval_mode": rag_result.get("retrieval_mode", ""),
        "exact_match": exact_match_score(prediction, expected),
        "token_precision": prf["precision"],
        "token_recall": prf["recall"],
        "f1_score": prf["f1"],
        "rouge_l": rouge_l_score(prediction, expected),
        "bleu": bleu_score(prediction, expected),
        "evidence_hit": evidence_hit(item.get("evidence", ""), sources),
        "sources_count": len(sources),
        "top_source_type": top_source.get("type", ""),
        "top_source_title": top_source.get("title", ""),
        "top_source_section": top_source.get("section_title", ""),
        "top_source_score": top_source.get("score", ""),
        "sources": sources,
        "error": "",
    }


def build_error_row(item: dict[str, Any], exc: Exception, retrieval_mode: str) -> dict[str, Any]:
    return {
        "id": item.get("id", ""),
        "question": item.get("question", ""),
        "expected_answer": item.get("expected_answer", ""),
        "prediction": "",
        "question_type": item.get("question_type", ""),
        "difficulty": item.get("difficulty", ""),
        "evidence": item.get("evidence", ""),
        "retrieval_mode": retrieval_mode,
        "exact_match": 0.0,
        "token_precision": 0.0,
        "token_recall": 0.0,
        "f1_score": 0.0,
        "rouge_l": 0.0,
        "bleu": 0.0,
        "evidence_hit": 0.0,
        "sources_count": 0,
        "top_source_type": "",
        "top_source_title": "",
        "top_source_section": "",
        "top_source_score": "",
        "sources": [],
        "error": str(exc),
    }


def summarize_sources(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for context in contexts:
        metadata = dict(context.get("metadata") or {})
        text = context.get("text", "") or ""
        sources.append(
            {
                "id": context.get("id", ""),
                "type": context.get("type", ""),
                "title": context.get("title", ""),
                "score": context.get("score", 0.0),
                "source_title": metadata.get("source_title", ""),
                "section_title": metadata.get("section_title", ""),
                "heading_path": metadata.get("heading_path", ""),
                "retrieval_mode": metadata.get("retrieval_mode", ""),
                "text_preview": text[:500],
            }
        )
    return sources


def evidence_hit(evidence: str, sources: list[dict[str, Any]]) -> float:
    terms = [term.strip().lower() for term in re.split(r"[;,]", evidence or "") if term.strip()]
    if not terms:
        return 0.0

    source_text = "\n".join(
        " ".join(
            str(source.get(field, ""))
            for field in ("title", "source_title", "section_title", "heading_path", "text_preview")
        )
        for source in sources
    ).lower()
    return float(any(term in source_text for term in terms))


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ["exact_match", "token_precision", "token_recall", "f1_score", "rouge_l", "bleu", "evidence_hit"]
    summary = {
        "count": len(rows),
        "errors": sum(1 for row in rows if row.get("error")),
        "overall": {metric: _mean(row.get(metric, 0.0) for row in rows) for metric in metrics},
        "by_question_type": _group_summary(rows, "question_type", metrics),
        "by_difficulty": _group_summary(rows, "difficulty", metrics),
    }
    return summary


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "question",
        "expected_answer",
        "prediction",
        "question_type",
        "difficulty",
        "evidence",
        "retrieval_mode",
        "exact_match",
        "token_precision",
        "token_recall",
        "f1_score",
        "rouge_l",
        "bleu",
        "evidence_hit",
        "sources_count",
        "top_source_type",
        "top_source_title",
        "top_source_section",
        "top_source_score",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)


def print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    overall = summary["overall"]
    print("\nFusarium QA evaluation")
    print(f"Samples: {summary['count']}")
    print(f"Errors: {summary['errors']}")
    print(f"EM: {overall['exact_match']:.4f}")
    print(f"F1: {overall['f1_score']:.4f}")
    print(f"ROUGE-L: {overall['rouge_l']:.4f}")
    print(f"Evidence hit: {overall['evidence_hit']:.4f}")
    print(f"CSV: {report['csv_path']}")
    print(f"JSON: {report['json_path']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GraphRAG on data/qa_eval_fusarium.jsonl")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to Fusarium QA JSONL")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for CSV/JSON outputs")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieval results")
    parser.add_argument(
        "--mode",
        "--retrieval-mode",
        dest="retrieval_mode",
        choices=VALID_RETRIEVAL_MODES,
        default="auto",
        help="GraphRAG retrieval mode",
    )
    parser.add_argument("--max-hops", type=int, default=2, help="Graph expansion depth")
    parser.add_argument("--no-graph", action="store_true", help="Force non-graph basic retrieval")
    parser.add_argument("--no-sources", action="store_true", help="Do not request expanded source contexts")
    parser.add_argument("--csv", dest="csv_path", default=None, help="Explicit CSV output path")
    parser.add_argument("--json", dest="json_path", default=None, help="Explicit JSON output path")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop at the first failed question")
    parser.add_argument("--quiet", action="store_true", help="Hide per-question progress")
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    report = await evaluate_fusarium_qa(
        input_path=args.input,
        output_dir=args.output_dir,
        limit=args.limit,
        top_k=args.top_k,
        retrieval_mode=args.retrieval_mode,
        max_hops=args.max_hops,
        use_graph=not args.no_graph,
        include_sources=not args.no_sources,
        csv_path=args.csv_path,
        json_path=args.json_path,
        continue_on_error=not args.stop_on_error,
        progress=not args.quiet,
    )
    print_summary(report)


def _validate_qa_item(item: dict[str, Any], line_number: int) -> None:
    for key in ("question", "expected_answer"):
        if not isinstance(item.get(key), str) or not item[key].strip():
            raise ValueError(f"Line {line_number} must contain a non-empty string field: {key}")


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))


def _mean(values: Any) -> float:
    numbers = [float(value or 0.0) for value in values]
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


def _group_summary(rows: list[dict[str, Any]], field: str, metrics: list[str]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get(field) or "unknown")
        grouped[key].append(row)
    return {
        key: {
            "count": len(items),
            **{metric: _mean(item.get(metric, 0.0) for item in items) for metric in metrics},
        }
        for key, items in sorted(grouped.items())
    }


if __name__ == "__main__":
    asyncio.run(async_main())
