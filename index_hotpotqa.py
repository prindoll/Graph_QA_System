#!/usr/bin/env python
"""Index HotPotQA contexts into the graph."""

import argparse
import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple





logger = logging.getLogger(__name__)

SOURCE_DOC = "hotpotqa"
DEFAULT_GRAPH_TYPE = "neo4j"


@dataclass
class ArticleAccumulator:
    id: str
    title: str
    sentence_texts: List[str] = field(default_factory=list)
    seen_sentence_texts: Set[str] = field(default_factory=set)

    def add_sentence(self, text: str) -> None:
        normalized = normalize_text(text)
        if normalized and normalized not in self.seen_sentence_texts:
            self.seen_sentence_texts.add(normalized)
            self.sentence_texts.append(text.strip())

    @property
    def content(self) -> str:
        return " ".join(self.sentence_texts)


@dataclass
class IndexStats:
    samples_read: int = 0
    valid_samples: int = 0
    skipped_samples: int = 0
    article_nodes: int = 0
    sentence_nodes: int = 0
    has_sentence_edges: int = 0
    next_sentence_edges: int = 0
    mention_edges: int = 0
    links_to_edges: int = 0
    db_nodes: Optional[int] = None
    db_edges: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "samples_read": self.samples_read,
            "valid_samples": self.valid_samples,
            "skipped_samples": self.skipped_samples,
            "article_nodes": self.article_nodes,
            "sentence_nodes": self.sentence_nodes,
            "edges": {
                "HAS_SENTENCE": self.has_sentence_edges,
                "NEXT_SENTENCE": self.next_sentence_edges,
                "MENTIONS_ARTICLE": self.mention_edges,
                "LINKS_TO": self.links_to_edges,
            },
            "db_stats": {
                "nodes": self.db_nodes,
                "edges": self.db_edges,
            },
        }


def stable_id(prefix: str, *parts: Any) -> str:
    raw = "\u241f".join(str(part) for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_title(title: str) -> str:
    return normalize_text(title).casefold()


def get_default_graph_type() -> str:
    try:
        from config.settings import settings as app_settings

        return app_settings.graph_db_type
    except Exception:
        return DEFAULT_GRAPH_TYPE

def resolve_input_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path

    # Fix a common copied Windows path case.
    if not any(sep in raw_path for sep in ("/", "\\")) and raw_path.startswith("data"):
        repaired = Path("data") / raw_path[len("data"):]
        if repaired.exists():
            return repaired

    return path

def load_hotpot_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected HotPotQA JSON list at {path}, got {type(data).__name__}")

    return data


def iter_limited(samples: Iterable[Dict[str, Any]], limit: Optional[int]) -> Iterable[Dict[str, Any]]:
    for index, sample in enumerate(samples):
        if limit is not None and index >= limit:
            break
        yield sample


def parse_context(context: Any) -> Optional[List[Tuple[str, List[str]]]]:
    if not isinstance(context, list):
        return None

    parsed: List[Tuple[str, List[str]]] = []
    for item in context:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue

        title, sentences = item
        title_text = normalize_text(str(title))
        if not title_text or not isinstance(sentences, list):
            continue

        sentence_texts = []
        for sentence in sentences:
            sentence_text = normalize_text(str(sentence))
            if sentence_text:
                sentence_texts.append(sentence_text)

        if sentence_texts:
            parsed.append((title_text, sentence_texts))

    return parsed or None


def make_title_pattern(title: str) -> Optional[re.Pattern[str]]:
    title = normalize_text(title)
    if len(title) < 3:
        return None

    escaped = re.escape(title)
    return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)


def add_edge(
    edges: Dict[Tuple[str, str, str], Dict[str, Any]],
    source: str,
    target: str,
    edge_type: str,
) -> None:
    if source and target and source != target:
        edges.setdefault(
            (source, target, edge_type),
            {
                "source": source,
                "target": target,
                "type": edge_type,
                "confidence": 1.0,
            },
        )


def build_index(samples: Iterable[Dict[str, Any]], limit: Optional[int] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], IndexStats]:
    stats = IndexStats()
    articles: Dict[str, ArticleAccumulator] = {}
    sentence_nodes: Dict[str, Dict[str, Any]] = {}
    edges: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    pending_contexts: List[List[Tuple[str, List[Tuple[int, str, str]]]]] = []

    for sample in iter_limited(samples, limit):
        stats.samples_read += 1
        context = parse_context(sample.get("context") if isinstance(sample, dict) else None)
        if not context:
            stats.skipped_samples += 1
            continue

        stats.valid_samples += 1
        sample_context: List[Tuple[str, List[Tuple[int, str, str]]]] = []

        for title, sentences in context:
            title_key = normalize_title(title)
            article_id = stable_id("article", title_key)
            article = articles.setdefault(
                title_key,
                ArticleAccumulator(id=article_id, title=title),
            )

            sentence_refs: List[Tuple[int, str, str]] = []
            for sentence_index, sentence in enumerate(sentences):
                article.add_sentence(sentence)

                sentence_id = stable_id("sentence", title_key, sentence_index, normalize_text(sentence))
                sentence_nodes.setdefault(
                    sentence_id,
                    {
                        "id": sentence_id,
                        "label": f"{title} [{sentence_index}]",
                        "title": title,
                        "sentence_index": sentence_index,
                        "content": sentence,
                        "source_doc": SOURCE_DOC,
                        "type": "sentence",
                    },
                )
                sentence_refs.append((sentence_index, sentence_id, sentence))
                add_edge(edges, article_id, sentence_id, "HAS_SENTENCE")

            for left, right in zip(sentence_refs, sentence_refs[1:]):
                add_edge(edges, left[1], right[1], "NEXT_SENTENCE")

            sample_context.append((title_key, sentence_refs))

        pending_contexts.append(sample_context)

    add_mention_edges(pending_contexts, articles, edges)

    article_nodes = [
        {
            "id": article.id,
            "label": article.title,
            "title": article.title,
            "content": article.content,
            "source_doc": SOURCE_DOC,
            "type": "article",
        }
        for article in articles.values()
    ]
    nodes = article_nodes + list(sentence_nodes.values())
    edge_list = list(edges.values())

    stats.article_nodes = len(article_nodes)
    stats.sentence_nodes = len(sentence_nodes)
    stats.has_sentence_edges = count_edges(edge_list, "HAS_SENTENCE")
    stats.next_sentence_edges = count_edges(edge_list, "NEXT_SENTENCE")
    stats.mention_edges = count_edges(edge_list, "MENTIONS_ARTICLE")
    stats.links_to_edges = count_edges(edge_list, "LINKS_TO")

    return nodes, edge_list, stats


def add_mention_edges(
    contexts: List[List[Tuple[str, List[Tuple[int, str, str]]]]],
    articles: Dict[str, ArticleAccumulator],
    edges: Dict[Tuple[str, str, str], Dict[str, Any]],
) -> None:
    # Link articles when a sentence mentions another title.
    title_patterns = {
        title_key: make_title_pattern(article.title)
        for title_key, article in articles.items()
    }

    for context in contexts:
        context_title_keys = [title_key for title_key, _ in context]

        for source_title_key, sentence_refs in context:
            source_article = articles[source_title_key]

            for target_title_key in context_title_keys:
                if target_title_key == source_title_key:
                    continue

                target_article = articles[target_title_key]
                pattern = title_patterns.get(target_title_key)
                if pattern is None:
                    continue

                linked_from_article = False
                for _, sentence_id, sentence in sentence_refs:
                    if pattern.search(sentence):
                        add_edge(edges, sentence_id, target_article.id, "MENTIONS_ARTICLE")
                        linked_from_article = True

                if linked_from_article:
                    add_edge(edges, source_article.id, target_article.id, "LINKS_TO")


def count_edges(edges: List[Dict[str, Any]], edge_type: str) -> int:
    return sum(1 for edge in edges if edge.get("type") == edge_type)


async def persist_index(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    graph_type: str,
    batch_size: int,
    clear: bool,
) -> Dict[str, int]:
    from src.graph.base import GraphManager

    graph_manager = GraphManager(graph_type=graph_type)

    if clear:
        logger.info("Clearing existing graph before HotPotQA index")
        await graph_manager.clear()

    if nodes:
        await graph_manager.batch_add_nodes(nodes, batch_size=batch_size)
    if edges:
        await graph_manager.batch_add_edges(edges, batch_size=batch_size)

    stats = await graph_manager.get_stats()
    await graph_manager.disconnect()
    return {
        "nodes": stats.get("nodes", 0),
        "edges": stats.get("edges", 0),
    }


def write_report(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def print_summary(stats: IndexStats, dry_run: bool, report_path: Optional[Path]) -> None:
    print("\n" + "=" * 60)
    print("HotPotQA Index Summary")
    print("=" * 60)
    print(f"Mode: {'dry-run' if dry_run else 'persist'}")
    print(f"Samples read: {stats.samples_read}")
    print(f"Valid samples: {stats.valid_samples}")
    print(f"Skipped samples: {stats.skipped_samples}")
    print(f"Article nodes: {stats.article_nodes}")
    print(f"Sentence nodes: {stats.sentence_nodes}")
    print(f"HAS_SENTENCE edges: {stats.has_sentence_edges}")
    print(f"NEXT_SENTENCE edges: {stats.next_sentence_edges}")
    print(f"MENTIONS_ARTICLE edges: {stats.mention_edges}")
    print(f"LINKS_TO edges: {stats.links_to_edges}")
    if stats.db_nodes is not None or stats.db_edges is not None:
        print(f"DB nodes: {stats.db_nodes}")
        print(f"DB edges: {stats.db_edges}")
    if report_path:
        print(f"Report: {report_path}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index HotPotQA context passages into GraphRAG without answer leakage."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a standard HotPotQA JSON file containing a list of samples.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the graph before indexing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Index only the first N samples.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for graph writes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and report index statistics without writing to the graph.",
    )
    parser.add_argument(
        "--graph-type",
        default=get_default_graph_type(),
        choices=["neo4j", "networkx"],
        help="Graph backend type. Defaults to config settings when available, otherwise neo4j.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional path to write a JSON summary report.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    input_path = resolve_input_path(args.input)
    report_path = Path(args.report) if args.report else None

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be a positive integer")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be a positive integer")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    samples = load_hotpot_json(input_path)
    nodes, edges, stats = build_index(samples, limit=args.limit)

    if not args.dry_run:
        db_stats = await persist_index(
            nodes=nodes,
            edges=edges,
            graph_type=args.graph_type,
            batch_size=args.batch_size,
            clear=args.clear,
        )
        stats.db_nodes = db_stats["nodes"]
        stats.db_edges = db_stats["edges"]

    report = stats.as_dict()
    report["input"] = str(input_path)
    report["limit"] = args.limit
    report["graph_type"] = args.graph_type
    report["dry_run"] = args.dry_run
    # Keep reports clear about what went into the graph.
    report["leakage_policy"] = "context_only_no_question_answer_or_supporting_facts"

    if report_path:
        write_report(report_path, report)

    print_summary(stats, dry_run=args.dry_run, report_path=report_path)


if __name__ == "__main__":
    asyncio.run(main())






