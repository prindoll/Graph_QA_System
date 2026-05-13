"""HotPotQA loading helpers."""

from __future__ import annotations

import logging
from typing import Any

from datasets import load_dataset
from langchain_core.documents import Document
from tqdm import tqdm

from config.settings import (
    HOTPOTQA_SPLIT,
    HOTPOTQA_SUBSET,
    HOTPOTQA_SAMPLE_SIZE,
)

logger = logging.getLogger(__name__)


def load_hotpotqa_raw(
    subset: str = HOTPOTQA_SUBSET,
    split: str = HOTPOTQA_SPLIT,
    sample_size: int = HOTPOTQA_SAMPLE_SIZE,
) -> list[dict[str, Any]]:
    """Fetch *sample_size* HotPotQA examples from Hugging Face."""
    logger.info("Loading HotPotQA (subset=%s, split=%s, n=%d)", subset, split, sample_size)
    dataset = load_dataset("hotpot_qa", subset, split=split)
    n = min(sample_size, len(dataset))
    samples = [dataset[i] for i in range(n)]
    logger.info("Loaded %d samples.", n)
    return samples


def _build_documents_from_sample(sample: dict[str, Any]) -> list[Document]:
    """Convert one HotPotQA sample's context paragraphs into LangChain Documents."""
    documents: list[Document] = []
    titles = sample["context"]["title"]
    sentences_list = sample["context"]["sentences"]

    for title, sentences in zip(titles, sentences_list):
        paragraph = " ".join(sentences)
        if not paragraph.strip():
            continue
        doc = Document(
            page_content=f"[{title}]\n{paragraph}",
            metadata={
                "source": title,
                "question": sample["question"],
                "answer": sample["answer"],
                "type": sample.get("type", ""),
                "level": sample.get("level", ""),
            },
        )
        documents.append(doc)
    return documents


def hotpotqa_to_documents(
    samples: list[dict[str, Any]] | None = None,
) -> tuple[list[Document], list[dict[str, Any]]]:
    """Convert HotPotQA samples into (documents, qa_pairs)."""
    if samples is None:
        samples = load_hotpotqa_raw()

    documents: list[Document] = []
    qa_pairs: list[dict[str, Any]] = []

    for sample in tqdm(samples, desc="Processing HotPotQA"):
        documents.extend(_build_documents_from_sample(sample))
        qa_pairs.append({
            "question": sample["question"],
            "answer": sample["answer"],
            "type": sample.get("type", ""),
            "level": sample.get("level", ""),
            "supporting_facts_titles": sample["supporting_facts"]["title"],
        })

    logger.info("Total: %d documents, %d QA pairs.", len(documents), len(qa_pairs))
    return documents, qa_pairs


def load_and_prepare(
    subset: str = HOTPOTQA_SUBSET,
    split: str = HOTPOTQA_SPLIT,
    sample_size: int = HOTPOTQA_SAMPLE_SIZE,
) -> tuple[list[Document], list[dict[str, Any]]]:
    """Load HotPotQA and convert to documents + qa_pairs in one step."""
    samples = load_hotpotqa_raw(subset=subset, split=split, sample_size=sample_size)
    return hotpotqa_to_documents(samples)
