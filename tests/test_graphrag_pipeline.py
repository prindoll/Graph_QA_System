from pathlib import Path

import pytest
from langchain_core.documents import Document as LCDocument

from src.graph.indexer import LangChainGraphIndexer, normalize_text, stable_id
from src.retrieval.base import RetrieverManager


class FakeLLM:
    async def generate_prompt(self, prompt, temperature=None, max_tokens=None):
        return '{"mode": "global"}'


class FakeEmbedding:
    async def embed_text(self, text):
        return [1.0, 0.0, 0.0]

    async def embed_texts(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeGraph:
    async def query(self, query, params=None):
        return []


def make_indexer():
    return LangChainGraphIndexer(
        llm_manager=FakeLLM(),
        embedding_manager=FakeEmbedding(),
        save_intermediates=False,
    )


def test_stable_id_and_text_normalization_are_deterministic():
    assert stable_id("doc", "a", "b") == stable_id("doc", "a", "b")
    assert stable_id("doc", "a", "b") != stable_id("doc", "b", "a")
    assert normalize_text("alpha   beta\n gamma") == "alpha beta gamma"


def test_text_unit_composition_uses_document_boundaries():
    indexer = make_indexer()
    document = LCDocument(
        page_content="GraphRAG connects entities and text units. " * 80,
        metadata={"id": "doc_1", "title": "doc.pdf"},
    )

    text_units = indexer._compose_text_units([document])

    assert text_units
    assert all(text_unit["document_id"] == "doc_1" for text_unit in text_units)
    assert all(text_unit["id"].startswith("tu_") for text_unit in text_units)


def test_knowledge_model_normalizes_entities_and_relationships():
    indexer = make_indexer()
    document = LCDocument(
        page_content="Alpha uses Beta.",
        metadata={"id": "doc_1", "title": "doc.pdf"},
    )
    text_units = [
        {
            "id": "tu_1",
            "human_readable_id": 0,
            "text": "Alpha uses Beta.",
            "n_tokens": 3,
            "document_id": "doc_1",
            "source_title": "doc.pdf",
            "position": 0,
            "entity_ids": [],
            "relationship_ids": [],
        }
    ]
    extraction_results = {
        "tu_1": {
            "entities": [
                {"title": "Alpha", "type": "concept", "description": "Source concept"},
                {"title": "Beta", "type": "concept", "description": "Target concept"},
            ],
            "relationships": [
                {"source": "Alpha", "target": "Beta", "type": "uses", "description": "Alpha uses Beta", "weight": 2}
            ],
        }
    }

    data = indexer._build_knowledge_model([document], text_units, extraction_results)

    assert len(data["documents"]) == 1
    assert len(data["entities"]) == 2
    assert len(data["relationships"]) == 1
    assert data["relationships"][0]["type"] == "USES"
    assert data["text_units"][0]["entity_ids"]
    assert data["text_units"][0]["relationship_ids"]


@pytest.mark.asyncio
async def test_router_uses_llm_mode_when_valid():
    retriever = RetrieverManager(
        embedding_manager=FakeEmbedding(),
        graph_manager=FakeGraph(),
        llm_manager=FakeLLM(),
    )

    assert await retriever.route_query("What are the main themes?") == "global"


def test_neo4j_edges_are_duplicate_safe_merges():
    source = Path("src/graph/neo4j_manager.py").read_text(encoding="utf-8")

    assert "MERGE (a)-[r:" in source
    assert "CREATE (a)-[:" not in source
