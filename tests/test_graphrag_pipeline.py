from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document as LCDocument

from src.embedding.base import EmbeddingManager
from src.graph.indexer import LangChainGraphIndexer, normalize_text, stable_id
from src.llm.openai_provider import OpenAIProvider
from src.retrieval.base import RetrieverManager


class FakeLLM:
    async def generate_prompt(self, prompt, temperature=None, max_tokens=None, json_mode=False):
        return '{"mode": "global"}'


class FakeEmbedding:
    async def embed_text(self, text):
        return [1.0, 0.0, 0.0]

    async def embed_texts(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeGraph:
    async def query(self, query, params=None):
        return []


class FakeOpenAIEmbeddings:
    def __init__(self):
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=[float(index), 1.0, 2.0])
                for index, _ in enumerate(kwargs["input"])
            ]
        )


class FakeOpenAIClient:
    def __init__(self):
        self.embeddings = FakeOpenAIEmbeddings()


class FakeChatCompletions:
    def __init__(self):
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))
            ]
        )


class FakeChatClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeChatCompletions())


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


def test_markdown_loader_chunks_with_heading_metadata(tmp_path):
    indexer = make_indexer()
    markdown_path = tmp_path / "paper.md"
    markdown_path.write_text(
        "# Paper Title\n\n"
        "Introductory context about Fusarium taxonomy and phylogeny.\n\n"
        "## Methods\n\n"
        "The study compares rpb1, rpb2, tef1, ITS, LSU, CaM, tub2, and act1 markers.",
        encoding="utf-8",
    )

    documents = indexer._load_markdown_documents(str(markdown_path))
    text_units = indexer._compose_text_units(indexer._to_langchain_documents(documents))

    assert documents[0]["title"] == "Paper Title"
    assert documents[0]["metadata"]["source_type"] == "markdown"
    assert text_units
    assert any(text_unit["section_title"] == "Methods" for text_unit in text_units)
    assert any(text_unit["heading_path"] == "Paper Title > Methods" for text_unit in text_units)
    assert all(text_unit["source_type"] == "markdown" for text_unit in text_units)


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


@pytest.mark.asyncio
async def test_openai_embedding_provider_uses_batch_request_with_dimensions():
    fake_client = FakeOpenAIClient()
    manager = EmbeddingManager(
        provider="openai",
        model_name="text-embedding-3-small",
        dimension=3,
        client=fake_client,
    )

    embeddings = await manager.embed_texts(["alpha", "beta"])

    assert len(embeddings) == 2
    assert embeddings[0].tolist() == [0.0, 1.0, 2.0]
    assert fake_client.embeddings.requests[0]["model"] == "text-embedding-3-small"
    assert fake_client.embeddings.requests[0]["dimensions"] == 3


@pytest.mark.asyncio
async def test_openai_provider_preserves_zero_temperature_and_json_mode():
    provider = OpenAIProvider()
    fake_client = FakeChatClient()
    provider.client = fake_client

    result = await provider.generate("Return JSON", temperature=0.0, max_tokens=64, json_mode=True)

    assert result == '{"ok": true}'
    request = fake_client.chat.completions.requests[0]
    assert request["temperature"] == 0.0
    assert request["response_format"] == {"type": "json_object"}


def test_hybrid_ranking_combines_vector_keyword_and_graph_scores():
    retriever = RetrieverManager(
        embedding_manager=FakeEmbedding(),
        graph_manager=FakeGraph(),
        llm_manager=None,
    )

    ranked = retriever._merge_ranked_results(
        vector_results=[
            {"id": "a", "type": "TextUnit", "title": "A", "text": "Alpha", "score": 0.7, "metadata": {"entity_ids": ["e1"]}},
            {"id": "b", "type": "TextUnit", "title": "B", "text": "Beta", "score": 0.9, "metadata": {}},
        ],
        keyword_results=[
            {"id": "a", "type": "TextUnit", "title": "A", "text": "Alpha", "score": 10.0, "metadata": {"relationship_ids": ["r1", "r2"]}},
        ],
        top_k=2,
    )

    assert ranked[0]["id"] == "a"
    assert ranked[0]["metadata"]["hybrid_scores"]["keyword"] == 1.0


def test_heuristic_router_handles_vietnamese_unicode_terms():
    retriever = RetrieverManager(
        embedding_manager=FakeEmbedding(),
        graph_manager=FakeGraph(),
        llm_manager=None,
    )

    assert retriever._heuristic_route("tổng quan tài liệu này") == "global"
    assert retriever._heuristic_route("so sánh Fusarium và Neocosmospora") == "drift"


def test_neo4j_edges_are_duplicate_safe_merges():
    source = Path("src/graph/neo4j_manager.py").read_text(encoding="utf-8")

    assert "UNWIND $rows AS row" in source
    assert "MERGE (a)-[r:" in source
    assert "CREATE (a)-[:" not in source
    assert "CREATE FULLTEXT INDEX" in source
