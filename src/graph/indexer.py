import asyncio
import json
import math
import re
import hashlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
from langchain_core.documents import Document as LCDocument
from langchain_core.prompts import ChatPromptTemplate

from config.settings import settings
from src.embedding.base import EmbeddingManager
from src.llm.base import LLMManager
from src.utils.logger import setup_logger
from src.utils.pdf_processor import PDFProcessor

logger = setup_logger(__name__)


def stable_id(prefix: str, *parts: Any) -> str:
    raw = "\u001f".join(str(part) for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_title(value: str) -> str:
    return normalize_text(value).casefold()


class LangChainGraphIndexer:
    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        save_intermediates: bool = True,
        output_dir: str = "data/processing",
        max_concurrent: int = 3,
    ):
        self.llm_manager = llm_manager or LLMManager()
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.save_intermediates = save_intermediates
        self.output_dir = Path(output_dir)
        self.max_concurrent = max_concurrent
        if save_intermediates:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("LangChain Graph Indexer initialized")

    async def index_pdf(
        self,
        pdf_path: str,
        graph_manager,
        start_page: int = 0,
        end_page: Optional[int] = None,
        clear: bool = False,
    ) -> Dict[str, Any]:
        documents = self._load_pdf_documents(pdf_path, start_page=start_page, end_page=end_page)
        return await self.index_documents(documents, graph_manager=graph_manager, clear=clear)

    async def index_documents(
        self,
        documents: List[Dict[str, Any]],
        graph_manager,
        clear: bool = False,
    ) -> Dict[str, Any]:
        if clear:
            await graph_manager.clear()
        schema_result = await graph_manager.ensure_schema()

        lc_documents = self._to_langchain_documents(documents)
        text_units = self._compose_text_units(lc_documents)
        if not text_units:
            return self._empty_result("No text units generated", schema_result)

        extraction_results = await self._extract_all_text_units(text_units)
        index_data = self._build_knowledge_model(lc_documents, text_units, extraction_results)
        communities, entity_community_map = self._detect_communities(
            index_data["entities"],
            index_data["relationships"],
            index_data["text_units"],
        )
        index_data["communities"] = communities
        index_data["community_reports"] = await self._generate_community_reports(
            communities,
            index_data["entities"],
            index_data["relationships"],
            index_data["text_units"],
        )
        index_data["entity_community_map"] = entity_community_map
        await self._embed_index(index_data)

        nodes, edges = self._to_neo4j_payload(index_data)
        self._save_intermediate("complete_graph", "graph", self._strip_embeddings({"nodes": nodes, "edges": edges}))

        if not nodes:
            return self._empty_result("No graph artifacts generated", schema_result)

        node_result = await graph_manager.add_nodes(nodes)
        edge_result = await graph_manager.add_edges(edges)

        return {
            "status": "success",
            "documents": len(index_data["documents"]),
            "text_units": len(index_data["text_units"]),
            "entities": len(index_data["entities"]),
            "relationships": len(index_data["relationships"]),
            "communities": len(index_data["communities"]),
            "community_reports": len(index_data["community_reports"]),
            "nodes_added": node_result.get("count", 0),
            "edges_added": edge_result.get("count", 0),
            "vector_indexes_created": schema_result.get("vector_indexes_created", 0),
        }

    def _load_pdf_documents(self, pdf_path: str, start_page: int = 0, end_page: Optional[int] = None) -> List[Dict[str, Any]]:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            from langchain_community.document_loaders import PyPDFLoader

            loader = PyPDFLoader(str(path))
            pages = loader.load()
            total_pages = len(pages)
            start = max(0, start_page)
            end = total_pages if end_page is None or end_page < 0 else min(end_page, total_pages)
            selected_pages = pages[start:end]
            text = "\n".join(
                f"\n--- Page {page.metadata.get('page', i) + 1} ---\n{page.page_content}"
                for i, page in enumerate(selected_pages, start=start)
            )
        except Exception as e:
            logger.warning(f"LangChain PDF loader failed, falling back to PDFProcessor: {str(e)}")
            processor = PDFProcessor()
            total_pages = processor.get_pdf_page_count(str(path)) or 0
            end = total_pages if end_page is None or end_page < 0 else min(end_page, total_pages)
            text = processor.extract_pdf_batch(str(path), start_page, end) or ""

        if not text.strip():
            return []

        doc_id = stable_id("doc", path.name, start_page, end_page if end_page is not None else "all", text[:500])
        return [
            {
                "id": doc_id,
                "title": path.name,
                "content": text,
                "metadata": {
                    "file": path.name,
                    "path": str(path),
                    "pages": f"{start_page + 1}-{end_page if end_page and end_page > 0 else 'end'}",
                },
            }
        ]

    def _to_langchain_documents(self, documents: List[Dict[str, Any]]) -> List[LCDocument]:
        lc_documents: List[LCDocument] = []
        for doc in documents:
            content = doc.get("content") or doc.get("text") or ""
            if not content.strip():
                continue
            doc_id = doc.get("id") or stable_id("doc", content[:1000])
            metadata = dict(doc.get("metadata") or {})
            metadata.update(
                {
                    "id": doc_id,
                    "title": doc.get("title") or metadata.get("file") or doc_id,
                }
            )
            lc_documents.append(LCDocument(page_content=content, metadata=metadata))
        return lc_documents

    def _compose_text_units(self, documents: List[LCDocument]) -> List[Dict[str, Any]]:
        splitter = self._get_text_splitter()
        text_units: List[Dict[str, Any]] = []
        human_id = 0

        for doc in documents:
            doc_id = doc.metadata["id"]
            title = doc.metadata.get("title", doc_id)
            chunks = splitter.split_text(doc.page_content)
            for position, chunk in enumerate(chunks):
                text = normalize_text(chunk)
                if len(text) < 20:
                    continue
                text_unit_id = stable_id("tu", doc_id, position, text[:300])
                text_units.append(
                    {
                        "id": text_unit_id,
                        "human_readable_id": human_id,
                        "text": text,
                        "n_tokens": self._estimate_tokens(text),
                        "document_id": doc_id,
                        "source_title": title,
                        "position": position,
                        "entity_ids": [],
                        "relationship_ids": [],
                    }
                )
                human_id += 1

        logger.info(f"Composed {len(text_units)} text units from {len(documents)} documents")
        return text_units

    def _get_text_splitter(self):
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except Exception:
            from langchain.text_splitter import RecursiveCharacterTextSplitter

        try:
            return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        except Exception:
            return RecursiveCharacterTextSplitter(
                chunk_size=settings.chunk_size * 4,
                chunk_overlap=settings.chunk_overlap * 4,
                separators=["\n\n", "\n", ". ", " ", ""],
            )

    async def _extract_all_text_units(self, text_units: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def extract(text_unit: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
            async with semaphore:
                result = await self._extract_text_unit_graph(text_unit["text"])
                return text_unit["id"], result

        results = await asyncio.gather(*(extract(text_unit) for text_unit in text_units))
        return dict(results)

    async def _extract_text_unit_graph(self, text: str) -> Dict[str, Any]:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Extract a compact knowledge graph from the text. Return only valid JSON.",
                ),
                (
                    "human",
                    """Text:
{text}

Return JSON with this shape:
{{
  "entities": [
    {{"title": "Entity name", "type": "concept|person|organization|location|event|method|metric", "description": "short grounded description"}}
  ],
  "relationships": [
    {{"source": "Entity name", "target": "Entity name", "type": "RELATED_TO", "description": "why they are connected", "weight": 1.0}}
  ]
}}

Rules:
- Extract up to 20 important entities and 30 relationships.
- Use entity names exactly enough to merge across chunks.
- Relationship type must be uppercase snake case.
- Do not invent facts not present in the text.""",
                ),
            ]
        )
        formatted_prompt = prompt.format(text=text[:6000])

        try:
            response = await self.llm_manager.generate_prompt(formatted_prompt, temperature=0.0)
            data = self._parse_json_object(response)
        except Exception as e:
            logger.warning(f"LLM graph extraction failed; using heuristic fallback: {str(e)}")
            data = self._heuristic_graph_extract(text)

        return {
            "entities": self._normalize_entities(data.get("entities", []), text),
            "relationships": self._normalize_relationships(data.get("relationships", [])),
        }

    def _build_knowledge_model(
        self,
        documents: List[LCDocument],
        text_units: List[Dict[str, Any]],
        extraction_results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        document_rows = []
        for doc in documents:
            text_unit_ids = [tu["id"] for tu in text_units if tu["document_id"] == doc.metadata["id"]]
            document_rows.append(
                {
                    "id": doc.metadata["id"],
                    "human_readable_id": len(document_rows),
                    "title": doc.metadata.get("title", doc.metadata["id"]),
                    "text": doc.page_content,
                    "text_unit_ids": text_unit_ids,
                    "metadata": {k: v for k, v in doc.metadata.items() if k not in {"id", "title"}},
                }
            )

        entities_by_key: Dict[str, Dict[str, Any]] = {}
        relationships_by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        text_unit_lookup = {tu["id"]: tu for tu in text_units}

        for text_unit in text_units:
            text_unit_id = text_unit["id"]
            extraction = extraction_results.get(text_unit_id, {})

            for entity in extraction.get("entities", []):
                title = entity["title"]
                key = normalize_title(title)
                entity_id = stable_id("entity", key)
                current = entities_by_key.setdefault(
                    key,
                    {
                        "id": entity_id,
                        "human_readable_id": len(entities_by_key),
                        "title": title,
                        "name": title,
                        "type": entity.get("type") or "concept",
                        "description_parts": [],
                        "text_unit_ids": set(),
                        "frequency": 0,
                        "degree": 0,
                    },
                )
                description = normalize_text(entity.get("description", ""))
                if description:
                    current["description_parts"].append(description)
                current["text_unit_ids"].add(text_unit_id)
                current["frequency"] += 1
                text_unit["entity_ids"].append(entity_id)

            for relationship in extraction.get("relationships", []):
                source_key = normalize_title(relationship.get("source", ""))
                target_key = normalize_title(relationship.get("target", ""))
                if not source_key or not target_key or source_key == target_key:
                    continue

                for entity_key, title in ((source_key, relationship["source"]), (target_key, relationship["target"])):
                    entities_by_key.setdefault(
                        entity_key,
                        {
                            "id": stable_id("entity", entity_key),
                            "human_readable_id": len(entities_by_key),
                            "title": title,
                            "name": title,
                            "type": "concept",
                            "description_parts": [],
                            "text_unit_ids": set(),
                            "frequency": 0,
                            "degree": 0,
                        },
                    )
                    entities_by_key[entity_key]["text_unit_ids"].add(text_unit_id)
                    text_unit["entity_ids"].append(entities_by_key[entity_key]["id"])

                rel_type = self._normalize_relationship_type(relationship.get("type") or "RELATED_TO")
                rel_key = (source_key, target_key, rel_type)
                source_id = entities_by_key[source_key]["id"]
                target_id = entities_by_key[target_key]["id"]
                rel_id = stable_id("rel", source_id, rel_type, target_id)
                current_rel = relationships_by_key.setdefault(
                    rel_key,
                    {
                        "id": rel_id,
                        "human_readable_id": len(relationships_by_key),
                        "source": entities_by_key[source_key]["title"],
                        "target": entities_by_key[target_key]["title"],
                        "source_id": source_id,
                        "target_id": target_id,
                        "type": rel_type,
                        "description_parts": [],
                        "weight": 0.0,
                        "text_unit_ids": set(),
                    },
                )
                description = normalize_text(relationship.get("description", ""))
                if description:
                    current_rel["description_parts"].append(description)
                current_rel["weight"] += self._safe_float(relationship.get("weight", 1.0), default=1.0)
                current_rel["text_unit_ids"].add(text_unit_id)
                text_unit["relationship_ids"].append(rel_id)

        degree_counts: Dict[str, int] = defaultdict(int)
        for relationship in relationships_by_key.values():
            degree_counts[relationship["source_id"]] += 1
            degree_counts[relationship["target_id"]] += 1

        entities = []
        for entity in entities_by_key.values():
            descriptions = self._dedupe_preserve_order(entity.pop("description_parts"))
            entity["description"] = " ".join(descriptions[:3]) or entity["title"]
            entity["text_unit_ids"] = sorted(entity["text_unit_ids"])
            entity["degree"] = degree_counts.get(entity["id"], 0)
            entities.append(entity)

        relationships = []
        for relationship in relationships_by_key.values():
            descriptions = self._dedupe_preserve_order(relationship.pop("description_parts"))
            relationship["description"] = " ".join(descriptions[:3]) or f"{relationship['source']} {relationship['type']} {relationship['target']}"
            relationship["text_unit_ids"] = sorted(relationship["text_unit_ids"])
            relationship["weight"] = round(max(relationship["weight"], 1.0), 3)
            relationship["combined_degree"] = degree_counts.get(relationship["source_id"], 0) + degree_counts.get(relationship["target_id"], 0)
            relationships.append(relationship)

        for text_unit in text_unit_lookup.values():
            text_unit["entity_ids"] = sorted(set(text_unit["entity_ids"]))
            text_unit["relationship_ids"] = sorted(set(text_unit["relationship_ids"]))

        return {
            "documents": document_rows,
            "text_units": list(text_unit_lookup.values()),
            "entities": entities,
            "relationships": relationships,
            "communities": [],
            "community_reports": [],
        }

    def _detect_communities(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        text_units: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
        if not entities:
            return [], {}

        entity_ids = [entity["id"] for entity in entities]
        entity_id_set = set(entity_ids)
        entity_text_units: Dict[str, set] = {entity["id"]: set(entity.get("text_unit_ids", [])) for entity in entities}
        relationship_by_id = {relationship["id"]: relationship for relationship in relationships}

        groups = self._run_community_detection(entity_ids, relationships)
        root_id = stable_id("community", "root", *sorted(entity_ids))
        communities: List[Dict[str, Any]] = [
            {
                "id": root_id,
                "human_readable_id": 0,
                "community": 0,
                "parent": None,
                "children": list(range(1, len(groups) + 1)),
                "level": 0,
                "title": "Root Knowledge Community",
                "entity_ids": sorted(entity_ids),
                "relationship_ids": [rel["id"] for rel in relationships],
                "text_unit_ids": sorted({tu_id for entity in entity_ids for tu_id in entity_text_units.get(entity, set())}),
                "size": len(entity_ids),
            }
        ]

        entity_community_map: Dict[str, List[str]] = defaultdict(list)
        for index, group in enumerate(groups, start=1):
            group_entities = sorted(entity_id for entity_id in group if entity_id in entity_id_set)
            if not group_entities:
                continue
            group_set = set(group_entities)
            group_relationships = [
                rel_id
                for rel_id, rel in relationship_by_id.items()
                if rel["source_id"] in group_set and rel["target_id"] in group_set
            ]
            group_text_units = sorted({tu_id for entity in group_entities for tu_id in entity_text_units.get(entity, set())})
            community_id = stable_id("community", index, *group_entities)
            communities.append(
                {
                    "id": community_id,
                    "human_readable_id": index,
                    "community": index,
                    "parent": 0,
                    "children": [],
                    "level": 1,
                    "title": f"Knowledge Community {index}",
                    "entity_ids": group_entities,
                    "relationship_ids": group_relationships,
                    "text_unit_ids": group_text_units,
                    "size": len(group_entities),
                }
            )
            for entity_id in group_entities:
                entity_community_map[entity_id].append(community_id)

        if len(communities) == 1:
            for entity_id in entity_ids:
                entity_community_map[entity_id].append(root_id)

        logger.info(f"Detected {len(communities)} communities")
        return communities, dict(entity_community_map)

    def _run_community_detection(self, entity_ids: List[str], relationships: List[Dict[str, Any]]) -> List[List[str]]:
        if len(entity_ids) == 1:
            return [entity_ids]

        if settings.community_algorithm.lower() == "leiden":
            try:
                import igraph as ig
                import leidenalg

                index_by_entity = {entity_id: index for index, entity_id in enumerate(entity_ids)}
                edges = []
                weights = []
                for relationship in relationships:
                    source_index = index_by_entity.get(relationship["source_id"])
                    target_index = index_by_entity.get(relationship["target_id"])
                    if source_index is not None and target_index is not None:
                        edges.append((source_index, target_index))
                        weights.append(float(relationship.get("weight", 1.0)))
                graph = ig.Graph(n=len(entity_ids), edges=edges, directed=False)
                if not edges:
                    return [entity_ids]
                partition = leidenalg.find_partition(
                    graph,
                    leidenalg.RBConfigurationVertexPartition,
                    weights=weights,
                )
                return [[entity_ids[index] for index in community] for community in partition]
            except Exception as e:
                logger.warning(f"Leiden community detection unavailable; falling back to NetworkX Louvain: {str(e)}")

        graph = nx.Graph()
        graph.add_nodes_from(entity_ids)
        for relationship in relationships:
            graph.add_edge(
                relationship["source_id"],
                relationship["target_id"],
                weight=float(relationship.get("weight", 1.0)),
            )
        if graph.number_of_edges() == 0:
            return [entity_ids]
        try:
            communities = nx.community.louvain_communities(graph, weight="weight", seed=42)
        except Exception:
            communities = nx.connected_components(graph)
        return [sorted(list(community)) for community in communities]

    async def _generate_community_reports(
        self,
        communities: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        text_units: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        entity_by_id = {entity["id"]: entity for entity in entities}
        rel_by_id = {relationship["id"]: relationship for relationship in relationships}
        text_unit_by_id = {text_unit["id"]: text_unit for text_unit in text_units}
        reports = []

        for community in communities:
            community_entities = [entity_by_id[eid] for eid in community.get("entity_ids", []) if eid in entity_by_id]
            community_relationships = [rel_by_id[rid] for rid in community.get("relationship_ids", []) if rid in rel_by_id]
            community_text_units = [text_unit_by_id[tid] for tid in community.get("text_unit_ids", []) if tid in text_unit_by_id]
            report_data = await self._generate_single_report(community, community_entities, community_relationships, community_text_units)
            report_id = stable_id("report", community["id"])
            reports.append(
                {
                    "id": report_id,
                    "human_readable_id": len(reports),
                    "community": community["community"],
                    "community_id": community["id"],
                    "parent": community.get("parent"),
                    "children": community.get("children", []),
                    "level": community.get("level", 0),
                    "title": report_data["title"],
                    "summary": report_data["summary"],
                    "full_content": report_data["full_content"],
                    "rank": report_data["rank"],
                    "rating_explanation": report_data["rating_explanation"],
                    "findings": report_data["findings"],
                    "period": datetime.now().date().isoformat(),
                    "size": community.get("size", 0),
                }
            )

        return reports

    async def _generate_single_report(
        self,
        community: Dict[str, Any],
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        text_units: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        entity_lines = "\n".join(f"- {entity['title']} ({entity.get('type', 'concept')}): {entity.get('description', '')}" for entity in entities[:30])
        relationship_lines = "\n".join(
            f"- {rel['source']} -[{rel['type']}]-> {rel['target']}: {rel.get('description', '')}"
            for rel in relationships[:40]
        )
        text_lines = "\n".join(f"- {text_unit['text'][:500]}" for text_unit in text_units[:8])

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Write a concise GraphRAG community report. Return only valid JSON.",
                ),
                (
                    "human",
                    """Community title: {title}
Entities:
{entities}

Relationships:
{relationships}

Source excerpts:
{text_units}

Return JSON:
{{
  "title": "short descriptive title",
  "summary": "one paragraph summary",
  "full_content": "multi-paragraph report grounded in the inputs",
  "rank": 0.0,
  "rating_explanation": "why this community matters",
  "findings": [{{"summary": "finding", "explanation": "evidence"}}]
}}""",
                ),
            ]
        )

        fallback_title = community.get("title") or "Knowledge Community"
        fallback_summary = self._fallback_community_summary(entities, relationships)
        try:
            response = await self.llm_manager.generate_prompt(
                prompt.format(
                    title=fallback_title,
                    entities=entity_lines or "None",
                    relationships=relationship_lines or "None",
                    text_units=text_lines or "None",
                ),
                temperature=0.2,
            )
            data = self._parse_json_object(response)
        except Exception as e:
            logger.warning(f"Community report generation failed; using fallback: {str(e)}")
            data = {}

        findings = data.get("findings")
        if not isinstance(findings, list):
            findings = [{"summary": fallback_title, "explanation": fallback_summary}]

        return {
            "title": normalize_text(data.get("title", "")) or fallback_title,
            "summary": normalize_text(data.get("summary", "")) or fallback_summary,
            "full_content": normalize_text(data.get("full_content", "")) or fallback_summary,
            "rank": self._safe_float(data.get("rank", min(10.0, math.log2(max(len(entities), 1) + 1))), default=1.0),
            "rating_explanation": normalize_text(data.get("rating_explanation", "")) or "Rank is based on community size and connectivity.",
            "findings": findings[:10],
        }

    async def _embed_index(self, index_data: Dict[str, Any]) -> None:
        await self._embed_rows(index_data["text_units"], "text", "text_embedding")
        await self._embed_rows(index_data["entities"], "description", "description_embedding", title_key="title")
        await self._embed_rows(index_data["community_reports"], "full_content", "content_embedding", title_key="title")

    async def _embed_rows(
        self,
        rows: List[Dict[str, Any]],
        text_key: str,
        embedding_key: str,
        title_key: Optional[str] = None,
    ) -> None:
        if not rows:
            return
        texts = []
        for row in rows:
            text = row.get(text_key) or ""
            if title_key and row.get(title_key):
                text = f"{row[title_key]}\n{text}"
            texts.append(text)
        embeddings = await self.embedding_manager.embed_texts(texts)
        for row, embedding in zip(rows, embeddings):
            row[embedding_key] = embedding.tolist()

    def _to_neo4j_payload(self, index_data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        for document in index_data["documents"]:
            nodes.append({"type": "Document", **document})
            for text_unit_id in document.get("text_unit_ids", []):
                edges.append(
                    {
                        "id": stable_id("edge", document["id"], "HAS_TEXT_UNIT", text_unit_id),
                        "source": document["id"],
                        "target": text_unit_id,
                        "type": "HAS_TEXT_UNIT",
                    }
                )

        for text_unit in index_data["text_units"]:
            nodes.append({"type": "TextUnit", **text_unit})
            for entity_id in text_unit.get("entity_ids", []):
                edges.append(
                    {
                        "id": stable_id("edge", text_unit["id"], "MENTIONS", entity_id),
                        "source": text_unit["id"],
                        "target": entity_id,
                        "type": "MENTIONS",
                    }
                )

        for entity in index_data["entities"]:
            nodes.append({"type": "Entity", **entity})
            for community_id in index_data.get("entity_community_map", {}).get(entity["id"], []):
                edges.append(
                    {
                        "id": stable_id("edge", entity["id"], "IN_COMMUNITY", community_id),
                        "source": entity["id"],
                        "target": community_id,
                        "type": "IN_COMMUNITY",
                    }
                )

        for relationship in index_data["relationships"]:
            nodes.append({"type": "RelationshipRecord", **relationship})
            edges.append(
                {
                    "id": stable_id("edge", relationship["source_id"], relationship["type"], relationship["target_id"], relationship["id"]),
                    "source": relationship["source_id"],
                    "target": relationship["target_id"],
                    "type": "RELATED_TO",
                    "relationship_type": relationship["type"],
                    "relationship_record_id": relationship["id"],
                    "description": relationship.get("description", ""),
                    "weight": relationship.get("weight", 1.0),
                }
            )

        for community in index_data["communities"]:
            nodes.append({"type": "Community", **community})
            parent = community.get("parent")
            if parent is not None:
                parent_node = next((item for item in index_data["communities"] if item.get("community") == parent), None)
                if parent_node:
                    edges.append(
                        {
                            "id": stable_id("edge", parent_node["id"], "PARENT_COMMUNITY", community["id"]),
                            "source": parent_node["id"],
                            "target": community["id"],
                            "type": "PARENT_COMMUNITY",
                        }
                    )

        for report in index_data["community_reports"]:
            report_for_node = dict(report)
            report_for_node["findings_json"] = json.dumps(report_for_node.pop("findings", []), ensure_ascii=False)
            nodes.append({"type": "CommunityReport", **report_for_node})
            edges.append(
                {
                    "id": stable_id("edge", report["community_id"], "HAS_REPORT", report["id"]),
                    "source": report["community_id"],
                    "target": report["id"],
                    "type": "HAS_REPORT",
                }
            )

        return nodes, edges

    def _normalize_entities(self, entities: Any, original_text: str) -> List[Dict[str, str]]:
        normalized = []
        if not isinstance(entities, list):
            return self._heuristic_graph_extract(original_text).get("entities", [])
        for entity in entities:
            if isinstance(entity, str):
                title = normalize_text(entity)
                entity_type = "concept"
                description = ""
            elif isinstance(entity, dict):
                title = normalize_text(entity.get("title") or entity.get("name") or entity.get("id") or "")
                entity_type = normalize_text(entity.get("type") or "concept").lower()
                description = normalize_text(entity.get("description") or "")
            else:
                continue
            if title and len(title) <= 120:
                normalized.append({"title": title, "type": entity_type or "concept", "description": description})
        if not normalized:
            return self._heuristic_graph_extract(original_text).get("entities", [])
        return self._dedupe_entities(normalized)

    def _normalize_relationships(self, relationships: Any) -> List[Dict[str, Any]]:
        normalized = []
        if not isinstance(relationships, list):
            return normalized
        for relationship in relationships:
            if not isinstance(relationship, dict):
                continue
            source = normalize_text(relationship.get("source") or relationship.get("subject") or "")
            target = normalize_text(relationship.get("target") or relationship.get("object") or "")
            if not source or not target or source.casefold() == target.casefold():
                continue
            normalized.append(
                {
                    "source": source,
                    "target": target,
                    "type": self._normalize_relationship_type(relationship.get("type") or relationship.get("predicate") or "RELATED_TO"),
                    "description": normalize_text(relationship.get("description") or relationship.get("reason") or ""),
                    "weight": self._safe_float(relationship.get("weight", 1.0), default=1.0),
                }
            )
        return normalized

    def _heuristic_graph_extract(self, text: str) -> Dict[str, Any]:
        phrases = re.findall(r"\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,3}\b", text)
        terms = []
        seen = set()
        for phrase in phrases:
            cleaned = normalize_text(phrase)
            if len(cleaned) < 3:
                continue
            key = cleaned.casefold()
            if key not in seen:
                seen.add(key)
                terms.append(cleaned)
            if len(terms) >= 12:
                break
        if not terms:
            words = [word.strip(".,;:()[]{}") for word in text.split() if len(word) > 5]
            terms = list(dict.fromkeys(words[:8]))
        return {
            "entities": [
                {"title": term, "type": "concept", "description": self._excerpt_for_term(text, term)}
                for term in terms
            ],
            "relationships": [
                {
                    "source": left,
                    "target": right,
                    "type": "RELATED_TO",
                    "description": "Entities co-occur in the same text unit.",
                    "weight": 1.0,
                }
                for left, right in zip(terms, terms[1:])
            ],
        }

    def _parse_json_object(self, response: str) -> Dict[str, Any]:
        if not response:
            return {}
        start = response.find("{")
        end = response.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        return json.loads(response[start : end + 1])

    def _normalize_relationship_type(self, value: str) -> str:
        value = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "RELATED_TO")).strip("_").upper()
        return value or "RELATED_TO"

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text.split()))

    def _excerpt_for_term(self, text: str, term: str) -> str:
        index = text.casefold().find(term.casefold())
        if index == -1:
            return normalize_text(text[:240])
        start = max(0, index - 100)
        end = min(len(text), index + len(term) + 180)
        return normalize_text(text[start:end])

    def _dedupe_entities(self, entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
        seen = set()
        result = []
        for entity in entities:
            key = normalize_title(entity["title"])
            if key not in seen:
                seen.add(key)
                result.append(entity)
        return result

    def _dedupe_preserve_order(self, items: Iterable[str]) -> List[str]:
        seen = set()
        result = []
        for item in items:
            key = item.casefold()
            if item and key not in seen:
                seen.add(key)
                result.append(item)
        return result

    def _fallback_community_summary(self, entities: Sequence[Dict[str, Any]], relationships: Sequence[Dict[str, Any]]) -> str:
        entity_titles = ", ".join(entity["title"] for entity in entities[:8]) or "No named entities"
        return f"This community contains {len(entities)} entities and {len(relationships)} relationships. Key entities include {entity_titles}."

    def _empty_result(self, message: str, schema_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "status": "error",
            "message": message,
            "documents": 0,
            "text_units": 0,
            "entities": 0,
            "relationships": 0,
            "communities": 0,
            "community_reports": 0,
            "nodes_added": 0,
            "edges_added": 0,
            "vector_indexes_created": (schema_result or {}).get("vector_indexes_created", 0),
        }

    def _save_intermediate(self, doc_id: str, stage: str, content: Any) -> None:
        if not self.save_intermediates:
            return
        try:
            doc_folder = self.output_dir / doc_id.replace(" ", "_")
            doc_folder.mkdir(parents=True, exist_ok=True)
            suffix = "json" if stage != "markdown" else "md"
            file_path = doc_folder / f"{stage}.{suffix}"
            with open(file_path, "w", encoding="utf-8") as f:
                if suffix == "json":
                    json.dump(content, f, indent=2, ensure_ascii=False)
                else:
                    f.write(str(content))
        except Exception as e:
            logger.warning(f"Failed to save intermediate {stage}: {str(e)}")

    def _strip_embeddings(self, payload: Any) -> Any:
        if isinstance(payload, dict):
            return {
                key: ("<embedding>" if key.endswith("_embedding") else self._strip_embeddings(value))
                for key, value in payload.items()
            }
        if isinstance(payload, list):
            return [self._strip_embeddings(item) for item in payload]
        return payload
