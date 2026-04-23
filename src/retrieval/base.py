import json
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from config.settings import settings
from ..embedding.base import EmbeddingManager
from ..graph.base import GraphManager
from ..llm.base import LLMManager
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class RetrieverManager:
    VALID_MODES = {"auto", "basic", "local", "global", "drift"}

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        graph_manager: GraphManager,
        llm_manager: Optional[LLMManager] = None,
        top_k: int = settings.retrieval_top_k,
        max_hops: int = settings.max_hops,
    ):
        self.embedding_manager = embedding_manager
        self.graph_manager = graph_manager
        self.llm_manager = llm_manager
        self.top_k = top_k
        self.max_hops = max_hops
        logger.info(f"Retriever Manager initialized (mode={settings.retrieval_mode_default}, max_hops={max_hops})")

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_graph: bool = True,
        retrieval_mode: Optional[str] = None,
        max_hops: Optional[int] = None,
        include_sources: bool = True,
    ) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        hops = max_hops or self.max_hops
        mode = (retrieval_mode or settings.retrieval_mode_default or "auto").lower()
        if mode not in self.VALID_MODES:
            mode = "auto"
        if not use_graph and mode in {"auto", "local", "global", "drift"}:
            mode = "basic"
        if mode == "auto":
            mode = await self.route_query(query, use_graph=use_graph)

        if mode == "basic":
            contexts = await self._basic_search(query, k)
        elif mode == "local":
            contexts = await self._local_search(query, k, hops)
        elif mode == "global":
            contexts = await self._global_search(query, k)
        elif mode == "drift":
            contexts = await self._drift_search(query, k, hops)
        else:
            contexts = await self._basic_search(query, k)

        deduped = self._dedupe_contexts(contexts)
        for item in deduped:
            item.setdefault("metadata", {})
            item["metadata"]["retrieval_mode"] = mode
        logger.info(f"Retrieved {len(deduped)} contexts with mode={mode}")
        return deduped[: max(k, 1) * 3 if include_sources else k]

    async def route_query(self, query: str, use_graph: bool = True) -> str:
        if not use_graph:
            return "basic"
        if self.llm_manager:
            prompt = f"""Classify the best GraphRAG retrieval mode for this user query.

Modes:
- basic: direct factual lookup from text chunks
- local: entity-specific reasoning using nearby graph entities and text units
- global: broad dataset/theme/summary questions using community reports
- drift: mixed or multi-hop questions that need global primer plus local follow-up

Return only JSON: {{"mode": "basic|local|global|drift"}}

Query: {query}"""
            try:
                response = await self.llm_manager.generate_prompt(prompt, temperature=0.0, max_tokens=128)
                data = self._parse_json(response)
                mode = str(data.get("mode", "")).lower()
                if mode in {"basic", "local", "global", "drift"}:
                    return mode
            except Exception as e:
                logger.warning(f"LLM query router failed; using heuristic router: {str(e)}")
        return self._heuristic_route(query)

    def _heuristic_route(self, query: str) -> str:
        query_lower = query.lower()
        global_terms = {
            "overview", "summarize", "summary", "themes", "topics", "dataset", "corpus",
            "overall", "all documents", "insights", "tổng quan", "tóm tắt", "chủ đề", "toàn bộ",
        }
        drift_terms = {
            "multi-hop", "multihop", "connect", "connection", "relationship", "related",
            "compare", "why", "cause", "between", "liên quan", "quan hệ", "so sánh", "tại sao",
        }
        if any(term in query_lower for term in global_terms):
            return "global"
        if any(term in query_lower for term in drift_terms):
            return "drift"
        if len(query.split()) <= 5:
            return "basic"
        return "local"

    async def _basic_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        embedding = await self.embedding_manager.embed_text(query)
        return await self._vector_search(
            index_name="textunit_embedding_vector",
            label="TextUnit",
            embedding_property="text_embedding",
            query_embedding=embedding.tolist(),
            top_k=max(top_k, 1),
        )

    async def _local_search(self, query: str, top_k: int, max_hops: int) -> List[Dict[str, Any]]:
        embedding = await self.embedding_manager.embed_text(query)
        seed_entities = await self._vector_search(
            index_name="entity_description_vector",
            label="Entity",
            embedding_property="description_embedding",
            query_embedding=embedding.tolist(),
            top_k=max(top_k * 2, 6),
        )
        if not seed_entities:
            return await self._basic_search(query, top_k)

        seed_ids = [item["id"] for item in seed_entities if item.get("id")]
        related_entities = await self._multi_hop_entities(seed_ids, max_hops=max_hops)
        all_entity_ids = list(dict.fromkeys(seed_ids + [item["id"] for item in related_entities if item.get("id")]))

        text_units = await self._text_units_for_entities(all_entity_ids, limit=max(top_k * 2, 8))
        reports = await self._reports_for_entities(all_entity_ids, limit=max(top_k, 3))

        relationship_context = self._relationship_summary_context(seed_entities, related_entities)
        return seed_entities[:top_k] + text_units + reports + relationship_context

    async def _global_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        embedding = await self.embedding_manager.embed_text(query)
        reports = await self._vector_search(
            index_name="community_report_vector",
            label="CommunityReport",
            embedding_property="content_embedding",
            query_embedding=embedding.tolist(),
            top_k=max(top_k * 3, 6),
        )
        if not reports:
            return await self._basic_search(query, top_k)

        points_context = await self._map_reduce_report_points(query, reports[: max(top_k * 2, 4)])
        return points_context + reports

    async def _drift_search(self, query: str, top_k: int, max_hops: int) -> List[Dict[str, Any]]:
        primer_reports = await self._global_search(query, max(2, min(top_k, 4)))
        followups = await self._generate_followups(query, primer_reports)
        contexts = [
            {
                "id": "drift_primer",
                "type": "drift_primer",
                "title": "DRIFT primer",
                "text": self._combine_context_text(primer_reports[:4]),
                "score": 1.0,
                "metadata": {"followups": followups},
            }
        ]
        for followup in followups[:3]:
            contexts.extend(await self._local_search(followup, max(2, top_k // 2), max_hops))
        return contexts

    async def _vector_search(
        self,
        index_name: str,
        label: str,
        embedding_property: str,
        query_embedding: List[float],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        retrieval_query = f"""
        CALL db.index.vector.queryNodes($index_name, $k, $embedding)
        YIELD node, score
        RETURN node.id AS id,
               labels(node)[0] AS label,
               coalesce(node.title, node.name, node.source_title, node.id) AS title,
               coalesce(node.text, node.full_content, node.summary, node.description, "") AS text,
               score,
               properties(node) AS metadata
        """
        records = await self.graph_manager.query(
            retrieval_query,
            {"index_name": index_name, "k": top_k, "embedding": query_embedding},
        )
        if records:
            return [self._record_to_context(record) for record in records]
        return await self._fallback_vector_scan(label, embedding_property, query_embedding, top_k)

    async def _fallback_vector_scan(
        self,
        label: str,
        embedding_property: str,
        query_embedding: List[float],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        query = f"""
        MATCH (node:{label})
        WHERE node.{embedding_property} IS NOT NULL
        RETURN node.id AS id,
               labels(node)[0] AS label,
               coalesce(node.title, node.name, node.source_title, node.id) AS title,
               coalesce(node.text, node.full_content, node.summary, node.description, "") AS text,
               node.{embedding_property} AS embedding,
               properties(node) AS metadata
        LIMIT 1000
        """
        records = await self.graph_manager.query(query)
        scored = []
        for record in records:
            score = self._cosine(query_embedding, record.get("embedding") or [])
            record["score"] = score
            scored.append(record)
        scored.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return [self._record_to_context(record) for record in scored[:top_k]]

    async def _multi_hop_entities(self, entity_ids: Sequence[str], max_hops: int) -> List[Dict[str, Any]]:
        if not entity_ids:
            return []
        hops = max(1, min(int(max_hops), 4))
        query = f"""
        MATCH path=(seed:Entity)-[:RELATED_TO*1..{hops}]-(entity:Entity)
        WHERE seed.id IN $entity_ids AND NOT entity.id IN $entity_ids
        WITH entity, min(length(path)) AS hop,
             collect(DISTINCT [
                rel IN relationships(path) |
                {
                    source: startNode(rel).title,
                    target: endNode(rel).title,
                    type: coalesce(rel.relationship_type, type(rel)),
                    description: rel.description,
                    weight: rel.weight
                }
             ]) AS rel_paths
        RETURN entity.id AS id,
               "Entity" AS label,
               entity.title AS title,
               entity.description AS text,
               1.0 / (hop + 1) AS score,
               properties(entity) AS metadata,
               hop,
               rel_paths[0] AS relationships
        ORDER BY score DESC, entity.degree DESC
        LIMIT 100
        """
        records = await self.graph_manager.query(query, {"entity_ids": list(entity_ids)})
        contexts = []
        for record in records:
            context = self._record_to_context(record)
            context["metadata"]["hop"] = record.get("hop")
            context["relationships"] = self._neo4j_relationships_to_dicts(record.get("relationships") or [])
            contexts.append(context)
        return contexts

    async def _text_units_for_entities(self, entity_ids: Sequence[str], limit: int) -> List[Dict[str, Any]]:
        if not entity_ids:
            return []
        query = """
        MATCH (tu:TextUnit)-[:MENTIONS]->(entity:Entity)
        WHERE entity.id IN $entity_ids
        WITH tu, collect(DISTINCT entity.title) AS matched_entities
        RETURN tu.id AS id,
               "TextUnit" AS label,
               coalesce(tu.source_title, tu.id) AS title,
               tu.text AS text,
               toFloat(size(matched_entities)) AS score,
               properties(tu) AS metadata,
               matched_entities
        ORDER BY score DESC, tu.position ASC
        LIMIT $limit
        """
        records = await self.graph_manager.query(query, {"entity_ids": list(entity_ids), "limit": limit})
        contexts = []
        for record in records:
            context = self._record_to_context(record)
            context["metadata"]["matched_entities"] = record.get("matched_entities", [])
            contexts.append(context)
        return contexts

    async def _reports_for_entities(self, entity_ids: Sequence[str], limit: int) -> List[Dict[str, Any]]:
        if not entity_ids:
            return []
        query = """
        MATCH (entity:Entity)-[:IN_COMMUNITY]->(community:Community)-[:HAS_REPORT]->(report:CommunityReport)
        WHERE entity.id IN $entity_ids
        WITH report, collect(DISTINCT entity.title) AS matched_entities
        RETURN report.id AS id,
               "CommunityReport" AS label,
               report.title AS title,
               coalesce(report.full_content, report.summary, "") AS text,
               coalesce(report.rank, 1.0) AS score,
               properties(report) AS metadata,
               matched_entities
        ORDER BY score DESC
        LIMIT $limit
        """
        records = await self.graph_manager.query(query, {"entity_ids": list(entity_ids), "limit": limit})
        contexts = []
        for record in records:
            context = self._record_to_context(record)
            context["metadata"]["matched_entities"] = record.get("matched_entities", [])
            contexts.append(context)
        return contexts

    async def _map_reduce_report_points(self, query: str, reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.llm_manager or not reports:
            return []
        report_text = self._combine_context_text(reports)
        prompt = f"""Use the community reports to extract the most important points for answering the query.
Return a concise bullet list. Include only grounded points.

Query: {query}

Community reports:
{report_text}
"""
        try:
            points = await self.llm_manager.generate_prompt(prompt, temperature=0.2, max_tokens=900)
        except Exception as e:
            logger.warning(f"Global map step failed: {str(e)}")
            return []
        return [
            {
                "id": "global_intermediate_points",
                "type": "global_points",
                "title": "Aggregated community points",
                "text": points,
                "score": 1.0,
                "metadata": {},
            }
        ]

    async def _generate_followups(self, query: str, primer_reports: List[Dict[str, Any]]) -> List[str]:
        fallback = [query]
        if not self.llm_manager:
            return fallback
        prompt = f"""Create 2-3 focused follow-up search queries for local GraphRAG search.
Return only JSON: {{"queries": ["..."]}}

Original query: {query}

Primer context:
{self._combine_context_text(primer_reports[:4])}
"""
        try:
            response = await self.llm_manager.generate_prompt(prompt, temperature=0.2, max_tokens=300)
            data = self._parse_json(response)
            queries = [str(item).strip() for item in data.get("queries", []) if str(item).strip()]
            return [query] + [item for item in queries if item.lower() != query.lower()]
        except Exception as e:
            logger.warning(f"DRIFT follow-up generation failed: {str(e)}")
            return fallback

    def _relationship_summary_context(
        self,
        seed_entities: List[Dict[str, Any]],
        related_entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        relationship_lines = []
        for entity in related_entities[:20]:
            for rel in entity.get("relationships", [])[:4]:
                source = rel.get("source", "")
                target = rel.get("target", "")
                rel_type = rel.get("type", "RELATED_TO")
                if source or target:
                    relationship_lines.append(f"{source} -[{rel_type}]-> {target}")
        if not relationship_lines:
            return []
        return [
            {
                "id": "local_relationships",
                "type": "relationships",
                "title": "Local graph relationships",
                "text": "\n".join(dict.fromkeys(relationship_lines)),
                "score": 0.9,
                "metadata": {"seed_entities": [item.get("title") for item in seed_entities[:8]]},
            }
        ]

    def _record_to_context(self, record: Dict[str, Any]) -> Dict[str, Any]:
        metadata = record.get("metadata") or {}
        metadata.pop("text_embedding", None)
        metadata.pop("description_embedding", None)
        metadata.pop("content_embedding", None)
        return {
            "id": record.get("id", ""),
            "type": record.get("label", metadata.get("type", "source")),
            "title": record.get("title") or record.get("id", ""),
            "text": record.get("text") or "",
            "score": float(record.get("score") or 0.0),
            "metadata": metadata,
        }

    def _neo4j_relationships_to_dicts(self, relationships: Sequence[Any]) -> List[Dict[str, Any]]:
        result = []
        for rel in relationships:
            try:
                props = dict(rel)
                result.append(
                    {
                        "source": props.get("source", ""),
                        "target": props.get("target", ""),
                        "type": props.get("relationship_type") or props.get("type") or "RELATED_TO",
                        "description": props.get("description", ""),
                        "weight": props.get("weight", 1.0),
                    }
                )
            except Exception:
                continue
        return result

    def _dedupe_contexts(self, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        result = []
        for context in contexts:
            key = context.get("id") or (context.get("type"), context.get("title"), context.get("text", "")[:80])
            if key in seen:
                continue
            seen.add(key)
            result.append(context)
        result.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return result

    def _combine_context_text(self, contexts: List[Dict[str, Any]]) -> str:
        parts = []
        for context in contexts:
            title = context.get("title", "")
            text = context.get("text", "")
            if title or text:
                parts.append(f"{title}\n{text}")
        return "\n\n".join(parts)[:8000]

    def _parse_json(self, value: str) -> Dict[str, Any]:
        start = value.find("{")
        end = value.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        return json.loads(value[start : end + 1])

    def _cosine(self, left: Sequence[float], right: Sequence[float]) -> float:
        if not left or not right:
            return 0.0
        left_array = np.array(left, dtype=float)
        right_array = np.array(right, dtype=float)
        denominator = np.linalg.norm(left_array) * np.linalg.norm(right_array)
        if denominator == 0:
            return 0.0
        return float(np.dot(left_array, right_array) / denominator)
