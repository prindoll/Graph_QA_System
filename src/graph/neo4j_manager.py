import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from neo4j import AsyncGraphDatabase

from config.settings import settings
from .base import BaseGraphManager
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class Neo4jManager(BaseGraphManager):
    VECTOR_INDEXES = {
        "textunit_embedding_vector": ("TextUnit", "text_embedding"),
        "entity_description_vector": ("Entity", "description_embedding"),
        "community_report_vector": ("CommunityReport", "content_embedding"),
    }
    FULLTEXT_INDEXES = {
        "textunit_fulltext": (
            "TextUnit",
            ["text", "source_title", "section_title", "heading_path"],
        ),
        "entity_fulltext": (
            "Entity",
            ["title", "name", "description", "type"],
        ),
        "community_report_fulltext": (
            "CommunityReport",
            ["title", "summary", "full_content", "rating_explanation"],
        ),
    }

    def __init__(
        self,
        uri: str = settings.neo4j_uri,
        user: str = settings.neo4j_user,
        password: str = settings.neo4j_password,
        database: str = settings.neo4j_database,
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
        logger.info(f"Neo4j Manager initialized with URI: {uri}, database: {database}")

    async def connect(self):
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            logger.info("Connected to Neo4j")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise

    async def disconnect(self):
        if self.driver:
            await self.driver.close()
            self.driver = None
            logger.info("Disconnected from Neo4j")

    def _session(self):
        if self.database:
            return self.driver.session(database=self.database)
        return self.driver.session()

    async def ensure_schema(self) -> Dict[str, Any]:
        if not self.driver:
            await self.connect()

        constraints = [
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (n:Document) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT textunit_id IF NOT EXISTS FOR (n:TextUnit) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT relationship_record_id IF NOT EXISTS FOR (n:RelationshipRecord) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT community_id IF NOT EXISTS FOR (n:Community) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT community_report_id IF NOT EXISTS FOR (n:CommunityReport) REQUIRE n.id IS UNIQUE",
        ]

        vector_indexes_created = 0
        vector_indexes_recreated = 0
        fulltext_indexes_created = 0
        async with self._session() as session:
            for statement in constraints:
                await session.run(statement)

            for index_name, (label, property_name) in self.VECTOR_INDEXES.items():
                if await self._drop_vector_index_if_dimension_changed(session, index_name):
                    vector_indexes_recreated += 1
                statement = f"""
                CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                FOR (n:{label}) ON (n.{property_name})
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: $dimension,
                    `vector.similarity_function`: 'cosine'
                }}}}
                """
                try:
                    await session.run(statement, dimension=settings.embedding_dimension)
                    vector_indexes_created += 1
                except Exception as e:
                    logger.warning(f"Could not create vector index {index_name}: {str(e)}")

            for index_name, (label, properties) in self.FULLTEXT_INDEXES.items():
                property_list = ", ".join(f"n.{property_name}" for property_name in properties)
                statement = f"""
                CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
                FOR (n:{label}) ON EACH [{property_list}]
                """
                try:
                    await session.run(statement)
                    fulltext_indexes_created += 1
                except Exception as e:
                    logger.warning(f"Could not create fulltext index {index_name}: {str(e)}")

        return {
            "status": "success",
            "constraints": len(constraints),
            "vector_indexes_created": vector_indexes_created,
            "vector_indexes_recreated": vector_indexes_recreated,
            "fulltext_indexes_created": fulltext_indexes_created,
        }

    async def _drop_vector_index_if_dimension_changed(self, session, index_name: str) -> bool:
        try:
            result = await session.run(
                """
                SHOW INDEXES
                YIELD name, type, options
                WHERE name = $name
                RETURN type, options
                """,
                name=index_name,
            )
            record = await result.single()
            if not record:
                return False

            options = record.get("options") or {}
            if isinstance(options, str):
                options = json.loads(options)
            index_config = options.get("indexConfig") or {}
            current_dimension = index_config.get("vector.dimensions")
            if current_dimension is None:
                return False
            if int(current_dimension) == int(settings.embedding_dimension):
                return False

            await session.run(f"DROP INDEX {index_name} IF EXISTS")
            logger.info(
                f"Dropped vector index {index_name} because dimension changed "
                f"from {current_dimension} to {settings.embedding_dimension}"
            )
            return True
        except Exception as e:
            logger.warning(f"Could not inspect vector index {index_name}: {str(e)}")
            return False

    async def add_nodes(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.driver:
            await self.connect()

        added_count = 0
        async with self._session() as session:
            grouped_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for node in nodes:
                node_id = node.get("id")
                if not node_id:
                    continue

                node_type = self._sanitize_label(node.get("type") or node.get("label_type") or "Entity")
                properties = self._sanitize_properties({k: v for k, v in node.items() if k not in {"id", "type"}})
                properties["id"] = node_id
                grouped_rows[node_type].append({"id": node_id, "props": properties})

            for node_type, rows in grouped_rows.items():
                query = f"""
                UNWIND $rows AS row
                MERGE (n:{node_type} {{id: row.id}})
                SET n += row.props
                RETURN count(n) AS count
                """
                for batch in self._batches(rows):
                    try:
                        result = await session.run(query, rows=batch)
                        record = await result.single()
                        added_count += int(record["count"]) if record else 0
                    except Exception as node_error:
                        logger.warning(f"Error adding {node_type} node batch: {str(node_error)}")

        logger.info(f"Merged {added_count}/{len(nodes)} nodes into Neo4j")
        return {"status": "success", "count": added_count}

    async def add_edges(self, edges: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.driver:
            await self.connect()

        added_count = 0
        async with self._session() as session:
            grouped_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for edge in edges:
                source = edge.get("source")
                target = edge.get("target")
                relation = self._sanitize_relationship(edge.get("type", "RELATED_TO"))
                if not source or not target:
                    continue

                edge_id = edge.get("id") or f"{relation}:{source}:{target}"
                properties = self._sanitize_properties({k: v for k, v in edge.items() if k not in {"source", "target", "type"}})
                properties["id"] = edge_id
                grouped_rows[relation].append(
                    {
                        "source": source,
                        "target": target,
                        "id": edge_id,
                        "props": properties,
                    }
                )

            for relation, rows in grouped_rows.items():
                query = f"""
                UNWIND $rows AS row
                MATCH (a {{id: row.source}}), (b {{id: row.target}})
                MERGE (a)-[r:{relation} {{id: row.id}}]->(b)
                SET r += row.props
                RETURN count(r) AS count
                """
                for batch in self._batches(rows):
                    try:
                        result = await session.run(query, rows=batch)
                        record = await result.single()
                        added_count += int(record["count"]) if record else 0
                    except Exception as edge_error:
                        logger.warning(f"Could not merge {relation} edge batch: {str(edge_error)}")

        logger.info(f"Merged {added_count}/{len(edges)} edges into Neo4j")
        return {"status": "success", "count": added_count}

    async def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self.driver:
            await self.connect()

        try:
            async with self._session() as session:
                result = await session.run(query, params or {})
                records = []
                async for record in result:
                    records.append(dict(record))
                return records
        except Exception as e:
            logger.error(f"Error querying Neo4j: {str(e)}")
            return []

    async def get_stats(self) -> Dict[str, Any]:
        if not self.driver:
            await self.connect()

        async with self._session() as session:
            nodes = await session.run("MATCH (n) RETURN count(n) as count")
            edges = await session.run("MATCH ()-[r]->() RETURN count(r) as count")
            labels = await session.run(
                """
                MATCH (n)
                UNWIND labels(n) AS label
                RETURN label, count(n) AS count
                ORDER BY label
                """
            )

            nodes_record = await nodes.single()
            edges_record = await edges.single()
            label_counts: Dict[str, int] = {}
            async for record in labels:
                label_counts[record["label"]] = record["count"]

        return {
            "nodes": nodes_record["count"],
            "edges": edges_record["count"],
            "labels": label_counts,
        }

    async def clear(self) -> bool:
        if not self.driver:
            await self.connect()

        try:
            async with self._session() as session:
                await session.run("MATCH (n) DETACH DELETE n")
            logger.info("Cleared Neo4j")
            return True
        except Exception as e:
            logger.error(f"Error clearing Neo4j: {str(e)}")
            raise

    def _sanitize_label(self, value: str) -> str:
        value = str(value or "Entity").replace(" ", "_").replace("-", "_")
        value = re.sub(r"[^A-Za-z0-9_]", "", value)
        return value or "Entity"

    def _sanitize_relationship(self, value: str) -> str:
        value = self._sanitize_label(value).upper()
        return value or "RELATED_TO"

    def _sanitize_properties(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        for key, value in properties.items():
            clean_key = re.sub(r"[^A-Za-z0-9_]", "_", str(key))
            if not clean_key:
                continue
            sanitized[clean_key] = self._to_neo4j_value(value)
        return sanitized

    def _batches(self, rows: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        batch_size = max(1, int(settings.batch_size or 100))
        return [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]

    def _to_neo4j_value(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, bool, int, float)):
            return value
        if hasattr(value, "item"):
            return value.item()
        if isinstance(value, list):
            if all(isinstance(item, (str, bool, int, float)) or item is None for item in value):
                return value
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, tuple):
            return self._to_neo4j_value(list(value))
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        return str(value)
