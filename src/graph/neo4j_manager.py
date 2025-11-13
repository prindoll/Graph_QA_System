
from typing import List, Dict, Any
from neo4j import AsyncGraphDatabase
import logging

from config.settings import settings
from .base import BaseGraphManager
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class Neo4jManager(BaseGraphManager):
    
    def __init__(
        self,
        uri: str = settings.neo4j_uri,
        user: str = settings.neo4j_user,
        password: str = settings.neo4j_password
    ):

        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        logger.info(f"Neo4j Manager initialized with URI: {uri}")
    
    async def connect(self):
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            logger.info("Connected to Neo4j")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise
    
    async def disconnect(self):
        if self.driver:
            await self.driver.close()
            logger.info("Disconnected from Neo4j")
    
    async def add_nodes(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.driver:
            await self.connect()
        
        try:
            async with self.driver.session() as session:
                for node in nodes:
                    node_id = node.get("id", "unknown")
                    node_type = node.get("type", "Entity")
 
                    node_type = node_type.replace(" ", "_").replace("-", "_")
                    node_type = "".join(c for c in node_type if c.isalnum() or c == "_")
                    properties = {k: v for k, v in node.items() if k != "id"}

                    prop_assignments = ", ".join([f"n.{k} = ${k}" for k in properties.keys()])
                    if prop_assignments:
                        prop_assignments = ", " + prop_assignments
                    
                    query = f"""
                    MERGE (n:{node_type} {{id: $id}})
                    SET n.id = $id{prop_assignments}
                    """

                    params = {"id": node_id}
                    params.update(properties)
                    
                    try:
                        await session.run(query, **params)
                    except Exception as node_error:
                        logger.warning(f"Error adding node {node_id}: {str(node_error)}")
                        continue
            
            logger.info(f"Added {len(nodes)} nodes to Neo4j")
            return {"status": "success", "count": len(nodes)}
            
        except Exception as e:
            logger.error(f"Error adding nodes: {str(e)}")
            raise
    
    async def add_edges(self, edges: List[Dict]) -> Dict[str, Any]:
        if not self.driver:
            await self.connect()
        
        try:
            async with self.driver.session() as session:
                for edge in edges:
                    if isinstance(edge, dict):
                        source = edge.get("source", "")
                        target = edge.get("target", "")
                        relation = edge.get("type", "RELATES_TO")
                        confidence = edge.get("confidence", 1.0)
                    else:
                        if len(edge) == 3:
                            source, target, relation = edge
                            confidence = 1.0
                        else:
                            continue
                    
                    if not source or not target:
                        continue

                    relation = relation.replace(" ", "_").upper()
                    
                    query = f"""
                    MATCH (a {{id: $source}}), (b {{id: $target}})
                    CREATE (a)-[:{relation} {{confidence: $confidence}}]->(b)
                    """
                    try:
                        await session.run(
                            query,
                            source=source,
                            target=target,
                            confidence=confidence
                        )
                    except Exception as edge_error:
                        logger.warning(f"Could not create edge {source}-{relation}-{target}: {str(edge_error)}")
                        continue
            
            logger.info(f"Added {len(edges)} edges to Neo4j")
            return {"status": "success", "count": len(edges)}
            
        except Exception as e:
            logger.error(f"Error adding edges: {str(e)}")
            raise
    
    async def query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if not self.driver:
            await self.connect()
        
        try:
            async with self.driver.session() as session:
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
        
        try:
            async with self.driver.session() as session:
                nodes = await session.run("MATCH (n) RETURN count(n) as count")
                edges = await session.run("MATCH ()-[r]->() RETURN count(r) as count")
                
                nodes_record = await nodes.single()
                edges_record = await edges.single()
                
                return {
                    "nodes": nodes_record["count"],
                    "edges": edges_record["count"]
                }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            raise
    
    async def clear(self) -> bool:
        if not self.driver:
            await self.connect()
        
        try:
            async with self.driver.session() as session:
                await session.run("MATCH (n) DETACH DELETE n")
            logger.info("Cleared Neo4j")
            return True
        except Exception as e:
            logger.error(f"Error clearing Neo4j: {str(e)}")
            raise
