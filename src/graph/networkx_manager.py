
from typing import List, Dict, Any
import networkx as nx
import logging

from .base import BaseGraphManager
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class NetworkXManager(BaseGraphManager):
    
    def __init__(self):
        self.graph = nx.DiGraph()
        logger.info("NetworkX Manager initialized")
    
    async def add_nodes(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            for node in nodes:
                node_id = node.get("id")
                self.graph.add_node(
                    node_id,
                    **{k: v for k, v in node.items() if k != "id"}
                )
            
            logger.info(f"Added {len(nodes)} nodes to NetworkX")
            return {"status": "success", "count": len(nodes)}
            
        except Exception as e:
            logger.error(f"Error adding nodes: {str(e)}")
            raise
    
    async def add_edges(self, edges: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            added_count = 0
            for edge in edges:
                source = edge.get("source")
                target = edge.get("target")
                rel_type = edge.get("type", "related_to")
                confidence = edge.get("confidence", 1.0)
                
                if source and target:

                    if source.startswith("*_") or target.startswith("*_"):
                        continue
                    
                    self.graph.add_edge(
                        source,
                        target,
                        relation=rel_type,
                        confidence=confidence
                    )
                    added_count += 1
            
            logger.info(f"Added {added_count} edges to NetworkX")
            return {"status": "success", "count": added_count}
            
        except Exception as e:
            logger.error(f"Error adding edges: {str(e)}")
            raise
    
    async def query(self, query: str) -> List[Dict[str, Any]]:

        try:

            results = []
            for node, attrs in self.graph.nodes(data=True):
                results.append({"id": node, **attrs})
            return results
        except Exception as e:
            logger.error(f"Error querying NetworkX: {str(e)}")
            raise
    
    async def get_stats(self) -> Dict[str, Any]:
        try:
            return {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges()
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            raise
    
    async def clear(self) -> bool:

        try:
            self.graph.clear()
            logger.info("Cleared NetworkX graph")
            return True
        except Exception as e:
            logger.error(f"Error clearing graph: {str(e)}")
            raise
