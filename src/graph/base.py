
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from enum import Enum

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class GraphType(Enum):
    NEO4J = "neo4j"
    NETWORKX = "networkx"


class BaseGraphManager(ABC):
    
    @abstractmethod
    async def add_nodes(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def add_edges(self, edges: List[Dict[str, Any]]) -> Dict[str, Any]:

        pass
    
    @abstractmethod
    async def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:

        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:

        pass
    
    @abstractmethod
    async def clear(self) -> bool:

        pass


class GraphManager:

    
    def __init__(self, graph_type: str = "neo4j", **kwargs):

        self.graph_type = graph_type.lower()
        self.backend: Optional[BaseGraphManager] = None
        
        if self.graph_type == GraphType.NEO4J.value:
            from .neo4j_manager import Neo4jManager
            self.backend = Neo4jManager(**kwargs)
            logger.info("GraphManager initialized with Neo4j backend")
        elif self.graph_type == GraphType.NETWORKX.value:
            from .networkx_manager import NetworkXManager
            self.backend = NetworkXManager(**kwargs)
            logger.info("GraphManager initialized with NetworkX backend")
        else:
            raise ValueError(f"Unsupported graph type: {graph_type}. Use 'neo4j' or 'networkx'")
    
    async def connect(self):
        if hasattr(self.backend, 'connect'):
            await self.backend.connect()
    
    async def disconnect(self):
        if hasattr(self.backend, 'disconnect'):
            await self.backend.disconnect()
    
    async def add_nodes(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:

        return await self.backend.add_nodes(nodes)
    
    async def add_edges(self, edges: List[Dict[str, Any]]) -> Dict[str, Any]:

        return await self.backend.add_edges(edges)
    
    async def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:

        return await self.backend.query(query, params)
    
    async def get_stats(self) -> Dict[str, Any]:

        return await self.backend.get_stats()
    
    async def clear(self) -> bool:

        return await self.backend.clear()
    
    async def batch_add_nodes(self, nodes: List[Dict[str, Any]], batch_size: int = 100) -> Dict[str, Any]:

        total_count = 0
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            result = await self.add_nodes(batch)
            total_count += result.get("count", 0)
            logger.debug(f"Added batch {i//batch_size + 1}: {result.get('count', 0)} nodes")
        
        logger.info(f"Batch add completed: {total_count} total nodes")
        return {"status": "success", "count": total_count}
    
    async def batch_add_edges(self, edges: List[Dict[str, Any]], batch_size: int = 100) -> Dict[str, Any]:

        total_count = 0
        for i in range(0, len(edges), batch_size):
            batch = edges[i:i + batch_size]
            result = await self.add_edges(batch)
            total_count += result.get("count", 0)
            logger.debug(f"Added batch {i//batch_size + 1}: {result.get('count', 0)} edges")
        
        logger.info(f"Batch add completed: {total_count} total edges")
        return {"status": "success", "count": total_count}
