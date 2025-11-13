from typing import List, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..embedding.base import EmbeddingManager
from ..graph.base import GraphManager
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class RetrieverManager:
    def __init__(self, embedding_manager: EmbeddingManager, graph_manager: GraphManager, top_k: int = 5):
        self.embedding_manager = embedding_manager
        self.graph_manager = graph_manager
        self.top_k = top_k
        logger.info("Retriever Manager initialized")
    
    async def retrieve(self, query: str, top_k: Optional[int] = None, use_graph: bool = True) -> List[str]:
        k = top_k or self.top_k
        
        try:
            query_embedding = await self.embedding_manager.embed_text(query)
            
            if use_graph:
                results = await self._graph_retrieve(query_embedding, max(k * 2, 10))
            else:
                results = await self._semantic_retrieve(query_embedding, max(k * 2, 10))
            
            logger.info(f"Retrieved {len(results)} documents for query")
            return results[:k]
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise
    
    async def _semantic_retrieve(self, query_embedding: np.ndarray, top_k: int) -> List[str]:
        try:
            nodes = await self.graph_manager.query(
                "MATCH (n) WHERE n.label IS NOT NULL RETURN n.label as text, n.id as id, n.source_doc as source, n.description as description, n.content as content, n.type as type LIMIT 500"
            )
            
            if not nodes or len(nodes) == 0:
                logger.warning("No nodes found in Neo4j")
                return []
            
            logger.info(f"Found {len(nodes)} nodes for semantic retrieval")

            node_texts = []
            node_data = []
            
            for node in nodes:
                text = node.get("text") or node.get("label") or ""
                if text and len(text.strip()) > 0:
                    node_texts.append(text)
                    node_data.append(node)
            
            if not node_texts:
                logger.warning("No valid node texts for embedding")
                return []

            node_embeddings = await self.embedding_manager.embed_texts(node_texts)
            similarities = cosine_similarity([query_embedding], node_embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                node = node_data[idx]
                text = node.get("text") or node.get("label") or ""
                source = node.get("source") or ""
                description = node.get("description") or ""
                content = node.get("content") or ""
                
                if content and len(content.strip()) > 0:
                    result = f"{text}: {content}"
                elif description and len(description.strip()) > 0:
                    result = f"{text}: {description}"
                else:
                    result = f"{text}"
                
                if source:
                    result += f" (from {source})"
                
                results.append(result)
                logger.debug(f"  - {text} (score: {similarities[idx]:.3f})")
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic retrieval error: {str(e)}", exc_info=True)
            return []
    
    async def _graph_retrieve(self, query_embedding: np.ndarray, top_k: int) -> List[str]:
        try:
            nodes = await self.graph_manager.query(
                "MATCH (n) WHERE n.label IS NOT NULL RETURN n.label as text, n.id as id, n.type as type, n.description as description, n.content as content, n.year as year, n.domain as domain LIMIT 500"
            )
            
            if not nodes:
                logger.warning("No nodes found for graph retrieval")
                return []

            node_texts = []
            node_info = []
            
            for node in nodes:
                text = node.get("text") or ""
                if text and len(text.strip()) > 0:
                    node_texts.append(text)
                    node_info.append({
                        "text": text,
                        "id": node.get("id", ""),
                        "type": node.get("type", ""),
                        "description": node.get("description", ""),
                        "content": node.get("content", "")
                    })
            
            if not node_texts:
                return []

            node_embeddings = await self.embedding_manager.embed_texts(node_texts)
            similarities = cosine_similarity([query_embedding], node_embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                info = node_info[idx]
                text = info.get("text", "")
                description = info.get("description", "")
                content = info.get("content", "")
                node_id = info.get("id", "")
                score = similarities[idx]
                
                if score > 0.0:
                    if content and len(content.strip()) > 0:
                        result_text = f"{text}: {content}"
                    elif description and len(description.strip()) > 0:
                        result_text = f"{text}: {description}"
                    else:
                        result_text = f"{text}"
                    
                    try:
                        related = await self.graph_manager.query(
                            "MATCH (n {id: $node_id})-[r]-(m) RETURN m.label as related, type(r) as rel_type, properties(r) as r_props LIMIT 3",
                            {"node_id": node_id}
                        )

                        if related:
                            rel_texts = []
                            for rel in related:
                                related_text = rel.get("related", "")
                                rel_type = rel.get("rel_type", "RELATED_TO")
                                r_props = rel.get("r_props") or {}
                                reason = r_props.get("reason", "")

                                if related_text:
                                    if reason:
                                        rel_texts.append(f"{rel_type} {related_text} ({reason})")
                                    else:
                                        rel_texts.append(f"{rel_type} {related_text}")

                            if rel_texts:
                                result_text += " [Relationships: " + "; ".join(rel_texts[:3]) + "]"
                    except Exception:
                        pass
                    
                    results.append(result_text)
                    logger.debug(f"  - {text} (score: {score:.3f}, content_len: {len(result_text)})")
            
            logger.info(f"Graph retrieval found {len(results)} documents with relationships")
            return results
            
        except Exception as e:
            logger.error(f"Graph retrieval error: {str(e)}", exc_info=True)
            return await self._semantic_retrieve(query_embedding, top_k)
