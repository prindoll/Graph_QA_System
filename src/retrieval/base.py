from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import asyncio

from ..embedding.base import EmbeddingManager
from ..graph.base import GraphManager
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class RetrieverManager:
    def __init__(
        self, 
        embedding_manager: EmbeddingManager, 
        graph_manager: GraphManager, 
        top_k: int = 5,
        max_hops: int = 2,
        rerank_weight: float = 0.3,
        batch_size: int = 100
    ):
        self.embedding_manager = embedding_manager
        self.graph_manager = graph_manager
        self.top_k = top_k
        self.max_hops = max_hops
        self.rerank_weight = rerank_weight
        self.batch_size = batch_size
        self._embedding_cache = {}
        logger.info(f"Retriever Manager initialized (max_hops={max_hops}, rerank_weight={rerank_weight})")
    
    async def retrieve(self, query: str, top_k: Optional[int] = None, use_graph: bool = True) -> List[str]:
        k = top_k or self.top_k
        
        try:
            query_embedding = await self.embedding_manager.embed_text(query)
            
            if use_graph:
                results = await self._graph_retrieve_enhanced(query, query_embedding, max(k * 3, 15))
            else:
                results = await self._semantic_retrieve_scalable(query_embedding, max(k * 2, 10))
            
            reranked = await self._rerank_results(query, results, k)
            
            logger.info(f"Retrieved and reranked {len(reranked)} documents for query")
            return reranked
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise
    
    async def _get_total_node_count(self) -> int:
        try:
            result = await self.graph_manager.query("MATCH (n) RETURN count(n) as total")
            if result:
                return result[0].get("total", 0)
            return 0
        except:
            return 0
    
    async def _semantic_retrieve_scalable(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        try:
            total_nodes = await self._get_total_node_count()
            logger.info(f"Total nodes in database: {total_nodes}")
            
            all_candidates = []
            offset = 0
            
            while offset < total_nodes:
                batch_query = f"""
                MATCH (n) WHERE n.label IS NOT NULL 
                RETURN n.label as label, n.id as id, n.source_doc as source, 
                       n.description as description, n.content as content, 
                       n.type as type, n.domain as domain
                SKIP {offset} LIMIT {self.batch_size}
                """
                
                nodes = await self.graph_manager.query(batch_query)
                
                if not nodes:
                    break
                
                batch_candidates = await self._process_batch_with_content_embedding(nodes, query_embedding)
                all_candidates.extend(batch_candidates)
                
                offset += self.batch_size
                
                if len(all_candidates) >= top_k * 10:
                    break
            
            if not all_candidates:
                logger.warning("No candidates found")
                return []
            
            all_candidates.sort(key=lambda x: x["score"], reverse=True)
            top_candidates = all_candidates[:top_k]
            
            logger.info(f"Semantic retrieval found {len(top_candidates)} candidates from {len(all_candidates)} total")
            return top_candidates
            
        except Exception as e:
            logger.error(f"Scalable semantic retrieval error: {str(e)}", exc_info=True)
            return []
    
    async def _process_batch_with_content_embedding(
        self, 
        nodes: List[Dict], 
        query_embedding: np.ndarray
    ) -> List[Dict[str, Any]]:
        candidates = []
        
        texts_to_embed = []
        node_indices = []
        
        for i, node in enumerate(nodes):
            label = node.get("label") or ""
            content = node.get("content") or ""
            description = node.get("description") or ""
            
            combined_text = label
            if content:
                combined_text += " " + content[:500]
            elif description:
                combined_text += " " + description[:300]
            
            if combined_text.strip():
                cache_key = hash(combined_text[:200])
                if cache_key in self._embedding_cache:
                    embedding = self._embedding_cache[cache_key]
                    similarity = cosine_similarity([query_embedding], [embedding])[0][0]
                    candidates.append({
                        "id": node.get("id", ""),
                        "label": label,
                        "content": content,
                        "description": description,
                        "source": node.get("source", ""),
                        "type": node.get("type", ""),
                        "domain": node.get("domain", ""),
                        "score": float(similarity),
                        "combined_text": combined_text
                    })
                else:
                    texts_to_embed.append(combined_text)
                    node_indices.append(i)
        
        if texts_to_embed:
            embeddings = await self.embedding_manager.embed_texts(texts_to_embed)
            similarities = cosine_similarity([query_embedding], embeddings)[0]
            
            for j, idx in enumerate(node_indices):
                node = nodes[idx]
                label = node.get("label") or ""
                content = node.get("content") or ""
                description = node.get("description") or ""
                
                cache_key = hash(texts_to_embed[j][:200])
                self._embedding_cache[cache_key] = embeddings[j]
                
                if len(self._embedding_cache) > 10000:
                    keys_to_remove = list(self._embedding_cache.keys())[:1000]
                    for k in keys_to_remove:
                        del self._embedding_cache[k]
                
                candidates.append({
                    "id": node.get("id", ""),
                    "label": label,
                    "content": content,
                    "description": description,
                    "source": node.get("source", ""),
                    "type": node.get("type", ""),
                    "domain": node.get("domain", ""),
                    "score": float(similarities[j]),
                    "combined_text": texts_to_embed[j]
                })
        
        return candidates
    
    async def _graph_retrieve_enhanced(
        self, 
        query: str, 
        query_embedding: np.ndarray, 
        top_k: int
    ) -> List[Dict[str, Any]]:
        try:
            initial_candidates = await self._semantic_retrieve_scalable(query_embedding, top_k)
            
            if not initial_candidates:
                return []
            
            enhanced_results = []
            seen_ids = set()
            
            for candidate in initial_candidates:
                node_id = candidate.get("id", "")
                if node_id in seen_ids:
                    continue
                seen_ids.add(node_id)
                
                multi_hop_context = await self._multi_hop_traversal(
                    node_id, 
                    candidate.get("label", ""),
                    hops=self.max_hops
                )
                
                candidate["relationships"] = multi_hop_context.get("relationships", [])
                candidate["related_entities"] = multi_hop_context.get("related_entities", [])
                candidate["hop_scores"] = multi_hop_context.get("hop_scores", {})
                
                graph_boost = self._calculate_graph_boost(multi_hop_context)
                candidate["final_score"] = candidate["score"] * (1 + graph_boost * 0.2)
                
                enhanced_results.append(candidate)
                
                for related in multi_hop_context.get("related_entities", [])[:3]:
                    related_id = related.get("id", "")
                    if related_id and related_id not in seen_ids:
                        seen_ids.add(related_id)
                        related["score"] = candidate["score"] * 0.7
                        related["final_score"] = related["score"]
                        related["from_traversal"] = True
                        enhanced_results.append(related)
            
            enhanced_results.sort(key=lambda x: x.get("final_score", x.get("score", 0)), reverse=True)
            
            logger.info(f"Graph enhanced retrieval: {len(enhanced_results)} results with multi-hop context")
            return enhanced_results[:top_k]
            
        except Exception as e:
            logger.error(f"Graph enhanced retrieval error: {str(e)}", exc_info=True)
            return await self._semantic_retrieve_scalable(query_embedding, top_k)
    
    async def _multi_hop_traversal(
        self, 
        start_node_id: str, 
        start_label: str,
        hops: int = 2
    ) -> Dict[str, Any]:
        result = {
            "relationships": [],
            "related_entities": [],
            "hop_scores": {}
        }
        
        try:
            visited = {start_node_id}
            current_frontier = [start_node_id]
            
            for hop in range(1, hops + 1):
                if not current_frontier:
                    break
                
                hop_query = """
                MATCH (n)-[r]-(m) 
                WHERE n.id IN $node_ids AND m.label IS NOT NULL
                RETURN DISTINCT 
                    n.id as source_id,
                    n.label as source_label,
                    m.id as target_id, 
                    m.label as target_label,
                    m.content as target_content,
                    m.description as target_description,
                    m.type as target_type,
                    type(r) as rel_type,
                    properties(r) as rel_props
                LIMIT 50
                """
                
                hop_results = await self.graph_manager.query(hop_query, {"node_ids": current_frontier})
                
                next_frontier = []
                hop_score = 1.0 / (hop + 1)
                
                for rel in hop_results:
                    target_id = rel.get("target_id", "")
                    
                    if target_id and target_id not in visited:
                        visited.add(target_id)
                        next_frontier.append(target_id)
                        
                        rel_info = {
                            "source": rel.get("source_label", ""),
                            "target": rel.get("target_label", ""),
                            "type": rel.get("rel_type", "RELATED_TO"),
                            "hop": hop,
                            "properties": rel.get("rel_props", {})
                        }
                        result["relationships"].append(rel_info)
                        
                        entity_info = {
                            "id": target_id,
                            "label": rel.get("target_label", ""),
                            "content": rel.get("target_content", ""),
                            "description": rel.get("target_description", ""),
                            "type": rel.get("target_type", ""),
                            "hop_distance": hop,
                            "connected_via": rel.get("rel_type", "")
                        }
                        result["related_entities"].append(entity_info)
                        
                        result["hop_scores"][target_id] = hop_score
                
                current_frontier = next_frontier[:20]
            
            logger.debug(f"Multi-hop from '{start_label}': {len(result['relationships'])} rels, {len(result['related_entities'])} entities")
            
        except Exception as e:
            logger.warning(f"Multi-hop traversal error: {str(e)}")
        
        return result
    
    def _calculate_graph_boost(self, multi_hop_context: Dict[str, Any]) -> float:
        boost = 0.0
        
        num_relationships = len(multi_hop_context.get("relationships", []))
        boost += min(num_relationships * 0.05, 0.3)
        
        hop_scores = multi_hop_context.get("hop_scores", {})
        if hop_scores:
            boost += sum(hop_scores.values()) * 0.1
        
        rel_types = set(r.get("type", "") for r in multi_hop_context.get("relationships", []))
        important_rels = {"USES", "BASED_ON", "IMPROVES_OVER", "SIMILAR_TO", "IS_A", "PART_OF"}
        boost += len(rel_types & important_rels) * 0.1
        
        return min(boost, 0.5)
    
    async def _rerank_results(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]], 
        top_k: int
    ) -> List[str]:
        if not candidates:
            return []
        
        try:
            query_lower = query.lower()
            query_terms = set(query_lower.split())
            
            reranked = []
            
            for candidate in candidates:
                base_score = candidate.get("final_score", candidate.get("score", 0))
                
                label = (candidate.get("label") or "").lower()
                content = (candidate.get("content") or "").lower()
                description = (candidate.get("description") or "").lower()
                
                term_match_score = 0
                for term in query_terms:
                    if len(term) > 2:
                        if term in label:
                            term_match_score += 0.3
                        if term in content:
                            term_match_score += 0.1
                        if term in description:
                            term_match_score += 0.1
                
                coverage_score = 0
                if content:
                    coverage_score += 0.1
                if description:
                    coverage_score += 0.05
                if candidate.get("relationships"):
                    coverage_score += 0.1
                
                relationship_score = 0
                relationships = candidate.get("relationships", [])
                for rel in relationships[:5]:
                    rel_target = (rel.get("target") or "").lower()
                    if any(term in rel_target for term in query_terms if len(term) > 2):
                        relationship_score += 0.15
                
                final_score = (
                    base_score * (1 - self.rerank_weight) +
                    (term_match_score + coverage_score + relationship_score) * self.rerank_weight
                )
                
                candidate["rerank_score"] = final_score
                reranked.append(candidate)
            
            reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
            
            results = []
            for candidate in reranked[:top_k]:
                result_text = self._format_result(candidate)
                results.append(result_text)
            
            return results
            
        except Exception as e:
            logger.error(f"Reranking error: {str(e)}")
            return [self._format_result(c) for c in candidates[:top_k]]
    
    def _format_result(self, candidate: Dict[str, Any]) -> str:
        label = candidate.get("label", "")
        content = candidate.get("content", "")
        description = candidate.get("description", "")
        source = candidate.get("source", "")
        relationships = candidate.get("relationships", [])
        
        content = self._clean_text(content)
        description = self._clean_text(description)
        
        if content and len(content.strip()) > 0:
            content_preview = content[:300] + "..." if len(content) > 300 else content
            result = f"{label}: {content_preview}"
        elif description and len(description.strip()) > 0:
            desc_preview = description[:200] + "..." if len(description) > 200 else description
            result = f"{label}: {desc_preview}"
        else:
            result = label
        
        if relationships:
            rel_texts = []
            for rel in relationships[:3]:
                rel_type = rel.get("type", "RELATED_TO")
                target = rel.get("target", "")
                if target:
                    rel_texts.append(f"{rel_type} → {target}")
            
            if rel_texts:
                result += f" [Related: {', '.join(rel_texts)}]"
        
        if source:
            result += f" --- {source}"
        
        return result
    
    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        
        import re
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\-.,;:!?()\'\"@#$%&*+=/<>[\]{}|\\^~`]', '', text)
        text = text.strip()
        
        return text
    
    async def _semantic_retrieve(self, query_embedding: np.ndarray, top_k: int) -> List[str]:
        candidates = await self._semantic_retrieve_scalable(query_embedding, top_k)
        return [self._format_result(c) for c in candidates]
    
    async def _graph_retrieve(self, query_embedding: np.ndarray, top_k: int) -> List[str]:
        candidates = await self._graph_retrieve_enhanced("", query_embedding, top_k)
        return [self._format_result(c) for c in candidates]
