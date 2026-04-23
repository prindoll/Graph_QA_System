
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..llm.base import LLMManager
from ..embedding.base import EmbeddingManager
from ..graph.base import GraphManager
from ..graph.kg_builder import KnowledgeGraphBuilder
from ..retrieval.base import RetrieverManager
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class GraphRAG:
    
    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        graph_manager: Optional[GraphManager] = None,
        retriever_manager: Optional[RetrieverManager] = None,
    ):

        self.llm_manager = llm_manager or LLMManager()
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.graph_manager = graph_manager or GraphManager()
        self.retriever_manager = retriever_manager or RetrieverManager(
            embedding_manager=self.embedding_manager,
            graph_manager=self.graph_manager,
            llm_manager=self.llm_manager,
        )
        
        logger.info("GraphRAG initialized successfully")
    
    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        logger.info(f"Adding {len(documents)} documents to the system")
        
        try:
            builder = KnowledgeGraphBuilder(
                llm_manager=self.llm_manager,
                embedding_manager=self.embedding_manager,
            )
            result = await builder.build_and_persist(documents, self.graph_manager)
            
            logger.info(f"Successfully added documents")
            return result
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    async def query(
        self,
        query: str,
        top_k: int = 5,
        use_graph: bool = True,
        retrieval_mode: Optional[str] = None,
        max_hops: Optional[int] = None,
        include_sources: bool = True,
    ) -> Dict[str, Any]:
        logger.info(f"Processing query: {query}")
        
        try:
            context = await self.retriever_manager.retrieve(
                query=query,
                top_k=top_k,
                use_graph=use_graph,
                retrieval_mode=retrieval_mode,
                max_hops=max_hops,
                include_sources=include_sources,
            )
            
            answer = await self._generate_answer(query, context)
            resolved_mode = None
            if context:
                resolved_mode = context[0].get("metadata", {}).get("retrieval_mode")
            
            result = {
                "query": query,
                "answer": answer,
                "context": context,
                "retrieval_mode": resolved_mode or retrieval_mode or "auto",
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("Query processed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    async def _generate_answer(self, query: str, context: List[Dict[str, Any]]) -> str:
        try:
            if not context:
                return "I don't have enough information to answer this question."
            
            answer = await self.llm_manager.generate(
                query=query,
                context=context
            )
            
            return answer.strip()
        
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "Unable to generate an answer at this time."
    
    async def get_graph_stats(self) -> Dict[str, Any]:
        return await self.graph_manager.get_stats()
    
    async def clear_all(self) -> bool:
        await self.graph_manager.clear()
        logger.info("All data cleared")
        return True
