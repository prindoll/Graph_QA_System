from typing import Any, Dict, List, Optional

from src.embedding.base import EmbeddingManager
from src.graph.indexer import LangChainGraphIndexer
from src.llm.base import LLMManager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class KnowledgeGraphBuilder:
    """Compatibility wrapper around the LangChain-based GraphRAG indexer."""

    def __init__(
        self,
        save_intermediates: bool = True,
        output_dir: str = "data/processing",
        max_concurrent: int = 3,
        llm_manager: Optional[LLMManager] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
    ):
        self.indexer = LangChainGraphIndexer(
            llm_manager=llm_manager,
            embedding_manager=embedding_manager,
            save_intermediates=save_intermediates,
            output_dir=output_dir,
            max_concurrent=max_concurrent,
        )
        logger.info("KnowledgeGraphBuilder initialized with LangChainGraphIndexer")

    async def build_and_persist(
        self,
        documents: List[Dict[str, Any]],
        graph_manager,
        clear: bool = False,
    ) -> Dict[str, Any]:
        return await self.indexer.index_documents(
            documents=documents,
            graph_manager=graph_manager,
            clear=clear,
        )

    async def index_pdf(
        self,
        pdf_path: str,
        graph_manager,
        start_page: int = 0,
        end_page: Optional[int] = None,
        clear: bool = False,
    ) -> Dict[str, Any]:
        return await self.indexer.index_pdf(
            pdf_path=pdf_path,
            graph_manager=graph_manager,
            start_page=start_page,
            end_page=end_page,
            clear=clear,
        )
