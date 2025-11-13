from typing import List, Dict, Any, Optional
import numpy as np

from sentence_transformers import SentenceTransformer
from config.settings import settings
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class EmbeddingManager:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.embedding_model
        self.model = SentenceTransformer(self.model_name)
        logger.info(f"Embedding Manager initialized with model: {self.model_name}")
    
    async def embed_text(self, text: str) -> np.ndarray:
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
            raise
    
    async def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding texts: {str(e)}")
            raise
    
    async def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            texts = [doc.get("content", "") for doc in documents]
            embeddings = await self.embed_texts(texts)
            
            result = []
            for doc, embedding in zip(documents, embeddings):
                doc_copy = doc.copy()
                doc_copy["embedding"] = embedding.tolist()
                result.append(doc_copy)
            
            logger.info(f"Embedded {len(result)} documents")
            return result
            
        except Exception as e:
            logger.error(f"Error embedding documents: {str(e)}")
            raise
    
    def get_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
