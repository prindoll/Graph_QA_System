import inspect
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_core.embeddings import Embeddings

from config.settings import settings
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [embedding.tolist() for embedding in embeddings]

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()


class OpenAIEmbeddingsAdapter(Embeddings):
    def __init__(self, model_name: str, dimension: int, client: Optional[Any] = None):
        self.model_name = model_name
        self.dimension = dimension
        self.client = client

    def _get_client(self):
        if self.client is None:
            from openai import OpenAI

            kwargs: Dict[str, Any] = {}
            if settings.openai_api_key:
                kwargs["api_key"] = settings.openai_api_key
            self.client = OpenAI(**kwargs)
        return self.client

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self._get_client().embeddings.create(**self._request_kwargs(texts))
        return [list(item.embedding) for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def _request_kwargs(self, texts: List[str]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"model": self.model_name, "input": texts}
        if self.model_name.startswith("text-embedding-3") and self.dimension:
            kwargs["dimensions"] = self.dimension
        return kwargs


class EmbeddingManager:
    def __init__(
        self,
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
        dimension: Optional[int] = None,
        client: Optional[Any] = None,
    ):
        self.provider = (provider or settings.embedding_provider or "local").lower()
        self.model_name = model_name or settings.embedding_model
        self.dimension = dimension or settings.embedding_dimension
        self.client = client
        self.model = None
        self._langchain_embeddings: Embeddings

        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "local":
            self._init_local()
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

        logger.info(f"Embedding Manager initialized with provider={self.provider}, model={self.model_name}")

    def _init_openai(self) -> None:
        if self.client is None:
            from openai import AsyncOpenAI

            kwargs: Dict[str, Any] = {}
            if settings.openai_api_key:
                kwargs["api_key"] = settings.openai_api_key
            self.client = AsyncOpenAI(**kwargs)
        self._langchain_embeddings = OpenAIEmbeddingsAdapter(self.model_name, self.dimension)

    def _init_local(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_name)
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        self._langchain_embeddings = LocalSentenceTransformerEmbeddings(self.model)

    async def embed_text(self, text: str) -> np.ndarray:
        embeddings = await self.embed_texts([text])
        return embeddings[0]

    async def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        try:
            if not texts:
                return []
            if self.provider == "openai":
                return await self._embed_texts_openai(texts)
            return self._embed_texts_local(texts)
        except Exception as e:
            logger.error(f"Error embedding texts: {str(e)}")
            raise

    def _embed_texts_local(self, texts: List[str]) -> List[np.ndarray]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [np.asarray(embedding, dtype=float) for embedding in embeddings]

    async def _embed_texts_openai(self, texts: List[str]) -> List[np.ndarray]:
        results: List[np.ndarray] = []
        batch_size = max(1, int(settings.batch_size or 32))
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self.client.embeddings.create(**self._openai_request_kwargs(batch))
            if inspect.isawaitable(response):
                response = await response
            data = self._response_data(response)
            data.sort(key=lambda item: item.get("index", 0))
            results.extend(np.asarray(item["embedding"], dtype=float) for item in data)
        return results

    def _openai_request_kwargs(self, texts: List[str]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"model": self.model_name, "input": texts}
        if self.model_name.startswith("text-embedding-3") and self.dimension:
            kwargs["dimensions"] = self.dimension
        return kwargs

    def _response_data(self, response: Any) -> List[Dict[str, Any]]:
        raw_data = response.get("data", []) if isinstance(response, dict) else getattr(response, "data", [])
        result = []
        for index, item in enumerate(raw_data):
            if isinstance(item, dict):
                result.append(
                    {
                        "index": item.get("index", index),
                        "embedding": item.get("embedding", []),
                    }
                )
            else:
                result.append(
                    {
                        "index": getattr(item, "index", index),
                        "embedding": getattr(item, "embedding", []),
                    }
                )
        return result

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
        return self.dimension

    def as_langchain_embeddings(self) -> Embeddings:
        return self._langchain_embeddings
