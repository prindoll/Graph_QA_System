from __future__ import annotations

from typing import Any, Dict, Optional

import aiohttp

from config.settings import settings
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class OllamaProvider:
    """Small async Ollama client for local chat/generation models."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model or settings.llm_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")

    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": settings.ollama_keep_alive,
            "options": {
                "temperature": settings.llm_temperature if temperature is None else temperature,
                "num_predict": max_tokens or settings.llm_max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"

        timeout = aiohttp.ClientTimeout(total=settings.llm_request_timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                text = await response.text()
                if response.status >= 400:
                    raise RuntimeError(
                        f"Ollama request failed ({response.status}) for model {self.model}: {text}"
                    )
                data = await response.json()

        result = str(data.get("response", "")).strip()
        if not result:
            logger.warning("Ollama returned an empty response for model %s", self.model)
        return result
