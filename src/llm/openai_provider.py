import asyncio
from typing import Optional

from openai import OpenAI

from config.settings import settings
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class OpenAIProvider:
    def __init__(self):
        api_key = settings.openai_api_key
        timeout = max(10, int(settings.llm_request_timeout_seconds or 120))
        if api_key:
            self.client = OpenAI(api_key=api_key, timeout=timeout)
        else:
            self.client = OpenAI(timeout=timeout)
        
        self.model = settings.llm_model
        logger.info(f"OpenAI Provider initialized with model: {self.model}")
    
    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        try:
            kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": settings.llm_temperature if temperature is None else temperature,
                "max_tokens": max_tokens or settings.llm_max_tokens,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = await asyncio.to_thread(self.client.chat.completions.create, **kwargs)
            return response.choices[0].message.content

        except Exception as e:
            if json_mode and "response_format" in str(e):
                logger.warning(f"OpenAI JSON mode unavailable, retrying without response_format: {str(e)}")
                return await self.generate(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=False,
                )
            logger.error(f"Error generating text: {str(e)}")
            raise
