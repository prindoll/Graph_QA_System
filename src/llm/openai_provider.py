from typing import Optional

from openai import OpenAI

from config.settings import settings
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class OpenAIProvider:
    def __init__(self):
        api_key = settings.openai_api_key
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI()
        
        self.model = settings.llm_model
        logger.info(f"OpenAI Provider initialized with model: {self.model}")
    
    async def generate(self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature or settings.llm_temperature,
                max_tokens=max_tokens or settings.llm_max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise
