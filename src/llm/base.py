
from typing import List, Optional

from config.settings import settings
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class LLMManager:
    def __init__(self):
        provider_type = settings.llm_provider.lower()
        
        if provider_type == "openai":
            from .openai_provider import OpenAIProvider
            self.provider = OpenAIProvider()
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_type}")
        
        logger.info(f"LLM Manager initialized with provider: {provider_type}")
    
    async def generate(self, query: str, context: List[str], system_prompt: Optional[str] = None) -> str:
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()
        
        if isinstance(context, str):
            context_str = context
        else:
            if not context:
                logger.warning(f"No context retrieved for query: {query}")
                return "I don't have information to answer this question based on the knowledge base."
            
            context_str = "\n\n".join([f"[Reference {i+1}]\n{ctx}" for i, ctx in enumerate(context)])
        
        prompt = f"""{system_prompt}

Retrieved Context from Knowledge Base:
{context_str}

Question: {query}

Answer (use ONLY the provided context above):"""
        
        answer = await self.provider.generate(
            prompt=prompt,
            temperature=settings.llm_temperature,
            max_tokens=min(settings.llm_max_tokens, 1024)
        )
        
        return answer
    
    @staticmethod
    def _get_default_system_prompt() -> str:
        return """You are a technical assistant that answers questions STRICTLY based on the provided context.

IMPORTANT RULES:
1. ONLY use information from the provided context - do NOT use any outside knowledge
2. If the answer is not in the context, explicitly say: "This information is not available in the knowledge base"
3. Quote or reference the context when answering
4. Be accurate and cite sources from the context
5. Do not make up or assume information not in the context"""
