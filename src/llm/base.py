
from typing import List, Optional, Dict, Any, Union

from langchain_core.prompts import ChatPromptTemplate
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

    def get_chat_model(self, temperature: Optional[float] = None):
        if settings.llm_provider.lower() != "openai":
            raise ValueError(f"Unsupported LangChain chat provider: {settings.llm_provider}")

        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": settings.llm_model,
            "temperature": settings.llm_temperature if temperature is None else temperature,
            "max_tokens": settings.llm_max_tokens,
        }
        if settings.openai_api_key:
            kwargs["api_key"] = settings.openai_api_key
        return ChatOpenAI(**kwargs)

    async def generate_prompt(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        return await self.provider.generate(
            prompt=prompt,
            temperature=temperature if temperature is not None else settings.llm_temperature,
            max_tokens=max_tokens or settings.llm_max_tokens,
        )
    
    async def generate(
        self, 
        query: str, 
        context: List[str], 
        system_prompt: Optional[str] = None,
        response_style: str = "detailed"
    ) -> str:
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt(response_style)
        
        if isinstance(context, str):
            context_str = context
            has_relevant_context = bool(context.strip())
        else:
            if not context:
                logger.warning(f"No context retrieved for query: {query}")
                return "Sorry, I couldn't find any relevant information for this question in the database."
            
            context_str = self._format_context(context)
            has_relevant_context = self._check_context_relevance(query, context)
        
        if not has_relevant_context:
            logger.warning(f"Context not relevant to query: {query}")
            return "Sorry, I couldn't find any relevant information for this question in the database."
        
        query_type = self._detect_query_type(query)
        
        prompt = self._build_prompt(query, context_str, system_prompt, query_type)
        
        temperature = self._get_temperature_for_query(query_type)
        
        answer = await self.provider.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=settings.llm_max_tokens
        )
        
        return self._post_process_answer(answer, query_type)
    
    def _check_context_relevance(self, query: str, context: List[Union[str, Dict[str, Any]]]) -> bool:
        if context:
            return True
        query_words = set(query.lower().split())
        stop_words = {'what', 'is', 'the', 'a', 'an', 'how', 'to', 'do', 'does', 'can', 'could', 
                      'would', 'should', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 
                      'has', 'had', 'i', 'you', 'we', 'they', 'it', 'this', 'that', 'for', 
                      'of', 'in', 'on', 'at', 'by', 'with', 'from', 'and', 'or', 'but', 'if',
                      'là', 'gì', 'như', 'thế', 'nào', 'sao', 'tại', 'vì', 'để', 'có', 'được',
                      'và', 'hoặc', 'hay', 'của', 'trong', 'ngoài', 'trên', 'dưới'}
        
        query_keywords = query_words - stop_words
        
        if not query_keywords:
            return False
        
        context_text = " ".join(self._context_item_to_text(item) for item in context).lower()
        
        matched_keywords = sum(1 for word in query_keywords if word in context_text)
        match_ratio = matched_keywords / len(query_keywords) if query_keywords else 0
        
        if match_ratio < 0.3:
            return False
        
        return True
    
    def _format_context(self, context: List[Union[str, Dict[str, Any]]]) -> str:
        formatted_parts = []
        
        for i, ctx in enumerate(context):
            if isinstance(ctx, dict):
                title = ctx.get("title") or ctx.get("label") or ctx.get("id") or f"Source {i+1}"
                content = ctx.get("text") or ctx.get("content") or ctx.get("summary") or ""
                metadata = ctx.get("metadata") or {}
                relationships = ctx.get("relationships") or []
                source_type = ctx.get("type") or metadata.get("type") or "source"
                rel_text = ""
                if relationships:
                    rel_parts = []
                    for rel in relationships[:6]:
                        if isinstance(rel, dict):
                            rel_parts.append(
                                f"{rel.get('source', '')} -[{rel.get('type', 'RELATED_TO')}]-> {rel.get('target', '')}"
                            )
                    if rel_parts:
                        rel_text = "\nRelationships:\n" + "\n".join(rel_parts)
                formatted_parts.append(
                    f"[Source {i+1}: {source_type}]\n"
                    f"Title: {title}\n"
                    f"Content: {content}{rel_text}"
                )
                continue

            has_graph = "[Graph Context:" in ctx or "[Relationships:" in ctx
            
            if has_graph:
                parts = ctx.split("[Graph Context:")
                if len(parts) > 1:
                    main_content = parts[0].strip()
                    graph_info = "[Graph Context:" + parts[1]
                    formatted_parts.append(
                        f"[Source {i+1}]\n"
                        f"Content: {main_content}\n"
                        f"Related Information: {graph_info}"
                    )
                else:
                    parts = ctx.split("[Relationships:")
                    if len(parts) > 1:
                        main_content = parts[0].strip()
                        rel_info = "[Relationships:" + parts[1]
                        formatted_parts.append(
                            f"[Source {i+1}]\n"
                            f"Content: {main_content}\n"
                            f"Connections: {rel_info}"
                        )
                    else:
                        formatted_parts.append(f"[Source {i+1}]\n{ctx}")
            else:
                formatted_parts.append(f"[Source {i+1}]\n{ctx}")
        
        return "\n\n".join(formatted_parts)

    def _context_item_to_text(self, item: Union[str, Dict[str, Any]]) -> str:
        if isinstance(item, str):
            return item
        if not isinstance(item, dict):
            return str(item)
        parts = [
            str(item.get("title") or item.get("label") or ""),
            str(item.get("text") or item.get("content") or item.get("summary") or ""),
        ]
        for rel in item.get("relationships") or []:
            if isinstance(rel, dict):
                parts.append(str(rel.get("source", "")))
                parts.append(str(rel.get("target", "")))
                parts.append(str(rel.get("type", "")))
        return " ".join(part for part in parts if part)
    
    def _detect_query_type(self, query: str) -> str:
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["so sánh", "compare", "khác nhau", "difference", "vs", "versus", "giống", "similar"]):
            return "comparison"
        
        if any(word in query_lower for word in ["tại sao", "why", "vì sao", "nguyên nhân", "reason", "how come"]):
            return "explanation"
        
        if any(word in query_lower for word in ["như thế nào", "how", "cách", "làm sao", "steps", "process", "quy trình"]):
            return "how_to"
        
        if any(word in query_lower for word in ["là gì", "what is", "define", "định nghĩa", "meaning", "nghĩa là"]):
            return "definition"
        
        if any(word in query_lower for word in ["liệt kê", "list", "những", "các loại", "types", "examples", "ví dụ"]):
            return "listing"
        
        if any(word in query_lower for word in ["time complexity", "space complexity", "độ phức tạp", "big o", "O(n)"]):
            return "complexity"
        
        if any(word in query_lower for word in ["ưu điểm", "nhược điểm", "pros", "cons", "advantage", "disadvantage", "trade-off"]):
            return "pros_cons"
        
        return "general"
    
    def _get_temperature_for_query(self, query_type: str) -> float:
        temperature_map = {
            "definition": 0.3,
            "complexity": 0.2,
            "comparison": 0.4,
            "explanation": 0.5,
            "how_to": 0.4,
            "listing": 0.3,
            "pros_cons": 0.4,
            "general": 0.5
        }
        return temperature_map.get(query_type, 0.5)
    
    def _build_prompt(self, query: str, context_str: str, system_prompt: str, query_type: str) -> str:
        
        type_instructions = {
            "comparison": "Compare the items naturally, highlighting key differences and similarities.",
            
            "explanation": "Explain the concept clearly with reasoning.",
            
            "how_to": "Describe the process step by step.",
            
            "definition": "Define and explain the concept.",
            
            "listing": "List the items with brief descriptions.",
            
            "complexity": "State and explain the complexity analysis.",
            
            "pros_cons": "Discuss advantages and disadvantages.",
            
            "general": "Answer the question directly and thoroughly."
        }
        
        instruction = type_instructions.get(query_type, type_instructions["general"])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            (
                "human",
                "Task: {instruction}\n\nReference Information:\n{context}\n\nQuestion: {query}\n\nAnswer:",
            ),
        ])
        return prompt.format(instruction=instruction, context=context_str, query=query)
    
    def _post_process_answer(self, answer: str, query_type: str) -> str:
        answer = answer.strip()
        
        if answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1]
        
        lines = answer.split('\n')
        cleaned_lines = []
        for line in lines:
            if line.strip() and not line.strip().startswith('---'):
                cleaned_lines.append(line)
        
        answer = '\n'.join(cleaned_lines)
        
        return answer
    
    def _get_default_system_prompt(self, style: str = "detailed") -> str:
        base_prompt = """You are a technical assistant answering questions from a knowledge base.

RULES:
1. Use the provided context as your primary information source
2. Write naturally - do not use phrases like "Additionally," "In general," "Furthermore," "It's worth noting"
3. Do not mention "context," "sources," "knowledge base," or "Source 1/2/3" in your answer
4. Integrate all information seamlessly as if it's one unified answer
5. If context has relevant data, expand on it naturally to give a complete answer
6. If no relevant context, give a brief general answer without mentioning lack of data
7. Never say things like "Based on the context" or "According to the sources"

WRITING STYLE:
- Write as if you naturally know this information
- Use smooth transitions between ideas
- Be informative and helpful
- Sound like an expert explaining to a colleague"""

        if style == "concise":
            return base_prompt + """

LENGTH: Keep answers brief, 3-5 sentences."""
        
        elif style == "detailed":
            return base_prompt + """

LENGTH: Provide thorough explanations with relevant details."""
        
        elif style == "technical":
            return base_prompt + """

LENGTH: Technical depth with precise terminology."""
        
        return base_prompt
