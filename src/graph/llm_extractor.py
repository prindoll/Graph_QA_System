import json
import re
from typing import Dict, List, Any
import hashlib

from src.llm.openai_provider import OpenAIProvider
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class LLMKnowledgeExtractor:
    
    CORE_PREDICATES = {
        "IS_A", "PART_OF", "HAS", "RELATED_TO", "USES", "CREATED_BY",
        "BELONGS_TO", "CONTAINS", "REQUIRES", "PRODUCES", "USED_FOR", "COMPARED_TO"
    }
    
    SECONDARY_PREDICATES = {
        "INVENTED_BY", "DEVELOPED_BY", "DESIGNED_BY", "IMPLEMENTED_BY",
        "DEPENDS_ON", "NEEDS", "PREREQUISITE_OF",
        "SIMILAR_TO", "CONNECTED_TO", "ASSOCIATED_WITH",
        "SOLVES", "APPLIES_TO", "UTILIZED_IN"
    }
    
    VALID_PREDICATES = CORE_PREDICATES | SECONDARY_PREDICATES
    
    ENTITY_IMPORTANCE = {
        "algorithm": 1.0, "method": 0.95, "technique": 0.9, "system": 0.9,
        "model": 0.85, "structure": 0.85, "function": 0.8, "process": 0.8,
        "concept": 0.7, "theory": 0.7, "principle": 0.7,
        "tool": 0.6, "framework": 0.6, "library": 0.6,
        "example": 0.3, "instance": 0.3, "case": 0.3
    }
    
    MAX_ENTITIES_PER_CHUNK = 25
    MAX_RELATIONSHIPS_PER_CHUNK = 30
    
    IMPORTANT_ENTITY_TYPES = {
        "algorithm", "data structure", "complexity", "technique", "method"
    }
    
    SKIP_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall",
        "this", "that", "these", "those", "it", "its",
        "i", "we", "you", "he", "she", "they", "them",
        "and", "or", "but", "if", "then", "else", "when", "where",
        "which", "who", "whom", "whose", "what", "how", "why",
        "all", "each", "every", "both", "few", "more", "most",
        "other", "some", "such", "no", "nor", "not", "only",
        "same", "so", "than", "too", "very", "just", "also",
        "example", "figure", "table", "chapter", "section", "page"
    }
    
    ENTITY_ATTRIBUTES = ["type", "year", "domain", "complexity"]
    
    def __init__(self):
        self.provider = OpenAIProvider()
        self.extraction_cache = {}
        logger.info("LLM Knowledge Extractor initialized with rich attributes support")
    
    async def extract_knowledge(self, text: str) -> Dict[str, Any]:
        try:
            if not text or len(text.strip()) < 50:
                logger.warning("Text too short for extraction (< 50 chars)")
                return self._empty_result()
            
            text_hash = self._hash_text(text)
            if text_hash in self.extraction_cache:
                return self.extraction_cache[text_hash]
            
            logger.info("Extracting process")
            result = await self._extract_combined_enriched(text)
            
            self.extraction_cache[text_hash] = result
            
            return result
        
        except Exception as e:
            logger.error(f"Extraction error: {str(e)}", exc_info=True)
            return self._empty_result()
    
    @staticmethod
    def _hash_text(text: str, length: int = 500) -> str:
        return hashlib.md5(text[:length].encode()).hexdigest()
    
    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {"entities": [], "relationships": [], "context_nodes": [], "entity_count": 0, "relationship_count": 0, "context_count": 0}
    
    async def _extract_combined(self, text: str) -> Dict[str, Any]:
        try:
            prompt = f"""Extract ALL important technical concepts and their relationships from this text.

ENTITIES TO EXTRACT:
- Algorithms (Quick Sort, Binary Search, Dijkstra, BFS, DFS, etc.)
- Data Structures (Array, Tree, Graph, Hash Table, Stack, Queue, Heap, etc.)
- Complexity notations (O(n), O(log n), O(n²), Θ(n), etc.)
- Techniques (Divide and Conquer, Dynamic Programming, Greedy, etc.)
- Concepts (recursion, iteration, traversal, sorting, searching, etc.)
- Authors/Inventors if mentioned
- Years/Dates if mentioned

RELATIONSHIPS TO EXTRACT:
- IS_A: "X is a type of Y", "X is a Y"
- PART_OF: "X is part of Y", "X belongs to Y"  
- USES: "X uses Y", "X is based on Y", "X implements Y"
- HAS: "X has Y", "X contains Y"
- REQUIRES: "X requires Y", "X needs Y"
- TIME_COMPLEXITY: "X runs in O(...)", "X has time complexity O(...)"
- SPACE_COMPLEXITY: "X uses O(...) space"
- CREATED_BY: "X was created/invented by Y"
- COMPARED_TO: "X is faster/better than Y", "X vs Y"
- RELATED_TO: general relationship between concepts

OUTPUT FORMAT:
{{
  "entities": ["Quick Sort", "O(n log n)", "Divide and Conquer", "Array", "Partition"],
  "relationships": [
    {{"subject": "Quick Sort", "predicate": "TIME_COMPLEXITY", "object": "O(n log n)"}},
    {{"subject": "Quick Sort", "predicate": "USES", "object": "Divide and Conquer"}},
    {{"subject": "Quick Sort", "predicate": "USES", "object": "Partition"}}
  ]
}}

RULES:
1. Extract up to 25 entities
2. Extract up to 30 relationships
3. Include BOTH well-known terms AND specific terms from the text
4. Relationships must connect entities that exist in the text

TEXT:
{text[:4000]}

Return ONLY valid JSON:"""

            response = await self.provider.generate(prompt, temperature=0.2, max_tokens=1500)
            
            logger.debug(f"LLM response received: {len(response)} chars")
            
            data = self._parse_json_object(response)
            
            valid_entities = self._validate_entities(data.get("entities", []))
            valid_rels = self._validate_relationships(data.get("relationships", []), valid_entities)
            
            return {
                "entities": valid_entities,
                "relationships": valid_rels,
                "context_nodes": [],
                "entity_count": len(valid_entities),
                "relationship_count": len(valid_rels),
                "context_count": 0
            }
        
        except Exception as e:
            logger.error(f"Combined extraction failed: {str(e)}")
            return self._empty_result()
    
    async def _extract_combined_enriched(self, text: str) -> Dict[str, Any]:
        try:
            basic_extraction = await self._extract_combined(text)
            enriched_data = await self._extract_attributes_and_context(text, basic_extraction)
            
            return enriched_data
        
        except Exception as e:
            logger.error(f"Enriched extraction failed: {str(e)}")
            return self._empty_result()
    
    async def _extract_attributes_and_context(self, text: str, basic_extraction: Dict[str, Any]) -> Dict[str, Any]:
        try:
            entities = basic_extraction.get("entities", [])
            relationships = basic_extraction.get("relationships", [])
            
            if not entities:
                return basic_extraction
                
            entities_str = ", ".join(entities[:10])
            
            prompt = f"""For these technical entities: {entities_str}

Provide brief attributes (ONLY if clearly known):

Return JSON:
{{
  "entity_attributes": {{
    "Quick Sort": {{"type": "Algorithm", "domain": "Sorting", "complexity": "O(n log n)"}},
    "Binary Search Tree": {{"type": "DataStructure", "domain": "Trees"}}
  }}
}}

RULES:
- Only include attributes you are certain about
- Skip unknown attributes (don't guess)
- Keep it minimal

TEXT CONTEXT:
{text[:1500]}

Return ONLY valid JSON:"""

            response = await self.provider.generate(prompt, temperature=0.1, max_tokens=800)
            
            enriched_data = self._parse_json_object(response)
            
            enriched_entities = self._enrich_entities(entities, enriched_data.get("entity_attributes", {}))
            enriched_rels = relationships
            context_nodes = self._create_context_nodes(enriched_data.get("context_information", {}), entities, text)
            
            return {
                "entities": enriched_entities,
                "relationships": enriched_rels,
                "context_nodes": context_nodes,
                "entity_count": len(enriched_entities),
                "relationship_count": len(enriched_rels),
                "context_count": len(context_nodes)
            }
        
        except Exception as e:
            logger.warning(f"Attribute extraction failed: {str(e)}, using basic extraction")
            return basic_extraction
    
    def _validate_entities(self, entities: List) -> List[str]:
        valid = []
        seen = set()
        
        for e in entities:
            if not isinstance(e, str):
                continue
            
            e_clean = e.strip()
            
            if len(e_clean) < 2 or e_clean.lower() in seen:
                continue
            
            if len(e_clean) > 100:
                continue
            
            if e_clean.lower() in self.SKIP_WORDS:
                continue
            
            valid.append(e_clean)
            seen.add(e_clean.lower())
            
            if len(valid) >= self.MAX_ENTITIES_PER_CHUNK:
                break
        
        return valid
    
    def _calculate_entity_importance(self, entity: str) -> float:
        score = 0.5
        
        for entity_type, importance in self.ENTITY_IMPORTANCE.items():
            if entity_type.lower() in entity.lower():
                score = max(score, importance)
                break
        
        if entity[0].isupper():
            score += 0.1
        
        if len(entity) > 3:
            score += 0.1
        
        words = entity.split()
        if 1 <= len(words) <= 4:
            score += 0.1
        elif len(words) > 6:
            score -= 0.2
        
        return min(score, 1.0)
    
    def _validate_relationships(self, relationships: List, valid_entities: List[str] = None) -> List[Dict[str, str]]:
        valid = []
        seen = set()
        
        for rel in relationships:
            if not isinstance(rel, dict):
                continue
            
            subject = rel.get("subject", "").strip()
            predicate = rel.get("predicate", "").strip().upper().replace(" ", "_")
            obj = rel.get("object", "").strip()
            
            if not (subject and obj and predicate and len(subject) > 1 and len(obj) > 1):
                continue
            
            if len(subject) > 100 or len(obj) > 100:
                continue
            
            if predicate not in self.VALID_PREDICATES:
                predicate = "RELATED_TO"
            
            rel_key = (subject.lower(), predicate, obj.lower())
            if rel_key not in seen:
                valid.append({
                    "subject": subject, 
                    "predicate": predicate, 
                    "object": obj
                })
                seen.add(rel_key)
                
                if len(valid) >= self.MAX_RELATIONSHIPS_PER_CHUNK:
                    break
        
        return valid
    
    def _map_to_core_predicate(self, predicate: str) -> str:
        mapping = {
            "INVENTED_BY": "CREATED_BY",
            "DEVELOPED_BY": "CREATED_BY", 
            "DESIGNED_BY": "CREATED_BY",
            "IMPLEMENTED_BY": "CREATED_BY",
            "DEPENDS_ON": "REQUIRES",
            "NEEDS": "REQUIRES",
            "PREREQUISITE_OF": "REQUIRES",
            "SIMILAR_TO": "RELATED_TO",
            "CONNECTED_TO": "RELATED_TO",
            "ASSOCIATED_WITH": "RELATED_TO",
            "SOLVES": "USED_FOR",
            "APPLIES_TO": "USED_FOR",
            "UTILIZED_IN": "USED_FOR",
        }
        return mapping.get(predicate, "RELATED_TO")
    
    def _enrich_entities(self, entities: List[str], attributes_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        enriched = []
        
        for entity in entities:
            entity_dict = {"name": entity, "type": "concept"}
            
            if entity in attributes_map:
                attrs = attributes_map[entity]
                if isinstance(attrs, dict):
                    for attr_key in ["type", "year", "domain", "inventor", "category", "complexity", "use_case"]:
                        if attr_key in attrs and attrs[attr_key]:
                            entity_dict[attr_key] = attrs[attr_key]
            
            for key, attrs in attributes_map.items():
                if key.lower() == entity.lower() and isinstance(attrs, dict):
                    for attr_key in ["type", "year", "domain", "inventor", "category", "complexity", "use_case"]:
                        if attr_key in attrs and attrs[attr_key] and attr_key not in entity_dict:
                            entity_dict[attr_key] = attrs[attr_key]
                    break
            
            enriched.append(entity_dict)
        
        return enriched
    
    def _enrich_relationships(self, relationships: List[Dict[str, str]], reasons_map: Dict[str, str]) -> List[Dict[str, Any]]:
        enriched = []
        
        for rel in relationships:
            rel_dict = {"subject": rel.get("subject"), "predicate": rel.get("predicate"), "object": rel.get("object")}
            
            rel_key = f"{rel['subject']}_{rel['predicate']}_{rel['object']}"
            if rel_key in reasons_map:
                rel_dict["reason"] = reasons_map[rel_key]
            
            enriched.append(rel_dict)
        
        return enriched
    
    def _create_context_nodes(self, context_info: Dict[str, Any], entities: List[str], original_text: str) -> List[Dict[str, Any]]:
        context_nodes = []
        
        section = context_info.get("section", "Main")
        pages = context_info.get("pages", "")
        source_phrases = context_info.get("source_key_phrases", [])
        
        if section or pages:
            context_node = {
                "id": f"context_{hash(section + pages) % 1000000}",
                "type": "context",
                "section": section,
                "pages": pages,
                "key_phrases": source_phrases[:3],
                "entities_count": len(entities),
                "text_excerpt": original_text[:200]
            }
            context_nodes.append(context_node)
        
        return context_nodes
    
    def _parse_json_object(self, response: str) -> Dict:
        try:
            response = response.strip()
            
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join([l for l in lines if not l.startswith("```")])
                response = response.strip()
            
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                json_str = match.group(0).strip()
                
                try:
                    parsed = json.loads(json_str)
                    return parsed
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {str(e)}")
                    
                    try:
                        json_str_fixed = json_str.replace("'", '"')
                        json_str_fixed = re.sub(r',(\s*[}\]])', r'\1', json_str_fixed)
                        
                        parsed = json.loads(json_str_fixed)
                        return parsed
                    except:
                        logger.info("Error decode")
                        return {"entities": [], "relationships": []}
            
            logger.warning("No JSON object found in response")
            return {"entities": [], "relationships": []}
        
        except Exception as e:
            logger.error(f"JSON parse error: {str(e)}")
            return {"entities": [], "relationships": []}

