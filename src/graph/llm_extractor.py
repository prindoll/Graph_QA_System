import json
import re
from typing import Dict, List, Any
import hashlib

from src.llm.openai_provider import OpenAIProvider
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class LLMKnowledgeExtractor:
    
    VALID_PREDICATES = {
        "TIME_COMPLEXITY", "SPACE_COMPLEXITY", "IMPROVES_OVER",
        "USES", "BASED_ON", "SIMILAR_TO", "HAS_PROPERTY",
        "YEAR", "PURPOSE", "DEVELOPED_BY", "APPLIES_TO",
        "RELATED_TO", "PROS", "CONS"
    }
    
    ENTITY_ATTRIBUTES = ["type", "year", "domain", "inventor", "category"]
    
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
            prompt = f"""Extract entities and relationships from the text about algorithms, data structures, and programming concepts.

INSTRUCTIONS:
1. Entities: algorithms, data structures, programming languages, code concepts, control flow statements, functions, methods, etc.
2. Include code constructs like: if statement, for loop, while loop, class, function definition, etc.
3. Relationships: [subject, predicate, object] triplets showing connections
4. Return ONLY valid JSON, no markdown code blocks, no extra text

EXAMPLE RELATIONSHIP TYPES:
TIME_COMPLEXITY, SPACE_COMPLEXITY, IMPROVES_OVER, USES, BASED_ON, SIMILAR_TO, 
HAS_PROPERTY, YEAR, PURPOSE, DEVELOPED_BY, APPLIES_TO, RELATED_TO, PROS, CONS

OUTPUT FORMAT:
{{
  "entities": ["entity1", "entity2", ...],
  "relationships": [
    {{"subject": "...", "predicate": "...", "object": "..."}},
    ...
  ]
}}

TEXT:
{text[:3500]}

Return only valid JSON:"""

            response = await self.provider.generate(prompt, temperature=0.1, max_tokens=1500)
            
            logger.debug(f"LLM response received: {len(response)} chars")
            
            data = self._parse_json_object(response)
            
            valid_entities = self._validate_entities(data.get("entities", []))
            valid_rels = self._validate_relationships(data.get("relationships", []))
            
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
            entities_str = ", ".join(entities[:20])
            
            prompt = f"""For these entities about algorithms: {entities_str}

Extract:
1. Entity attributes (type, year, domain, inventor)
2. Context: key phrases, page references, source information
3. Relationship reasons/details

Return JSON:
{{
  "entity_attributes": {{
    "entity_name": {{"type": "Algorithm/DataStructure/...", "year": 1956, "domain": "Graph Theory", "inventor": "...", "category": "..."}},
    ...
  }},
  "relationship_reasons": {{
    "subject_predicate_object": "reason or details",
    ...
  }},
  "context_information": {{
    "section": "Introduction",
    "pages": "pp. 120-125",
    "source_key_phrases": ["phrase1", "phrase2"]
  }}
}}

TEXT:
{text[:2000]}

Return only valid JSON:"""

            response = await self.provider.generate(prompt, temperature=0.1, max_tokens=1200)
            
            enriched_data = self._parse_json_object(response)
            
            enriched_entities = self._enrich_entities(entities, enriched_data.get("entity_attributes", {}))
            enriched_rels = self._enrich_relationships(relationships, enriched_data.get("relationship_reasons", {}))
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
            
            if len(e_clean) > 1 and e_clean.lower() not in seen:
                valid.append(e_clean)
                seen.add(e_clean.lower())
        
        return valid
    
    def _validate_relationships(self, relationships: List) -> List[Dict[str, str]]:
        valid = []
        seen = set()
        
        for rel in relationships:
            if not isinstance(rel, dict):
                continue
            
            subject = rel.get("subject", "").strip()
            predicate = rel.get("predicate", "").strip().upper()
            obj = rel.get("object", "").strip()
            
            if (subject and obj and predicate and len(subject) > 1 and len(obj) > 1 and predicate in self.VALID_PREDICATES):
                rel_key = (subject.lower(), predicate, obj.lower())
                if rel_key not in seen:
                    valid.append({"subject": subject, "predicate": predicate, "object": obj})
                    seen.add(rel_key)
        
        return valid
    
    def _enrich_entities(self, entities: List[str], attributes_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        enriched = []
        
        for entity in entities:
            entity_dict = {"name": entity, "type": "concept"}
            
            if entity in attributes_map:
                attrs = attributes_map[entity]
                if isinstance(attrs, dict):
                    for attr_key in ["type", "year", "domain", "inventor", "category"]:
                        if attr_key in attrs and attrs[attr_key]:
                            entity_dict[attr_key] = attrs[attr_key]
            
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
        
        if section or pages or source_phrases:
            context_node = {
                "id": f"context_{hash(section + pages) % 1000000}",
                "type": "context",
                "section": section,
                "pages": pages,
                "key_phrases": source_phrases[:5],
                "entities_count": len(entities),
                "text_excerpt": original_text[:300]
            }
            context_nodes.append(context_node)
        
        for entity in entities[:10]:
            entity_context = {
                "id": f"ctx_{hash(entity + section) % 1000000}",
                "type": "entity_context",
                "entity": entity,
                "section": section,
                "relevance": "high" if entity.lower() in original_text[:500].lower() else "medium"
            }
            context_nodes.append(entity_context)
        
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

