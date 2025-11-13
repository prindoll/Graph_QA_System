import json
import asyncio
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

from src.graph.llm_extractor import LLMKnowledgeExtractor
from src.utils.pdf_to_markdown import PDFToMarkdown
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class KnowledgeGraphBuilder:
    def __init__(self, save_intermediates: bool = True, output_dir: str = "data/processing", max_concurrent: int = 3):
        self.extractor = LLMKnowledgeExtractor()
        self.markdown_converter = PDFToMarkdown()
        self.save_intermediates = save_intermediates
        self.output_dir = Path(output_dir)
        self.max_concurrent = max_concurrent
        
        if save_intermediates:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Knowledge Graph Builder initialized (optimized with batch processing)")
    
    def _save_intermediate(self, doc_id: str, stage: str, content: Any) -> None:
        if not self.save_intermediates:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            doc_folder = self.output_dir / doc_id.replace(" ", "_")
            doc_folder.mkdir(parents=True, exist_ok=True)
            
            if stage == "markdown":
                file_path = doc_folder / f"markdown.md"
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                logger.info(f"Saved markdown: {file_path}")
            
            elif stage == "extraction":
                file_path = doc_folder / f"extraction.json"
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(content, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved extraction: {file_path}")
            
            elif stage == "graph":
                file_path = doc_folder / f"graph.json"
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(content, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved graph: {file_path}")
        
        except Exception as e:
            logger.warning(f"Failed to save intermediate {stage}: {str(e)}")

    async def build_and_persist(self, documents: List[Dict[str, Any]], graph_manager) -> Dict[str, Any]:
        try:
            all_nodes = []
            all_edges = []
            entity_id_map = {}
            
            for doc in documents:
                content = doc.get("content", "")
                doc_id = doc.get("id", "doc")
                
                if not content or len(content.strip()) < 100:
                    logger.warning(f"Skipping {doc_id}: insufficient content")
                    continue
                
                logger.info(f"Processing document: {doc_id} ({len(content)} chars)")
                
                markdown_text = self.markdown_converter.convert(content, doc_name=doc_id)
                logger.info(f"Converted to Markdown ({len(markdown_text)} chars)")
                
                self._save_intermediate(doc_id, "markdown", markdown_text)
                
                chunks = self._chunk_markdown(markdown_text, chunk_size=5000, max_chunks=5)
                logger.info(f"Split into {len(chunks)} chunks for extraction")
                
                logger.info(f"Extracting knowledge from {len(chunks)} chunks...")
                extraction_results = await self._extract_chunks_parallel(chunks)
                
                entities = []
                relationships = []
                context_nodes = []
                
                for result in extraction_results:
                    entities.extend(result.get("entities", []))
                    relationships.extend(result.get("relationships", []))
                    context_nodes.extend(result.get("context_nodes", []))
                
                entities_dict = self._deduplicate_entities(entities)
                relationships = self._deduplicate_relationships(relationships)
                extraction_data = {
                    "entities": entities_dict,
                    "relationships": relationships,
                    "context_nodes": context_nodes,
                    "counts": {
                        "entities": len(entities_dict),
                        "relationships": len(relationships),
                        "contexts": len(context_nodes)
                    }
                }
                self._save_intermediate(doc_id, "extraction", extraction_data)
                
                logger.info(f"Extracted {len(entities_dict)} entities, {len(relationships)} relationships, {len(context_nodes)} contexts")
                
                for entity_obj in entities_dict:
                    if isinstance(entity_obj, dict):
                        entity_name = entity_obj.get("name", "").strip()
                        entity_attrs = entity_obj 
                    else:
                        entity_name = str(entity_obj).strip()
                        entity_attrs = {"name": entity_name, "type": "concept"}
                    
                    if not entity_name or len(entity_name) < 1:
                        continue
                    norm_key = entity_name.lower().strip()
                    
                    if norm_key not in entity_id_map:
                        node_id = f"ent_{hash(norm_key) % 1000000}"
                        entity_id_map[norm_key] = {"id": node_id, "label": entity_name}
                        
                        description = ""
                        content_field = ""
                        try:
                            lines = markdown_text.split("\n")
                            matched_lines = []
                            context_window = 5
                            entity_lower = entity_name.lower()
                            
                            for i, line in enumerate(lines):
                                if entity_lower in line.lower() and len(line.strip()) > 0:
                                    start = max(0, i - context_window)
                                    end = min(len(lines), i + context_window + 1)
                                    context_lines = lines[start:end]
                                    context_text = " ".join([l.strip() for l in context_lines if l.strip()])
                                    matched_lines.append(context_text)
                            
                            if matched_lines:
                                description = matched_lines[0][:500]
                                content_field = " ".join(matched_lines)[:2500]
                        except:
                            pass
                        
                        if not content_field:
                            code_snippet = markdown_text[markdown_text.find(entity_name):markdown_text.find(entity_name)+1500] if entity_name in markdown_text else ""
                            if code_snippet:
                                content_field = code_snippet[:2500]
                        
                        node = {
                            "id": node_id,
                            "label": entity_name,
                            "type": entity_attrs.get("type", "concept"),
                            "source_doc": doc_id
                        }
                        
                        for attr_key in ["year", "domain", "inventor", "category"]:
                            if attr_key in entity_attrs and entity_attrs[attr_key]:
                                node[attr_key] = entity_attrs[attr_key]
                        
                        if description:
                            node["description"] = description
                        if content_field:
                            node["content"] = content_field
                        
                        all_nodes.append(node)
                
                edge_count = 0
                for rel in relationships:
                    subject = rel.get("subject", "").strip()
                    predicate = rel.get("predicate", "RELATES_TO").upper()
                    obj = rel.get("object", "").strip()
                    reason = rel.get("reason", "")
                    
                    if not subject or not obj:
                        continue
                    subject_norm = subject.lower().strip()
                    obj_norm = obj.lower().strip()
                    if subject_norm not in entity_id_map:
                        node_id = f"ent_{hash(subject_norm) % 1000000}"
                        entity_id_map[subject_norm] = {
                            "id": node_id,
                            "label": subject
                        }
                        node = {
                            "id": node_id,
                            "label": subject,
                            "type": "concept",
                            "source_doc": doc_id
                        }
                        all_nodes.append(node)
                    if obj_norm not in entity_id_map:
                        node_id = f"ent_{hash(obj_norm) % 1000000}"
                        entity_id_map[obj_norm] = {
                            "id": node_id,
                            "label": obj
                        }
                        node = {
                            "id": node_id,
                            "label": obj,
                            "type": "concept",
                            "source_doc": doc_id
                        }
                        all_nodes.append(node)
                    subject_id = entity_id_map[subject_norm]["id"]
                    obj_id = entity_id_map[obj_norm]["id"]
                    
                    edge = {
                        "source": subject_id,
                        "target": obj_id,
                        "type": predicate,
                        "label": f"{subject} {predicate.lower()} {obj}"
                    }
                    if reason:
                        edge["reason"] = reason
                    
                    all_edges.append(edge)
                    edge_count += 1
                    
                    logger.debug(f"  + Relation: {subject} -{predicate}-> {obj}" + (f" ({reason})" if reason else ""))
                
                logger.info(f"Created {edge_count} relationships")
                
                for ctx_node in context_nodes:
                    ctx_dict = {
                        "id": ctx_node.get("id"),
                        "type": "context",
                        "section": ctx_node.get("section", ""),
                        "pages": ctx_node.get("pages", ""),
                        "key_phrases": ctx_node.get("key_phrases", []),
                        "text_excerpt": ctx_node.get("text_excerpt", ""),
                        "source_doc": doc_id
                    }
                    all_nodes.append(ctx_dict)
                
                logger.info(f"Added {len(context_nodes)} context nodes")
            
            graph_data = {
                "nodes": all_nodes,
                "edges": all_edges,
                "summary": {
                    "total_nodes": len(all_nodes),
                    "total_edges": len(all_edges),
                    "timestamp": datetime.now().isoformat()
                }
            }
            self._save_intermediate("complete_graph", "graph", graph_data)
            
            if not all_nodes:
                logger.warning("No entities extracted")
                return {"status": "error", "message": "No entities extracted", "nodes_added": 0, "edges_added": 0}
            
            logger.info(f"Persisting to database: {len(all_nodes)} nodes, {len(all_edges)} edges")
            
            nodes_result = await graph_manager.add_nodes(all_nodes)
            edges_result = await graph_manager.add_edges(all_edges)
            
            logger.info(f"✓ Successfully persisted to database")
            
            return {"status": "success", "nodes_added": len(all_nodes), "edges_added": len(all_edges)}
        
        except Exception as e:
            logger.error(f"Error building knowledge graph: {str(e)}", exc_info=True)
            return {"status": "error", "message": str(e), "nodes_added": 0, "edges_added": 0}

    def _chunk_markdown(self, text: str, chunk_size: int = 5000, max_chunks: int = 5) -> List[str]:
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        current_pos = 0
        
        for i in range(max_chunks):
            if current_pos >= len(text):
                break
            
            end_pos = min(current_pos + chunk_size, len(text))
            
            if end_pos < len(text):
                split_pos = text.rfind("\n\n", current_pos, end_pos)
                if split_pos != -1 and split_pos > current_pos:
                    end_pos = split_pos
            
            chunk = text[current_pos:end_pos].strip()
            if chunk:
                chunks.append(chunk)
            
            current_pos = end_pos
        
        logger.debug(f"Split text into {len(chunks)} chunks")
        return chunks

    async def _extract_chunks_parallel(self, chunks: List[str]) -> List[Dict[str, Any]]:
        tasks = []
        
        for i, chunk in enumerate(chunks):
            logger.debug(f"  Queuing chunk {i+1}/{len(chunks)} for extraction")
            tasks.append(self.extractor.extract_knowledge(chunk))
        
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def bounded_extract(task):
            async with semaphore:
                return await task
        
        results = await asyncio.gather(*[bounded_extract(task) for task in tasks], return_exceptions=True)
        
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Chunk {i+1} extraction failed: {str(result)}")
            else:
                valid_results.append(result)
        
        logger.info(f"Extracted {len(valid_results)}/{len(chunks)} chunks successfully")
        return valid_results

    def _deduplicate_relationships(self, relationships: List[Dict[str, str]]) -> List[Dict[str, str]]:
        seen = set()
        unique = []
        
        for rel in relationships:
            key = (rel.get("subject", "").lower(), rel.get("predicate", "").upper(), rel.get("object", "").lower())
            if key not in seen:
                seen.add(key)
                unique.append(rel)
        
        return unique
    
    def _deduplicate_entities(self, entities: List) -> List[Dict[str, Any]]:
        seen = {}
        
        for entity in entities:
            if isinstance(entity, dict):
                entity_name = entity.get("name", "").strip().lower()
                if entity_name and entity_name not in seen:
                    seen[entity_name] = entity
            else:
                entity_name = str(entity).strip().lower()
                if entity_name and entity_name not in seen:
                    seen[entity_name] = {"name": entity_name.title(), "type": "concept"}
        
        return list(seen.values())
