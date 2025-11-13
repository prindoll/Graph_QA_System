import re
from typing import List, Dict, Any
from pathlib import Path

from .logger import setup_logger

logger = setup_logger(__name__)


class PDFToMarkdown:
    
    def __init__(self):
        pass
    
    def convert(self, text: str, doc_name: str = "Document") -> str:
        
        try:
            text = self._clean_text(text)
            
            markdown = f"# {doc_name}\n\n"
            
            sections = self._extract_sections(text)
            
            for section_title, section_content in sections:
                markdown += f"## {section_title}\n\n"
                
                paragraphs = self._extract_paragraphs(section_content)
                for para in paragraphs:
                    if para.strip():
                        markdown += f"{para}\n\n"
            
            logger.info(f"Converted to Markdown: {len(markdown)} chars")
            return markdown
        
        except Exception as e:
            logger.error(f"Error converting to Markdown: {str(e)}")
            return f"# {doc_name}\n\n{text}"
    
    def _clean_text(self, text: str) -> str:
        text = re.sub(r' +', ' ', text)
        
        text = re.sub(r'\n\n+', '\n', text)
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            if re.match(r'^\d+$', line):
                continue
            if len(line) < 3:
                continue
            if line.startswith('Page '):
                continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _extract_sections(self, text: str) -> List[tuple]:

        sections = []
        
        pattern = r'(?:Chapter|Section|Part|Chapter)\s+[\dIVX]+\.?\s*([^\n]+)'
        
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        
        if not matches:
            return [("Content", text)]
        
        for i, match in enumerate(matches):
            section_title = match.group(1).strip()
            start = match.end()
            end = matches[i+1].start() if i+1 < len(matches) else len(text)
            
            section_content = text[start:end]
            sections.append((section_title, section_content))
        
        return sections if sections else [("Content", text)]
    
    def _extract_paragraphs(self, text: str, max_para_size: int = 500) -> List[str]:

        paragraphs = []
        
        blocks = text.split('\n\n')
        
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            
            if len(block) > max_para_size:
                sentences = re.split(r'(?<=[.!?])\s+', block)
                current_para = ""
                
                for sentence in sentences:
                    if len(current_para) + len(sentence) < max_para_size:
                        current_para += sentence + " "
                    else:
                        if current_para:
                            paragraphs.append(current_para.strip())
                        current_para = sentence + " "
                
                if current_para:
                    paragraphs.append(current_para.strip())
            else:
                paragraphs.append(block)
        
        return paragraphs
    
    def save_markdown(self, markdown: str, output_path: str) -> bool:

        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown)
            
            logger.info(f"Saved Markdown to: {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving Markdown: {str(e)}")
            return False
