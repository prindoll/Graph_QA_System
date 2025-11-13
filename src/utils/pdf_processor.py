from pathlib import Path
from typing import List, Dict, Any, Optional
import pdfplumber
import traceback
from .logger import setup_logger

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

logger = setup_logger(__name__)


class PDFProcessor:
    def __init__(self, chunk_size: int = 800, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            if PdfReader:
                logger.info(f"Extracting PDF with pypdf: {pdf_path}")
                text = self._extract_with_pypdf(pdf_path)
                if text and len(text.strip()) > 50:
                    logger.info(f"Successfully extracted {len(text)} chars with pypdf")
                    return text
            
            logger.info(f"Falling back to pdfplumber for: {pdf_path}")
            text = self._extract_text_pdfplumber(pdf_path)
            if text and len(text.strip()) > 50:
                logger.info(f"Successfully extracted {len(text)} chars with pdfplumber")
                return text
            
            logger.warning(f"No text extracted from {pdf_path}")
            return ""
                
        except Exception as e:
            logger.error(f"Error extracting PDF {pdf_path}: {str(e)}")
            return ""
    
    def _extract_with_pypdf(self, pdf_path: str) -> str:
        if not PdfReader:
            return ""
        
        try:
            text = ""
            reader = PdfReader(pdf_path)
            max_pages = min(len(reader.pages), 50)
            logger.info(f"Extracting {max_pages} pages with pypdf from {len(reader.pages)} total")
            
            for page_num in range(max_pages):
                try:
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text and len(page_text.strip()) > 5:
                        text += f"\n--- Page {page_num + 1} ---\n"
                        text += page_text
                except Exception as e:
                    logger.warning(f"Skipping page {page_num + 1}: {str(e)}")
                    continue
            
            return text
        except Exception as e:
            logger.warning(f"pypdf extraction failed: {str(e)}")
            return ""
    
    def _extract_text_pdfplumber(self, pdf_path: str) -> str:
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                max_pages = min(len(pdf.pages), 30)
                logger.info(f"Direct extraction with pdfplumber: first {max_pages} pages")
                
                for page_num in range(max_pages):
                    try:
                        page = pdf.pages[page_num]
                        page_text = page.extract_text()
                        if page_text and len(page_text.strip()) > 10:
                            text += f"\n--- Page {page_num + 1} ---\n"
                            text += page_text
                    except Exception as page_error:
                        logger.warning(f"Skipping page {page_num + 1}: {str(page_error)}")
                        continue
            
            return text
                
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {str(e)}")
            return ""
    
    def extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                metadata = {
                    "title": pdf.metadata.get("Title", "") if pdf.metadata else "",
                    "author": pdf.metadata.get("Author", "") if pdf.metadata else "",
                    "subject": pdf.metadata.get("Subject", "") if pdf.metadata else "",
                    "pages": len(pdf.pages) if pdf.pages else 0,
                    "file_path": str(Path(pdf_path).absolute()),
                    "file_name": Path(pdf_path).name,
                }
            
            logger.info(f"Extracted metadata from {pdf_path}")
            return metadata
            
        except Exception as e:
            logger.warning(f"Error extracting metadata from PDF {pdf_path}: {str(e)}")
            return {
                "title": "",
                "author": "",
                "subject": "",
                "pages": 0,
                "file_path": str(Path(pdf_path).absolute()),
                "file_name": Path(pdf_path).name,
            }
    
    def chunk_text(self, text: str, source_id: str) -> List[Dict[str, Any]]:
        chunks = []
        chunk_size = 800
        overlap = 100
        max_chunks = 100
        
        logger.info(f"Chunking text ({len(text)} chars) - max {max_chunks} chunks of {chunk_size} chars")
        
        start = 0
        chunk_count = 0
        
        while start < len(text) and chunk_count < max_chunks:
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            
            if chunk and len(chunk) > 20:
                chunks.append({
                    "content": chunk,
                    "source_id": source_id,
                    "start_pos": start,
                    "end_pos": end,
                    "chunk_size": len(chunk)
                })
                chunk_count += 1
            
            start = end - overlap
        
        logger.info(f"✓ Created {len(chunks)} chunks from {len(text)} characters (avg {len(text)//len(chunks) if chunks else 0} chars/chunk)")
        return chunks
    
    def process_pdf(self, pdf_path: str, source_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        try:
            if not Path(pdf_path).exists():
                logger.error(f"PDF file not found: {pdf_path}")
                return None
            
            text = self.extract_text_from_pdf(pdf_path)
            if not text or not text.strip():
                logger.warning(f"No text content extracted from {pdf_path}")
                return None
            
            metadata = self.extract_metadata(pdf_path)
            
            if source_id is None:
                source_id = Path(pdf_path).stem
            
            metadata["source_id"] = source_id
            chunks = self.chunk_text(text, source_id)
            
            if not chunks:
                logger.warning(f"No chunks created from {pdf_path}")
                return None
            
            result = {
                "source_id": source_id,
                "metadata": metadata,
                "full_text": text,
                "chunks": chunks,
                "total_chunks": len(chunks)
            }
            
            logger.info(f"Successfully processed PDF: {pdf_path} ({len(chunks)} chunks)")
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def process_multiple_pdfs(self, pdf_paths: List[str]) -> List[Dict[str, Any]]:
        results = []
        
        for pdf_path in pdf_paths:
            try:
                result = self.process_pdf(pdf_path)
                if result is not None: 
                    results.append(result)
                    logger.info(f"Processed {len(results)}/{len(pdf_paths)} PDFs successfully")
                else:
                    logger.warning(f"Skipped {pdf_path} due to errors")
                
            except Exception as e:
                logger.error(f"Skipping PDF {pdf_path}: {str(e)}")
                continue
        
        logger.info(f"Successfully processed {len(results)} out of {len(pdf_paths)} PDFs")
        return results
    
    def get_pdf_page_count(self, pdf_path: str) -> Optional[int]:
        try:
            if PdfReader:
                reader = PdfReader(pdf_path)
                return len(reader.pages)
            else:
                with pdfplumber.open(pdf_path) as pdf:
                    return len(pdf.pages)
        except Exception as e:
            logger.error(f"Error getting page count: {str(e)}")
            return None
    
    def generate_batch_ranges(self, pdf_path: str, batch_size: int = 50) -> List[tuple]:
        try:
            total_pages = self.get_pdf_page_count(pdf_path)
            
            if total_pages is None:
                logger.error(f"Could not determine page count for {pdf_path}")
                return []
            
            ranges = []
            for start in range(0, total_pages, batch_size):
                end = min(start + batch_size, total_pages)
                ranges.append((start, end))
            
            return ranges
            
        except Exception as e:
            logger.error(f"Error generating batch ranges: {str(e)}")
            return []
    
    def extract_pdf_batch(self, pdf_path: str, start_page: int = 0, end_page: Optional[int] = None) -> Optional[str]:
        try:
            if not Path(pdf_path).exists():
                logger.error(f"PDF file not found: {pdf_path}")
                return None
            
            if not PdfReader:
                logger.error("pypdf not installed")
                return None
            
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            
            if end_page is None:
                end_page = total_pages
            start_page = max(0, start_page)
            end_page = min(end_page, total_pages)
            
            if start_page >= end_page:
                logger.error(f"Invalid page range: {start_page}-{end_page}")
                return None
            
            batch_text = ""
            extracted_count = 0
            skipped_count = 0
            
            for page_num in range(start_page, end_page):
                try:
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    
                    if page_text and len(page_text.strip()) > 5:
                        batch_text += f"\n--- Page {page_num + 1} ---\n"
                        batch_text += page_text
                        extracted_count += 1
                    else:
                        skipped_count += 1
                        
                except Exception as e:
                    logger.debug(f"Skipping page {page_num + 1}: {str(e)}")
                    skipped_count += 1
                    continue
            
            logger.info(f"Batch complete: {extracted_count} pages extracted, {skipped_count} empty/skipped")
            
            return batch_text if batch_text.strip() else None
                
        except Exception as e:
            logger.error(f"Error in batch extraction: {str(e)}")
            return None
    
    def process_pdf_in_batches(self, pdf_path: str, batch_size: int = 50, source_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        try:
            if not Path(pdf_path).exists():
                logger.error(f"PDF file not found: {pdf_path}")
                return None
            
            batch_ranges = self.generate_batch_ranges(pdf_path, batch_size)
            
            if not batch_ranges:
                logger.error("No batch ranges generated")
                return None
            
            metadata = self.extract_metadata(pdf_path)
            
            if source_id is None:
                source_id = Path(pdf_path).stem
            
            metadata["source_id"] = source_id
            metadata["processing_mode"] = "batch"
            metadata["batch_size"] = batch_size
            metadata["num_batches"] = len(batch_ranges)
            
            all_text = ""
            total_extracted = 0
            
            logger.info(f"\n🔄 Processing {len(batch_ranges)} batches from {Path(pdf_path).name}...")
            
            for i, (start, end) in enumerate(batch_ranges, 1):
                batch_text = self.extract_pdf_batch(pdf_path, start, end)
                
                if batch_text:
                    all_text += batch_text
                    total_extracted += 1
                    logger.info(f"[{i}/{len(batch_ranges)}] ✓ Batch processed ({len(batch_text)} chars)")
                else:
                    logger.warning(f"[{i}/{len(batch_ranges)}] ⚠ Batch returned no content")
            
            logger.info(f"\nAll {len(batch_ranges)} batches processed!")
            logger.info(f"  Total extracted: {total_extracted}/{len(batch_ranges)} batches")
            logger.info(f"  Total text: {len(all_text)} characters\n")
            
            chunks = self.chunk_text(all_text, source_id)
            
            if not chunks:
                logger.warning(f"No chunks created from {pdf_path}")
                return None
            
            result = {
                "source_id": source_id,
                "metadata": metadata,
                "full_text": all_text,
                "chunks": chunks,
                "total_chunks": len(chunks),
                "batches_processed": len(batch_ranges),
                "batches_succeeded": total_extracted
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
