#!/usr/bin/env python
"""
GraphRAG - FastAPI Server with Chat Interface
"""
import asyncio
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json

from src.core.graphrag import GraphRAG
from src.utils.pdf_processor import PDFProcessor
from src.graph.kg_builder import KnowledgeGraphBuilder
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

app = FastAPI(
    title="GraphRAG Chat API",
    description="Knowledge Graph RAG System with Chat Interface",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

rag_instance: Optional[GraphRAG] = None
chat_history: Dict[str, List[Dict[str, Any]]] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    top_k: int = 5
    use_graph: bool = True
    response_style: str = "detailed"


class ChatResponse(BaseModel):
    session_id: str
    message: str
    answer: str
    sources: List[str]
    timestamp: str
    query_type: Optional[str] = None


class UploadResponse(BaseModel):
    status: str
    message: str
    file_name: Optional[str] = None
    nodes_added: int = 0
    edges_added: int = 0


def get_rag() -> GraphRAG:
    global rag_instance
    if rag_instance is None:
        rag_instance = GraphRAG()
    return rag_instance


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta http-equiv="refresh" content="0; url=/static/index.html">
    </head>
    <body>
        <p>Redirecting to chat interface...</p>
    </body>
    </html>
    """


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        rag = get_rag()
        
        session_id = request.session_id or str(uuid.uuid4())
        
        if session_id not in chat_history:
            chat_history[session_id] = []
        
        result = await rag.query(
            query=request.message,
            top_k=request.top_k,
            use_graph=request.use_graph
        )
        
        chat_history[session_id].append({
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now().isoformat()
        })
        
        chat_history[session_id].append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("context", []),
            "timestamp": datetime.now().isoformat()
        })
        
        if len(chat_history[session_id]) > 50:
            chat_history[session_id] = chat_history[session_id][-50:]
        
        return ChatResponse(
            session_id=session_id,
            message=request.message,
            answer=result["answer"],
            sources=result.get("context", [])[:3],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        try:
            rag = get_rag()
            session_id = request.session_id or str(uuid.uuid4())
            
            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Searching knowledge base...'})}\n\n"
            
            context = await rag.retriever_manager.retrieve(
                query=request.message,
                top_k=request.top_k,
                use_graph=request.use_graph
            )
            
            yield f"data: {json.dumps({'type': 'sources', 'sources': context[:3]})}\n\n"
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating answer...'})}\n\n"
            
            answer = await rag.llm_manager.generate(
                query=request.message,
                context=context,
                response_style=request.response_style
            )
            
            words = answer.split(' ')
            for i, word in enumerate(words):
                yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"
                await asyncio.sleep(0.02)
            
            yield f"data: {json.dumps({'type': 'done', 'full_answer': answer})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    start_page: int = Form(0),
    end_page: int = Form(-1),
    clear_existing: bool = Form(False)
):
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        rag = get_rag()
        processor = PDFProcessor()
        kg_builder = KnowledgeGraphBuilder(save_intermediates=True)
        
        if clear_existing:
            await rag.graph_manager.clear()
        
        page_count = processor.get_pdf_page_count(str(file_path))
        
        if end_page == -1 or end_page > page_count:
            end_page = page_count
        
        batch_text = processor.extract_pdf_batch(str(file_path), start_page, end_page)
        
        if not batch_text:
            return UploadResponse(
                status="error",
                message="No text extracted from PDF",
                file_name=file.filename
            )
        
        docs = [{
            "id": f"{file_path.stem}_p{start_page+1}_{end_page}",
            "content": batch_text,
            "metadata": {
                "pages": f"{start_page+1}-{end_page}",
                "file": file.filename
            }
        }]
        
        result = await kg_builder.build_and_persist(docs, rag.graph_manager)
        
        return UploadResponse(
            status="success",
            message=f"Processed pages {start_page+1}-{end_page}",
            file_name=file.filename,
            nodes_added=result.get("nodes_added", 0),
            edges_added=result.get("edges_added", 0)
        )
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    try:
        rag = get_rag()
        stats = await rag.graph_manager.get_stats()
        return {
            "status": "success",
            "nodes": stats.get("nodes", 0),
            "edges": stats.get("edges", 0),
            "active_sessions": len(chat_history)
        }
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    if session_id not in chat_history:
        return {"messages": []}
    return {"messages": chat_history[session_id]}


@app.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    if session_id in chat_history:
        del chat_history[session_id]
    return {"status": "success", "message": "History cleared"}


@app.post("/api/clear")
async def clear_database():
    try:
        rag = get_rag()
        await rag.graph_manager.clear()
        return {"status": "success", "message": "Database cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
