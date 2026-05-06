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


class ChatHistoryItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    top_k: int = 5
    use_graph: bool = True
    response_style: str = "detailed"
    retrieval_mode: str = "auto"
    max_hops: int = 2
    include_sources: bool = True
    history: Optional[List[ChatHistoryItem]] = None


class ChatResponse(BaseModel):
    session_id: str
    message: str
    answer: str
    sources: List[Dict[str, Any]]
    timestamp: str
    query_type: Optional[str] = None
    retrieval_mode: Optional[str] = None


class UploadResponse(BaseModel):
    status: str
    message: str
    file_name: Optional[str] = None
    nodes_added: int = 0
    edges_added: int = 0
    documents: int = 0
    text_units: int = 0
    entities: int = 0
    relationships: int = 0
    communities: int = 0
    community_reports: int = 0
    vector_indexes_created: int = 0


def get_rag() -> GraphRAG:
    global rag_instance
    if rag_instance is None:
        rag_instance = GraphRAG()
    return rag_instance


def build_history_text(
    history: Optional[List[ChatHistoryItem]],
    max_messages: int = 8,
    max_chars: int = 1500,
) -> str:
    if not history:
        return ""

    items = history[-max_messages:]
    lines = []
    total = 0

    for item in items:
        content = (item.content or "").strip()
        if not content:
            continue
        role = "User" if item.role == "user" else "Assistant"
        line = f"{role}: {content}"
        if total + len(line) > max_chars:
            remaining = max_chars - total
            if remaining > 0:
                lines.append(line[:remaining])
            break
        lines.append(line)
        total += len(line)

    return "\n".join(lines)


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
        history_text = build_history_text(request.history)
        history_prompt = history_text if history_text else None
        
        session_id = request.session_id or str(uuid.uuid4())
        
        if session_id not in chat_history:
            chat_history[session_id] = []
        
        result = await rag.query(
            query=request.message,
            top_k=request.top_k,
            use_graph=request.use_graph,
            retrieval_mode=request.retrieval_mode,
            max_hops=request.max_hops,
            include_sources=request.include_sources,
            history=history_prompt,
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
            "retrieval_mode": result.get("retrieval_mode"),
            "timestamp": datetime.now().isoformat()
        })
        
        if len(chat_history[session_id]) > 50:
            chat_history[session_id] = chat_history[session_id][-50:]
        
        return ChatResponse(
            session_id=session_id,
            message=request.message,
            answer=result["answer"],
            sources=result.get("context", [])[:3],
            timestamp=datetime.now().isoformat(),
            retrieval_mode=result.get("retrieval_mode")
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
            history_text = build_history_text(request.history)
            history_prompt = history_text if history_text else None
            
            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Searching knowledge base...'})}\n\n"
            
            context = await rag.retriever_manager.retrieve(
                query=request.message,
                top_k=request.top_k,
                use_graph=request.use_graph,
                retrieval_mode=request.retrieval_mode,
                max_hops=request.max_hops,
                include_sources=request.include_sources,
            )
            
            yield f"data: {json.dumps({'type': 'sources', 'sources': context[:3]})}\n\n"
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating answer...'})}\n\n"
            
            answer = await rag.llm_manager.generate(
                query=request.message,
                context=context,
                response_style=request.response_style,
                history=history_prompt,
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
        suffix = Path(file.filename).suffix.lower()
        if suffix not in {".pdf", ".md", ".markdown"}:
            raise HTTPException(status_code=400, detail="Only PDF and Markdown files are supported")
        
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        rag = get_rag()
        kg_builder = KnowledgeGraphBuilder(
            save_intermediates=True,
            llm_manager=rag.llm_manager,
            embedding_manager=rag.embedding_manager,
        )
        end_page_arg = None if suffix != ".pdf" or end_page == -1 else end_page
        result = await kg_builder.index_path(
            str(file_path),
            rag.graph_manager,
            start_page=start_page,
            end_page=end_page_arg,
            clear=clear_existing,
        )
        
        return UploadResponse(
            status=result.get("status", "success"),
            message=result.get("message") or (
                f"Processed pages starting at {start_page + 1}" if suffix == ".pdf" else "Processed Markdown document"
            ),
            file_name=file.filename,
            nodes_added=result.get("nodes_added", 0),
            edges_added=result.get("edges_added", 0),
            documents=result.get("documents", 0),
            text_units=result.get("text_units", 0),
            entities=result.get("entities", 0),
            relationships=result.get("relationships", 0),
            communities=result.get("communities", 0),
            community_reports=result.get("community_reports", 0),
            vector_indexes_created=result.get("vector_indexes_created", 0),
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
            "labels": stats.get("labels", {}),
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
