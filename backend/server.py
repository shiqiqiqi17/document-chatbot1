import os
import uuid
import time
from typing import Dict, List, Any
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from langserve import add_routes
from langchain_core.runnables import RunnablePassthrough, RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from .models import ChatRequest, ChatResponse, DocumentUploadResponse
from .services.document_service import document_service
from .services.llm_service import llm_service
from .services.vectorstore_service import vectorstore_service
from .chains.rag_chain import rag_chain_service

docs_folder = Path("./docs")
docs_folder.mkdir(exist_ok=True)

app = FastAPI(
    title="Document Chatbot API",
    description="RAG-based Document Chatbot using LangChain Server (LangServe) with High Concurrency Support",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_context(question: str) -> str:
    """从向量数据库获取上下文"""
    docs = vectorstore_service.similarity_search(question, k=4)
    if not docs:
        return "No relevant documents found."
    context = "\n\n".join([doc.page_content for doc in docs])
    return context


def create_chat_chain():
    """创建带有上下文检索的 RAG 链"""
    system_prompt = """You are a helpful AI assistant that answers questions based on the provided context.

When answering:
1. Only use information from the provided context
2. If the context doesn't contain enough information, say so
3. Quote relevant parts of the context when appropriate
4. Be clear and concise in your answers

Context: {context}
"""

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{question}")
    ])

    llm = llm_service.get_llm()

    chain = (
        RunnablePassthrough.assign(context=lambda x: get_context(x["question"]))
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    return chain


@app.on_event("startup")
async def startup_event():
    print(f"[{time.strftime('%H:%M:%S')}] Application startup with LangServe")
    print(f"[{time.strftime('%H:%M:%S')}] Initializing RAG chain...")
    rag_chain_service.create_chain()
    print(f"[{time.strftime('%H:%M:%S')}] RAG chain initialized successfully")
    print(f"[{time.strftime('%H:%M:%S')}] High concurrency configuration loaded")


@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_documents(files: list[UploadFile] = File(...)):
    saved_files = []
    try:
        for file in files:
            filename = file.filename or "unnamed_file"
            safe_filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
            if not safe_filename:
                safe_filename = "unnamed_file"
            file_path = docs_folder / safe_filename
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            saved_files.append(str(file_path))

        document_service.process_files(saved_files)
        rag_chain_service.create_chain()

        return DocumentUploadResponse(
            success=True,
            message=f"Successfully processed {len(saved_files)} documents",
            documents_processed=len(saved_files),
            file_names=[file.filename for file in files]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        start_time = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Chat request: {request.question[:50]}...")

        result = rag_chain_service.invoke({"question": request.question})
        answer = result if isinstance(result, str) else str(result)

        sources = rag_chain_service.get_sources(request.question, k=4)

        elapsed = time.time() - start_time
        print(f"[{time.strftime('%H:%M:%S')}] Chat completed in {elapsed:.2f}s")

        return ChatResponse(
            answer=answer,
            sources=sources,
            session_id=request.session_id or str(uuid.uuid4()),
            time=elapsed
        )
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Chat error: {str(e)}")
        import traceback
        traceback.print_exc()
        return ChatResponse(
            answer=f"处理问题时出错：{str(e)}",
            sources=[],
            session_id=request.session_id or str(uuid.uuid4()),
            time=0
        )


@app.get("/health")
async def health_check():
    vectorstore = vectorstore_service.get_vectorstore()
    chain_ready = rag_chain_service.get_chain() is not None
    return {
        "status": "healthy",
        "documents_loaded": vectorstore is not None,
        "rag_chain_ready": chain_ready,
        "service": "LangServe",
        "concurrency_support": "enabled",
        "version": "2.0.0"
    }


@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    frontend_path = Path(__file__).parent.parent / "frontend.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    return {"message": "Document Chatbot API - Powered by LangServe", "version": "2.0.0"}


@app.get("/frontend.html")
async def frontend():
    from fastapi.responses import FileResponse
    frontend_path = Path(__file__).parent.parent / "frontend.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    return {"message": "Frontend not found"}


@app.get("/stats")
async def get_stats():
    """获取系统统计信息"""
    vectorstore = vectorstore_service.get_vectorstore()
    if vectorstore:
        try:
            doc_count = len(vectorstore.get()["ids"])
        except:
            doc_count = 0
    else:
        doc_count = 0
    
    return {
        "documents_in_vectorstore": doc_count,
        "service_status": "running",
        "service_type": "LangServe",
        "api_version": "2.0.0"
    }


# LangServe 高并发优化配置
add_routes(
    app,
    rag_chain_service,
    path="/rag",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
    playground_type="chat",
    # 高并发配置
    batch_size=4,
    max_batch_size=8,
    timeout=60,
    max_concurrent_requests=32,
)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=4,
        timeout_keep_alive=120,
        limit_concurrency=100,
    )