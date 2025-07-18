from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import time
import os
from typing import List
import uvicorn

from api.models import QueryRequest, QueryResponse, HealthResponse
from src.data_pipeline import DataPipeline
from src.embeddings import EmbeddingsHandler
from src.llm_handler import LLMHandler
from src.retriever import RAGRetriever
from src.utils import load_config, setup_logging, validate_environment

# Initialize FastAPI app
app = FastAPI(
    title="JMP Wash RAG API",
    description="Multilingual RAG system for JMP Wash documents",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
config = None
retriever = None

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global config, retriever
    
    setup_logging()
    validate_environment()
    
    # Load configuration
    config = load_config()
    
    # Initialize components
    embeddings_handler = EmbeddingsHandler(
        model_name=config["embeddings"]["model_name"],
        db_path=config["embeddings"]["db_path"]
    )
    
    llm_handler = LLMHandler(
        model_name=config["llm"]["model_name"]
    )
    
    retriever = RAGRetriever(embeddings_handler, llm_handler)
    
    print("API initialized successfully")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="healthy", message="API is running")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents and get answer"""
    if not retriever:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    start_time = time.time()
    
    try:
        response = retriever.retrieve_and_answer(
            question=request.question,
            k=request.k,
            language_filter=request.language_filter
        )
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            question=response['question'],
            answer=response['answer'],
            citations=response['citations'],
            language=response['language'],
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document"""
    if not file.filename.endswith(('.pdf', '.html', '.htm')):
        raise HTTPException(status_code=400, detail="Only PDF and HTML files are supported")
    
    try:
        # Save uploaded file
        file_path = f"data/raw/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process document
        pipeline = DataPipeline(config)
        chunks = pipeline.process_documents("data/raw", "data/processed")
        
        # Add to vector database
        if retriever:
            retriever.embeddings.upsert_chunks(chunks)
        
        return {"message": f"Document {file.filename} processed successfully", 
                "chunks_added": len(chunks)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search_documents(query: str, k: int = 10):
    """Search documents without generating answer"""
    if not retriever:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        results = retriever.get_similar_documents(query, k=k)
        return {"query": query, "results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
