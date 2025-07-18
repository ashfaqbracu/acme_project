from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class QueryRequest(BaseModel):
    question: str
    k: int = 4
    language_filter: Optional[str] = None

class Citation(BaseModel):
    id: str
    text: str
    source: str
    language: str
    relevance_score: Optional[float] = None

class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation]
    language: str
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    message: str
