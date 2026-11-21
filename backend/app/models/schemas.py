"""Pydantic request and response models."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RAGMode(str, Enum):
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"


class HealthResponse(BaseModel):
    status: str
    service: str


class SourceCitation(BaseModel):
    document_id: Optional[str] = None
    title: Optional[str] = None
    snippet: Optional[str] = None
    score: Optional[float] = None
    page: Optional[int] = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    mode: RAGMode = RAGMode.VECTOR
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    question: str
    mode: RAGMode
    answer: str
    sources: list[SourceCitation] = Field(default_factory=list)
    implemented: bool = False


class DocumentUploadResponse(BaseModel):
    filename: str
    stored_path: str
    message: str
    implemented: bool = False
