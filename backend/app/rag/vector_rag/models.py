"""Request and response models for VectorRAG."""

from typing import Optional

from pydantic import BaseModel, Field


class IndexResult(BaseModel):
    paper_id: str
    status: str
    chunk_count: int
    embedded_chunk_count: int
    embedding_model: str


class IndexStatus(BaseModel):
    paper_id: str
    status: str
    chunk_count: int
    embedding_model: str


class Citation(BaseModel):
    paper_id: str
    title: Optional[str] = None
    page: int
    chunk_id: str
    source_filename: str


class RetrievedChunk(BaseModel):
    rank: int
    score: Optional[float] = None
    chunk_id: str
    paper_id: str
    title: Optional[str] = None
    page: int
    section: Optional[str] = None
    source_filename: str
    text: str


class VectorRetrievalRequest(BaseModel):
    question: str = Field(min_length=1)
    paper_ids: list[str] = Field(min_length=1)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)


class VectorRetrievalResponse(BaseModel):
    question: str
    retrieval_method: str = "VectorRAG"
    results: list[RetrievedChunk]
    evidence_found: bool
    message: Optional[str] = None
    embedding_latency_ms: float
    retrieval_latency_ms: float
    latency_ms: float
    embedding_model: str


class VectorQueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    embedding_latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    embedding_model: str
    llm_model: str
    retrieval_method: str = "VectorRAG"
