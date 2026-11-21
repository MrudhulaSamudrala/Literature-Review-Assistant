"""Request and response models for GraphRAG."""

from typing import Optional

from pydantic import BaseModel, Field

from app.rag.vector_rag.models import Citation, RetrievedChunk


class GraphIndexResult(BaseModel):
    paper_id: str
    status: str
    chunk_count: int
    entity_count: int
    relationship_count: int
    llm_provider: str
    llm_model: str


class GraphEntity(BaseModel):
    name: str
    paper_id: str
    chunk_id: str


class GraphRelationship(BaseModel):
    source: str
    relation: str
    target: str
    paper_id: str
    chunk_id: str


class GraphPath(BaseModel):
    path: str
    source: str
    relation: str
    target: str
    paper_id: str
    chunk_id: str


class GraphRetrievalRequest(BaseModel):
    question: str = Field(min_length=1)
    paper_ids: list[str] = Field(min_length=1)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    path_depth: int = Field(default=2, ge=1, le=4)


class GraphRetrievalResponse(BaseModel):
    question: str
    retrieval_method: str = "GraphRAG"
    entities: list[GraphEntity]
    relationships: list[GraphRelationship]
    paths: list[GraphPath]
    results: list[RetrievedChunk]
    evidence_found: bool
    latency_ms: float


class GraphQueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[Citation]
    entities: list[GraphEntity]
    relationships: list[GraphRelationship]
    retrieved_chunks: list[RetrievedChunk]
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    llm_model: str
    retrieval_method: str = "GraphRAG"
