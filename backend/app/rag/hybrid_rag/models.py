"""Request and response models for HybridRAG."""

from typing import Optional

from pydantic import BaseModel, Field

from app.rag.graph_rag.models import GraphEntity, GraphRelationship
from app.rag.vector_rag.models import Citation


class HybridQueryRequest(BaseModel):
    question: str = Field(min_length=1)
    paper_ids: list[str] = Field(min_length=1)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    path_depth: int = Field(default=2, ge=1, le=4)


class HybridEvidence(BaseModel):
    paper_id: str
    title: Optional[str] = None
    page: int
    section: Optional[str] = None
    chunk_id: str
    source_filename: str
    text: str
    score: Optional[float] = None
    origin: str


class HybridQueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[Citation]
    vector_evidence: list[HybridEvidence]
    graph_evidence: list[HybridEvidence]
    combined_evidence: list[HybridEvidence]
    entities: list[GraphEntity]
    relationships: list[GraphRelationship]
    vector_retrieval_latency_ms: float
    graph_retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    embedding_model: str
    llm_model: str
    retrieval_method: str = "HybridRAG"
