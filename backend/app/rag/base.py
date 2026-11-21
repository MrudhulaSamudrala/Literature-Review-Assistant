"""Shared interface for VectorRAG, GraphRAG, and HybridRAG pipelines."""

from abc import ABC, abstractmethod

from app.models.schemas import QueryRequest, QueryResponse, RAGMode


class BaseRAGPipeline(ABC):
    mode: RAGMode

    @abstractmethod
    def query(self, request: QueryRequest) -> QueryResponse:
        """Run retrieval and generation for a user question."""
