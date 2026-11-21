"""Routes queries to the selected RAG pipeline."""

from functools import lru_cache

from app.models.schemas import QueryRequest, QueryResponse, RAGMode
from app.rag.base import BaseRAGPipeline
from app.rag.graph_rag import GraphRAGPipeline
from app.rag.hybrid_rag import HybridRAGPipeline
from app.rag.vector_rag import VectorRAGPipeline


class RAGService:
    def __init__(self) -> None:
        self._pipelines: dict[RAGMode, BaseRAGPipeline] = {
            RAGMode.VECTOR: VectorRAGPipeline(),
            RAGMode.GRAPH: GraphRAGPipeline(),
            RAGMode.HYBRID: HybridRAGPipeline(),
        }

    def query(self, request: QueryRequest) -> QueryResponse:
        pipeline = self._pipelines[request.mode]
        return pipeline.query(request)


@lru_cache
def get_rag_service() -> RAGService:
    return RAGService()
