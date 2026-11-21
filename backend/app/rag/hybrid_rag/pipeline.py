"""HybridRAG pipeline placeholder.

Will combine vector similarity search and graph traversal over Neo4j,
then generate an answer with GPT-4o-mini.
"""

from app.core.exceptions import PipelineNotImplementedError
from app.models.schemas import QueryRequest, QueryResponse, RAGMode
from app.rag.base import BaseRAGPipeline


class HybridRAGPipeline(BaseRAGPipeline):
    mode = RAGMode.HYBRID

    def query(self, request: QueryRequest) -> QueryResponse:
        raise PipelineNotImplementedError(
            "HybridRAG is not implemented. Stage 1 only provides project setup."
        )
