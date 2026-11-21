"""GraphRAG pipeline placeholder.

Will use LlamaIndex property-graph construction over Neo4j and GPT-4o-mini
for entity/relation extraction and graph-aware generation.
"""

from app.core.exceptions import PipelineNotImplementedError
from app.models.schemas import QueryRequest, QueryResponse, RAGMode
from app.rag.base import BaseRAGPipeline


class GraphRAGPipeline(BaseRAGPipeline):
    mode = RAGMode.GRAPH

    def query(self, request: QueryRequest) -> QueryResponse:
        raise PipelineNotImplementedError(
            "GraphRAG is not implemented. Stage 1 only provides project setup."
        )
