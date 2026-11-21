"""VectorRAG pipeline placeholder.

Will use LlamaIndex semantic chunking, OpenAI embeddings
(text-embedding-ada-002), and Neo4j as the vector store.
"""

from app.core.exceptions import PipelineNotImplementedError
from app.models.schemas import QueryRequest, QueryResponse, RAGMode
from app.rag.base import BaseRAGPipeline


class VectorRAGPipeline(BaseRAGPipeline):
    mode = RAGMode.VECTOR

    def query(self, request: QueryRequest) -> QueryResponse:
        raise PipelineNotImplementedError(
            "VectorRAG is not implemented. Stage 1 only provides project setup."
        )
