"""Retrieve top-k chunks with LlamaIndex VectorIndexRetriever."""

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import QueryBundle
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever

from app.rag.vector_rag.models import Citation, RetrievedChunk


def build_paper_filter(paper_ids: list[str]) -> MetadataFilters:
    return MetadataFilters(
        filters=[
            MetadataFilter(
                key="paper_id",
                operator=FilterOperator.IN,
                value=paper_ids,
            )
        ]
    )


def retrieve_chunks(
    store: Neo4jVectorStore,
    embed_model: BaseEmbedding,
    question: str,
    query_embedding: list[float],
    paper_ids: list[str],
    top_k: int,
) -> list[RetrievedChunk]:
    """Search Neo4j with an embedding that was already computed."""

    index = VectorStoreIndex.from_vector_store(store, embed_model=embed_model)
    retriever = VectorIndexRetriever(
        index,
        similarity_top_k=top_k,
        filters=build_paper_filter(paper_ids),
        embed_model=embed_model,
    )
    nodes = retriever.retrieve(QueryBundle(query_str=question, embedding=query_embedding))
    results: list[RetrievedChunk] = []
    for rank, item in enumerate(nodes, start=1):
        metadata = item.node.metadata or {}
        chunk_id = str(metadata.get("chunk_id") or item.node.node_id)
        paper_id = str(metadata.get("paper_id") or "")
        page = metadata.get("page")
        if not paper_id or not isinstance(page, int):
            continue
        results.append(
            RetrievedChunk(
                rank=rank,
                score=item.score,
                chunk_id=chunk_id,
                paper_id=paper_id,
                title=_optional_text(metadata.get("title")),
                page=page,
                section=_optional_text(metadata.get("section")),
                source_filename=str(metadata.get("source_filename") or ""),
                text=item.node.get_content(),
            )
        )
    return results


def citations_from(chunks: list[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(
            paper_id=chunk.paper_id,
            title=chunk.title,
            page=chunk.page,
            chunk_id=chunk.chunk_id,
            source_filename=chunk.source_filename,
        )
        for chunk in chunks
    ]


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None
