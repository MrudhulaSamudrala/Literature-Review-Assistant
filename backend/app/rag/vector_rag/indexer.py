"""Build and update the Neo4j vector index from persisted semantic chunks."""

import logging
import re

from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import TextNode
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore

try:
    from llama_index.embeddings.openai import OpenAIEmbedding
except ImportError:
    OpenAIEmbedding = None  # type: ignore

from app.config import Settings
from app.core.exceptions import IngestionError, VectorRAGError
from app.ingestion.models import ChunkRecord
from app.ingestion.persistence import load_chunks

logger = logging.getLogger(__name__)

REQUIRED_EMBEDDING_MODEL = "text-embedding-ada-002"
REQUIRED_LLM_MODEL = "gpt-4o-mini"
DEFAULT_LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_PLACEHOLDER_KEYS = {"", "your_openai_api_key_here"}
_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_PROVIDERS = {"local", "openai"}


def normalize_embedding_provider(settings: Settings) -> str:
    provider = settings.embedding_provider.strip().lower()
    if provider not in _PROVIDERS:
        raise VectorRAGError(
            "EMBEDDING_PROVIDER must be local or openai.",
            status_code=400,
        )
    return provider


def configured_embedding_model_name(settings: Settings) -> str:
    """Return the selected model name without replacing a configured OpenAI model."""

    if normalize_embedding_provider(settings) == "local":
        return settings.local_embedding_model.strip()
    return settings.openai_embedding_model.strip()


def _evaluation_run(settings: Settings) -> bool:
    return settings.evaluation_mode or settings.app_env.strip().lower() == "evaluation"


def require_openai_embeddings(settings: Settings) -> None:
    """OpenAI VectorRAG uses text-embedding-ada-002 and never the ingestion fallback."""

    if settings.openai_api_key.strip() in _PLACEHOLDER_KEYS:
        raise VectorRAGError(
            "VectorRAG requires OPENAI_API_KEY. The development embedding fallback is not used.",
            status_code=400,
        )
    if settings.openai_embedding_model != REQUIRED_EMBEDDING_MODEL:
        raise VectorRAGError(
            "VectorRAG requires OPENAI_EMBEDDING_MODEL=text-embedding-ada-002.",
            status_code=400,
        )


def require_vector_embeddings(settings: Settings) -> None:
    """Validate the selected embedding provider.

    local is development-only and does not call OpenAI embeddings.
    openai keeps the reference-paper model.
    """

    provider = normalize_embedding_provider(settings)
    if provider == "openai":
        require_openai_embeddings(settings)
        return
    if _evaluation_run(settings):
        raise VectorRAGError(
            "Evaluation requires EMBEDDING_PROVIDER=openai and "
            f"OPENAI_EMBEDDING_MODEL={REQUIRED_EMBEDDING_MODEL}. "
            "The local embedding model is for development only.",
            status_code=400,
        )
    if not settings.local_embedding_model.strip():
        raise VectorRAGError(
            "LOCAL_EMBEDDING_MODEL is required when EMBEDDING_PROVIDER=local.",
            status_code=400,
        )


def require_openai_llm(settings: Settings) -> None:
    if settings.openai_api_key.strip() in _PLACEHOLDER_KEYS:
        raise VectorRAGError(
            "VectorRAG answer generation requires OPENAI_API_KEY.",
            status_code=400,
        )
    if settings.openai_llm_model != REQUIRED_LLM_MODEL:
        raise VectorRAGError(
            "VectorRAG requires OPENAI_LLM_MODEL=gpt-4o-mini.",
            status_code=400,
        )


def log_embedding_configuration(settings: Settings) -> None:
    """Log the selected provider and model. Never log credentials."""

    try:
        provider = normalize_embedding_provider(settings)
        model_name = configured_embedding_model_name(settings)
    except VectorRAGError as exc:
        logger.error("embedding configuration invalid: %s", exc)
        return
    logger.info("embedding provider = %s", provider)
    logger.info("embedding model = %s", model_name)


def vector_index_target(settings: Settings) -> tuple[str, str]:
    """Keep the reference index separate from the local development index."""

    if normalize_embedding_provider(settings) == "local":
        return settings.local_vector_index_name, settings.local_vector_node_label
    return settings.vector_index_name, settings.vector_node_label


def _reported_dimension(embed_model: BaseEmbedding) -> int | None:
    value = getattr(embed_model, "embed_dim", None)
    if isinstance(value, int) and value > 0:
        return value
    sentence_model = getattr(embed_model, "_model", None)
    for method_name in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        getter = getattr(sentence_model, method_name, None)
        if not callable(getter):
            continue
        dimension = getter()
        if isinstance(dimension, int) and dimension > 0:
            return dimension
    return None


def embedding_dimension(settings: Settings, embed_model: BaseEmbedding) -> int:
    """Use the model dimension. Local models never inherit the OpenAI index size."""

    reported = _reported_dimension(embed_model)
    if normalize_embedding_provider(settings) == "local":
        if reported is None:
            raise VectorRAGError(
                "The local embedding model did not report a vector dimension. "
                f"Refusing to use vector dimension {settings.vector_embedding_dimension}.",
                status_code=500,
            )
        return reported
    if reported is not None:
        return reported
    if settings.vector_embedding_dimension < 1:
        raise VectorRAGError("VECTOR_EMBEDDING_DIMENSION must be positive.", status_code=500)
    return settings.vector_embedding_dimension


def _build_local_embedding(model_name: str) -> BaseEmbedding:
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        return HuggingFaceEmbedding(model_name=model_name)
    except Exception as exc:
        logger.exception("Failed to load local embedding model %s", model_name)
        raise VectorRAGError(
            f"Failed to load local embedding model {model_name}.",
            status_code=500,
        ) from exc


def require_neo4j(settings: Settings) -> None:
    if not settings.neo4j_uri.strip() or not settings.neo4j_username.strip():
        raise VectorRAGError("Neo4j URI and username are required for VectorRAG.", status_code=400)
    if not settings.neo4j_password:
        raise VectorRAGError("NEO4J_PASSWORD is required for VectorRAG.", status_code=400)
    index_name, node_label = vector_index_target(settings)
    if not _LABEL.match(node_label) or not _LABEL.match(index_name):
        raise VectorRAGError("The Neo4j vector index name is not valid.", status_code=500)


def build_embedding_model(settings: Settings) -> BaseEmbedding:
    require_vector_embeddings(settings)
    provider = normalize_embedding_provider(settings)
    model_name = configured_embedding_model_name(settings)
    logger.info("embedding provider = %s", provider)
    logger.info("embedding model = %s", model_name)
    if provider == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding

        return OpenAIEmbedding(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )
    return _build_local_embedding(model_name)


def connect_vector_store(settings: Settings, embedding_dimension: int) -> Neo4jVectorStore:
    require_neo4j(settings)
    index_name, node_label = vector_index_target(settings)
    model_name = configured_embedding_model_name(settings)
    logger.info(
        "Neo4j vector index target name=%s label=%s dimension=%s",
        index_name,
        node_label,
        embedding_dimension,
    )
    try:
        store = Neo4jVectorStore(
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            url=settings.neo4j_uri,
            embedding_dimension=embedding_dimension,
            database=settings.neo4j_database,
            index_name=index_name,
            node_label=node_label,
            hybrid_search=False,
        )
    except ValueError as exc:
        logger.exception("Neo4j vector store connection failed")
        raise VectorRAGError("Could not connect to Neo4j.", status_code=503) from exc
    if store.embedding_dimension != embedding_dimension:
        raise VectorRAGError(
            f"The existing Neo4j vector index '{store.index_name}' on :{store.node_label} "
            f"has dimension {store.embedding_dimension}, which does not match embedding model "
            f"{model_name} (dimension {embedding_dimension}). "
            "Refusing to reuse an incompatible index. "
            "EMBEDDING_PROVIDER=local uses the separate development index "
            f"{settings.local_vector_index_name} on :{settings.local_vector_node_label}.",
            status_code=409,
        )
    return store


def chunks_to_nodes(chunks: list[ChunkRecord]) -> list[TextNode]:
    """Use the persisted chunk id as the Neo4j node id so re-indexing updates in place."""

    nodes: list[TextNode] = []
    for chunk in chunks:
        nodes.append(
            TextNode(
                id_=chunk.chunk_id,
                text=chunk.text,
                ref_doc_id=chunk.paper_id,
                metadata={
                    "paper_id": chunk.paper_id,
                    "chunk_id": chunk.chunk_id,
                    "title": chunk.title or "",
                    "section": chunk.section or "",
                    "page": chunk.page,
                    "source_filename": chunk.source_filename,
                },
            )
        )
    return nodes


def chunks_needing_embedding(
    existing_text_by_id: dict[str, str],
    chunks: list[ChunkRecord],
) -> list[ChunkRecord]:
    """Skip chunks already stored with the same text."""

    return [chunk for chunk in chunks if existing_text_by_id.get(chunk.chunk_id) != chunk.text]


def load_paper_chunks(settings: Settings, paper_id: str) -> list[ChunkRecord]:
    try:
        chunks = load_chunks(settings, paper_id)
    except IngestionError as exc:
        raise VectorRAGError(str(exc), status_code=exc.status_code) from exc
    if not chunks:
        raise VectorRAGError(f"Paper {paper_id} has no semantic chunks to index.", status_code=422)
    return chunks


def read_indexed_chunks(store: Neo4jVectorStore, settings: Settings, paper_id: str) -> dict[str, str]:
    label = vector_index_target(settings)[1]
    rows = store.database_query(
        f"MATCH (c:`{label}`) WHERE c.paper_id = $paper_id "
        "RETURN c.id AS id, c.text AS text",
        params={"paper_id": paper_id},
    )
    return {row["id"]: row.get("text") or "" for row in rows if row.get("id")}


def delete_chunk_ids(store: Neo4jVectorStore, settings: Settings, chunk_ids: list[str]) -> None:
    if not chunk_ids:
        return
    label = vector_index_target(settings)[1]
    store.database_query(
        f"MATCH (c:`{label}`) WHERE c.id IN $ids DETACH DELETE c",
        params={"ids": chunk_ids},
    )


def embed_and_store(store: Neo4jVectorStore, embed_model: BaseEmbedding, nodes: list[TextNode]) -> None:
    """Embed with the configured model and write the nodes through VectorStoreIndex."""

    if not nodes:
        return
    index = VectorStoreIndex.from_vector_store(store, embed_model=embed_model)
    index.insert_nodes(nodes)


def count_indexed_chunks(store: Neo4jVectorStore, settings: Settings, paper_id: str) -> int:
    label = vector_index_target(settings)[1]
    rows = store.database_query(
        f"MATCH (c:`{label}`) WHERE c.paper_id = $paper_id RETURN count(c) AS chunk_count",
        params={"paper_id": paper_id},
    )
    if not rows:
        return 0
    return int(rows[0].get("chunk_count") or 0)
