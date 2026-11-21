"""LlamaIndex semantic chunking for extracted paper pages.

Breakpoint detection follows the reference paper:

    buffer_size = 1
    breakpoint_percentile_threshold = 95
    embedding model = text-embedding-ada-002

SemanticSplitterNodeParser scores sentence groups with an embedding model.
Those vectors are used only to locate split points. They are not written to
disk and they are not indexed in Neo4j.

If OPENAI_API_KEY is missing, a development-only deterministic adapter is used.
That adapter is not equivalent to the reference-paper setup and is refused
when evaluation mode is enabled.
"""

import hashlib
import logging
from typing import Any

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser.text.semantic_splitter import SemanticSplitterNodeParser
from llama_index.core.schema import BaseNode, Document, MetadataMode

from app.config import Settings
from app.core.exceptions import IngestionError
from app.ingestion.document_processor import infer_section
from app.ingestion.models import BreakpointEmbeddingStatus, ChunkRecord, PaperMetadata

logger = logging.getLogger(__name__)

BUFFER_SIZE = 1
BREAKPOINT_PERCENTILE_THRESHOLD = 95
REFERENCE_EMBEDDING_MODEL = "text-embedding-ada-002"
DEVELOPMENT_FALLBACK_MODEL = "development-fallback:deterministic"


class DeterministicTextEmbedding(BaseEmbedding):
    """Development fallback. Not equivalent to text-embedding-ada-002.

    Vectors stay in memory so SemanticSplitterNodeParser can score sentence
    groups when no OpenAI API key is configured. Do not use this for evaluation.
    """

    embed_dim: int = 32

    def __init__(self, embed_dim: int = 32, **kwargs: Any) -> None:
        super().__init__(
            embed_dim=embed_dim,
            model_name=DEVELOPMENT_FALLBACK_MODEL,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "DeterministicTextEmbedding"

    def _vector(self, text: str) -> list[float]:
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        values: list[float] = []
        while len(values) < self.embed_dim:
            seed = hashlib.sha256(seed).digest()
            for byte in seed:
                values.append((byte / 127.5) - 1.0)
                if len(values) == self.embed_dim:
                    break
        return values

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._vector(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._vector(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._vector(text)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._vector(text)


class SemanticChunker:
    """Split page documents with LlamaIndex SemanticSplitterNodeParser."""

    def __init__(
        self,
        embed_model: BaseEmbedding | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.embed_model = embed_model or build_breakpoint_embed_model(settings)
        self.embedding_status = describe_breakpoint_embedding(self.embed_model, settings)
        _reject_fallback_during_evaluation(settings, self.embedding_status)
        _log_breakpoint_embedding(self.embedding_status)
        self.embed_model_name = self.embedding_status.model
        self.parser = SemanticSplitterNodeParser.from_defaults(
            embed_model=self.embed_model,
            buffer_size=BUFFER_SIZE,
            breakpoint_percentile_threshold=BREAKPOINT_PERCENTILE_THRESHOLD,
        )

    def chunk_documents(
        self,
        documents: list[Document],
        paper: PaperMetadata,
    ) -> list[ChunkRecord]:
        logger.info("Semantic chunking started paper_id=%s", paper.paper_id)
        if not documents:
            raise IngestionError("The PDF contains no extractable text.", status_code=422)
        try:
            nodes = self.parser.get_nodes_from_documents(documents)
        except Exception as exc:
            logger.exception("Semantic chunking failed paper_id=%s", paper.paper_id)
            raise IngestionError("Semantic chunking failed.", status_code=500) from exc

        chunks = _records_from_nodes(nodes, paper)
        if not chunks:
            raise IngestionError("Semantic chunking produced no chunks.", status_code=422)
        logger.info(
            "Semantic chunking completed paper_id=%s chunks=%s",
            paper.paper_id,
            len(chunks),
        )
        return chunks


def _build_local_embedding(model_name: str) -> BaseEmbedding:
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        return HuggingFaceEmbedding(model_name=model_name)
    except Exception as exc:
        logger.exception("Failed to load local embedding model %s", model_name)
        raise IngestionError(
            f"Failed to load local embedding model {model_name}.",
            status_code=500,
        ) from exc


def build_breakpoint_embed_model(settings: Settings | None) -> BaseEmbedding:
    """Use the configured embedding model based on settings.

    In local mode, uses HuggingFaceEmbedding with sentence-transformers/all-MiniLM-L6-v2.
    In openai mode, uses OpenAIEmbedding when a valid API key is present.
    The deterministic development fallback is used only when unconfigured.
    Its vectors are not persisted.
    """

    if settings is None:
        return DeterministicTextEmbedding()

    provider = settings.embedding_provider.strip().lower()
    if provider == "local":
        model_name = (
            settings.local_embedding_model.strip()
            if settings.local_embedding_model.strip()
            else "sentence-transformers/all-MiniLM-L6-v2"
        )
        return _build_local_embedding(model_name)

    api_key = settings.openai_api_key.strip()
    model_name = (
        settings.openai_embedding_model.strip()
        if settings.openai_embedding_model.strip()
        else REFERENCE_EMBEDDING_MODEL
    )
    if api_key and api_key not in {"your_openai_api_key_here"}:
        from llama_index.embeddings.openai import OpenAIEmbedding

        return OpenAIEmbedding(model=model_name, api_key=api_key)
    return DeterministicTextEmbedding()


def describe_breakpoint_embedding(
    embed_model: BaseEmbedding,
    settings: Settings | None = None,
) -> BreakpointEmbeddingStatus:
    """Identify whether chunk boundaries come from local embeddings, OpenAI, or the development fallback."""

    if isinstance(embed_model, DeterministicTextEmbedding):
        return BreakpointEmbeddingStatus(
            mode="development_fallback",
            model=DEVELOPMENT_FALLBACK_MODEL,
            development_fallback=True,
            reference_equivalent=False,
            vectors_persisted=False,
        )

    model_name = getattr(embed_model, "model_name", None) or (
        settings.embedding_model if settings is not None else REFERENCE_EMBEDDING_MODEL
    )
    provider = settings.embedding_provider.strip().lower() if settings is not None else ""
    is_local = (
        provider == "local"
        or type(embed_model).__name__ == "HuggingFaceEmbedding"
        or "MiniLM" in str(model_name)
    )
    if is_local:
        return BreakpointEmbeddingStatus(
            mode="local",
            model=model_name,
            development_fallback=False,
            reference_equivalent=False,
            vectors_persisted=False,
        )

    return BreakpointEmbeddingStatus(
        mode="openai",
        model=model_name,
        development_fallback=False,
        reference_equivalent=model_name == REFERENCE_EMBEDDING_MODEL,
        vectors_persisted=False,
    )


def is_evaluation_run(settings: Settings | None) -> bool:
    if settings is None:
        return False
    return settings.evaluation_mode or settings.app_env.strip().lower() == "evaluation"


def _reject_fallback_during_evaluation(
    settings: Settings | None,
    status: BreakpointEmbeddingStatus,
) -> None:
    if not is_evaluation_run(settings):
        return
    if status.development_fallback:
        logger.error(
            "Evaluation run refused the development fallback embedding adapter. "
            "Semantic chunk boundaries would not match the reference-paper setup. "
            "Set EMBEDDING_PROVIDER=openai, OPENAI_API_KEY, and OPENAI_EMBEDDING_MODEL=%s.",
            REFERENCE_EMBEDDING_MODEL,
        )
        raise IngestionError(
            "Evaluation runs require OPENAI_API_KEY so semantic chunking can use "
            f"{REFERENCE_EMBEDDING_MODEL}. The development fallback embedding "
            "adapter is not used for evaluation.",
            status_code=400,
        )
    if status.mode == "local":
        logger.error(
            "Evaluation run refused local embedding model %s. "
            "Evaluation requires EMBEDDING_PROVIDER=openai and OPENAI_EMBEDDING_MODEL=%s.",
            status.model,
            REFERENCE_EMBEDDING_MODEL,
        )
        raise IngestionError(
            "Evaluation runs require EMBEDDING_PROVIDER=openai and OPENAI_API_KEY so semantic chunking can use "
            f"{REFERENCE_EMBEDDING_MODEL}. The local embedding model is for development only.",
            status_code=400,
        )


def _log_breakpoint_embedding(status: BreakpointEmbeddingStatus) -> None:
    if status.development_fallback:
        logger.warning(
            "DEVELOPMENT FALLBACK: semantic chunking is using %s. "
            "Chunk boundaries are not equivalent to the reference-paper setup "
            "(%s, buffer_size=%s, breakpoint_percentile_threshold=%s). "
            "Embedding vectors are not stored.",
            status.model,
            REFERENCE_EMBEDDING_MODEL,
            BUFFER_SIZE,
            BREAKPOINT_PERCENTILE_THRESHOLD,
        )
        return
    if status.mode == "local":
        logger.info(
            "Semantic chunking is using local embedding model %s. "
            "Reference-paper equivalent=%s. Embedding vectors are not stored.",
            status.model,
            status.reference_equivalent,
        )
        return
    logger.info(
        "Semantic chunking is using OpenAI embedding model %s. "
        "Reference-paper equivalent=%s. Embedding vectors are not stored.",
        status.model,
        status.reference_equivalent,
    )


def _records_from_nodes(nodes: list[BaseNode], paper: PaperMetadata) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    for node in nodes:
        text = node.get_content(metadata_mode=MetadataMode.NONE).strip()
        if not text:
            continue
        page = node.metadata.get("page")
        if not isinstance(page, int):
            raise IngestionError("A semantic chunk is missing its page number.", status_code=500)
        source_filename = node.metadata.get("source_filename") or paper.filename
        section = infer_section(text) or _optional_str(node.metadata.get("section"))
        records.append(
            ChunkRecord(
                chunk_id=f"{paper.paper_id}_C{len(records) + 1:03d}",
                paper_id=paper.paper_id,
                title=paper.title,
                section=section,
                page=page,
                source_filename=str(source_filename),
                text=text,
            )
        )
    return records


def _optional_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
