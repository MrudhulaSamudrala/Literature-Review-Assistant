"""Entity and relationship extraction for GraphRAG.

OpenAI uses LlamaIndex SimpleLLMPathExtractor with gpt-4o-mini.
Local development uses a deterministic extractor. That extractor is not
equivalent to gpt-4o-mini.
"""

import re
from typing import Any, Sequence

from llama_index.core.graph_stores.types import (
    EntityNode,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    Relation,
)
from llama_index.core.schema import BaseNode, TransformComponent

from app.config import Settings
from app.core.exceptions import GraphRAGError
from app.rag.vector_rag.indexer import REQUIRED_LLM_MODEL, _evaluation_run, require_openai_llm

MAX_PATHS_PER_CHUNK = 10
DEVELOPMENT_GRAPH_EXTRACTOR = "development-fallback:graph-extract"

_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "because",
    "been",
    "between",
    "from",
    "have",
    "into",
    "more",
    "only",
    "than",
    "that",
    "their",
    "them",
    "then",
    "these",
    "this",
    "those",
    "through",
    "using",
    "with",
    "would",
}


def canonical_entity(value: str) -> str:
    """Match the capitalization used by LlamaIndex LLMSynonymRetriever."""

    collapsed = " ".join(value.split())
    return collapsed.capitalize()


def source_properties(metadata: dict[str, Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for key in (
        "paper_id",
        "chunk_id",
        "title",
        "page",
        "section",
        "source_filename",
        "original_chunk_text",
    ):
        value = metadata.get(key)
        if value is None or value == "":
            continue
        properties[key] = value
    return properties


def extract_terms(text: str, limit: int = 6) -> list[str]:
    terms: list[str] = []
    for word in re.findall(r"\b[A-Za-z][A-Za-z0-9-]{3,}\b", text):
        if word.lower() in _STOPWORDS:
            continue
        name = canonical_entity(word)
        if name not in terms:
            terms.append(name)
        if len(terms) == limit:
            break
    return terms


def deterministic_triples(text: str, metadata: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Build at most 10 traceable triples from one chunk."""

    paper_name = canonical_entity(str(metadata.get("paper_id") or "Paper"))
    title = metadata.get("title")
    triples: list[tuple[str, str, str]] = []
    if isinstance(title, str) and title.strip():
        triples.append((canonical_entity(title), "DISCUSSED_IN", paper_name))
    section = metadata.get("section")
    if isinstance(section, str) and section.strip():
        triples.append((canonical_entity(section), "SECTION_OF", paper_name))
    terms = extract_terms(text)
    for term in terms:
        triples.append((term, "MENTIONED_IN", paper_name))
    for left, right in zip(terms, terms[1:]):
        triples.append((left, "RELATED_TO", right))
    unique: list[tuple[str, str, str]] = []
    for triple in triples:
        if triple not in unique:
            unique.append(triple)
        if len(unique) == MAX_PATHS_PER_CHUNK:
            break
    return unique


class DeterministicPathExtractor(TransformComponent):
    """Development graph extractor. Not equivalent to gpt-4o-mini."""

    max_paths_per_chunk: int = MAX_PATHS_PER_CHUNK

    def __call__(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> Sequence[BaseNode]:
        for node in nodes:
            metadata = source_properties(node.metadata)
            metadata["extraction"] = DEVELOPMENT_GRAPH_EXTRACTOR
            triples = deterministic_triples(node.get_content(), node.metadata)[
                : self.max_paths_per_chunk
            ]
            entities: list[EntityNode] = []
            relations: list[Relation] = []
            for subject, label, target in triples:
                subject_node = EntityNode(name=subject, label="Entity", properties=dict(metadata))
                target_node = EntityNode(name=target, label="Entity", properties=dict(metadata))
                relations.append(
                    Relation(
                        label=label,
                        source_id=subject_node.id,
                        target_id=target_node.id,
                        properties=dict(metadata),
                    )
                )
                entities.extend([subject_node, target_node])
            node.metadata[KG_NODES_KEY] = entities
            node.metadata[KG_RELATIONS_KEY] = relations
        return nodes

    async def acall(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> Sequence[BaseNode]:
        return self.__call__(nodes, show_progress=show_progress, **kwargs)


def build_graph_extractor(settings: Settings) -> tuple[TransformComponent, str, str]:
    """Return the extractor, provider, and model name that will actually run."""

    from app.rag.vector_rag.service import normalize_llm_provider

    provider = normalize_llm_provider(settings)
    if provider == "openai":
        require_openai_llm(settings)
        if settings.openai_llm_model != REQUIRED_LLM_MODEL:
            raise GraphRAGError(
                "GraphRAG requires OPENAI_LLM_MODEL=gpt-4o-mini.",
                status_code=400,
            )
        from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
        from llama_index.llms.openai import OpenAI

        extractor = SimpleLLMPathExtractor(
            llm=OpenAI(
                model=settings.openai_llm_model,
                api_key=settings.openai_api_key,
                temperature=0,
            ),
            max_paths_per_chunk=MAX_PATHS_PER_CHUNK,
        )
        return extractor, provider, settings.openai_llm_model

    if _evaluation_run(settings):
        raise GraphRAGError(
            "Evaluation requires LLM_PROVIDER=openai and "
            f"OPENAI_LLM_MODEL={REQUIRED_LLM_MODEL}. "
            "The development graph extractor is not used for evaluation.",
            status_code=400,
        )
    return DeterministicPathExtractor(), provider, DEVELOPMENT_GRAPH_EXTRACTOR
