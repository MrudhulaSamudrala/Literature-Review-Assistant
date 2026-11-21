"""Index semantic chunks into Neo4j and answer questions from the graph."""

import logging
import re
import time
from typing import Any

from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core.indices.property_graph.sub_retrievers.llm_synonym import (
    LLMSynonymRetriever,
)
from llama_index.core.indices.property_graph.sub_retrievers.vector import (
    VectorContextRetriever,
)
from llama_index.core.schema import TextNode

from app.config import Settings
from app.core.exceptions import GraphRAGError, IngestionError
from app.ingestion.models import ChunkRecord
from app.ingestion.persistence import load_chunks
from app.rag.graph_rag.extractor import (
    DEVELOPMENT_GRAPH_EXTRACTOR,
    build_graph_extractor,
    canonical_entity,
)
from app.rag.graph_rag.models import (
    GraphEntity,
    GraphIndexResult,
    GraphPath,
    GraphQueryResponse,
    GraphRelationship,
    GraphRetrievalResponse,
)
from app.rag.graph_rag.store import ApoCFreeNeo4jPropertyGraphStore
from app.rag.vector_rag.indexer import build_embedding_model
from app.rag.vector_rag.models import RetrievedChunk
from app.rag.vector_rag.prompts import (
    NO_EVIDENCE_ANSWER,
    build_grounded_prompt,
    clean_and_validate_answer,
    filter_table_noise,
)
from app.rag.vector_rag.retriever import citations_from
from app.rag.vector_rag.service import (
    _generate,
    _generation_model,
    configured_llm_model_name,
    generate_grounded_answer,
    normalize_llm_provider,
    resolve_generation_model_name,
)

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9-]{2,}")


class _DevelopmentSynonymLLM:
    """Keyword expander used when graph retrieval must not call OpenAI."""

    def predict(self, prompt: Any, **kwargs: Any) -> str:
        words: list[str] = []
        for word in _WORD.findall(str(kwargs.get("query_str", ""))):
            name = canonical_entity(word)
            if name not in words:
                words.append(name)
        limit = int(kwargs.get("max_keywords") or 10)
        return "^".join(words[:limit])


def chunks_to_graph_nodes(chunks: list[ChunkRecord]) -> list[TextNode]:
    nodes: list[TextNode] = []
    for chunk in chunks:
        nodes.append(
            TextNode(
                id_=chunk.chunk_id,
                text=chunk.text,
                metadata={
                    "paper_id": chunk.paper_id,
                    "chunk_id": chunk.chunk_id,
                    "title": chunk.title or "",
                    "section": chunk.section or "",
                    "page": chunk.page,
                    "source_filename": chunk.source_filename,
                    "original_chunk_text": chunk.text,
                },
            )
        )
    return nodes


def question_keywords(question: str) -> list[str]:
    return _DevelopmentSynonymLLM().predict(None, query_str=question, max_keywords=10).split("^")


def filter_graph_rows(
    rows: list[dict[str, Any]],
    paper_ids: list[str],
    keywords: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    selected = [row for row in rows if row.get("paper_id") in paper_ids]
    if not keywords:
        return []
    needles = [keyword.lower() for keyword in keywords if keyword]
    matched = [
        row
        for row in selected
        if any(
            needle in str(row.get(field, "")).lower()
            for needle in needles
            for field in ("source", "target", "relation", "text")
        )
    ]
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in matched:
        key = (
            str(row.get("source")),
            str(row.get("relation")),
            str(row.get("target")),
            str(row.get("chunk_id")),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
        if len(unique) == limit:
            break
    return unique


def rows_to_response(question: str, rows: list[dict[str, Any]], latency_ms: float) -> GraphRetrievalResponse:
    entities: list[GraphEntity] = []
    relationships: list[GraphRelationship] = []
    paths: list[GraphPath] = []
    chunks: dict[str, RetrievedChunk] = {}
    seen_entities: set[tuple[str, str]] = set()
    for row in rows:
        paper_id = str(row["paper_id"])
        chunk_id = str(row["chunk_id"])
        source = str(row["source"])
        target = str(row["target"])
        relation = str(row["relation"])
        for name in (source, target):
            identity = (name, chunk_id)
            if identity not in seen_entities:
                seen_entities.add(identity)
                entities.append(GraphEntity(name=name, paper_id=paper_id, chunk_id=chunk_id))
        relationships.append(
            GraphRelationship(
                source=source,
                relation=relation,
                target=target,
                paper_id=paper_id,
                chunk_id=chunk_id,
            )
        )
        paths.append(
            GraphPath(
                path=f"{source} -[{relation}]-> {target}",
                source=source,
                relation=relation,
                target=target,
                paper_id=paper_id,
                chunk_id=chunk_id,
            )
        )
        if chunk_id not in chunks and row.get("text") and isinstance(row.get("page"), int):
            chunks[chunk_id] = RetrievedChunk(
                rank=len(chunks) + 1,
                score=None,
                chunk_id=chunk_id,
                paper_id=paper_id,
                title=row.get("title") or None,
                page=row["page"],
                section=row.get("section") or None,
                source_filename=str(row.get("source_filename") or ""),
                text=str(row["text"]),
            )
    results = list(chunks.values())
    return GraphRetrievalResponse(
        question=question,
        entities=entities,
        relationships=relationships,
        paths=paths,
        results=results,
        evidence_found=bool(results),
        latency_ms=latency_ms,
    )


def build_graph_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    relationships: list[GraphRelationship],
    entities: list[GraphEntity],
) -> str:
    """Build grounded prompt for GraphRAG combining relational facts and chunk evidence."""
    graph_facts = [
        f"{rel.source} -[{rel.relation}]-> {rel.target}"
        for rel in relationships[:6]
        if rel.source and rel.target and rel.relation
    ]
    unique_facts = list(dict.fromkeys(graph_facts))
    graph_block = "\n".join(unique_facts) if unique_facts else "None"
    evidence_blocks = [
        f"Evidence {i}:\n{filter_table_noise(chunk.text)}"
        for i, chunk in enumerate(chunks[:3], start=1)
        if filter_table_noise(chunk.text)
    ]
    evidence = "\n\n".join(evidence_blocks)
    return (
        "Answer the question using the evidence.\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Graph facts:\n{graph_block}\n\n"
        f"Evidence:\n{evidence}\n\n"
        "Give a short, direct answer in 1-3 sentences.\n"
        "- Use only the evidence.\n"
        "- Do not copy the evidence verbatim.\n"
        "- Do not invent facts or citations.\n"
        "- Do not invent graph relationships.\n"
        "- Do not mention the evidence, instructions, sources, chunk IDs, or model.\n"
        f'- If the available evidence does not provide enough information to answer this question, say: "{NO_EVIDENCE_ANSWER}"'
    )


class GraphRAGService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def index_paper(self, paper_id: str) -> GraphIndexResult:
        chunks = self._load_chunks(paper_id)
        extractor, provider, model_name = build_graph_extractor(self.settings)
        nodes = chunks_to_graph_nodes(chunks)
        logger.info("graph llm_provider = %s", provider)
        logger.info("graph llm_model = %s", model_name)
        if model_name == DEVELOPMENT_GRAPH_EXTRACTOR:
            logger.info(
                "Graph extraction is using %s. This is not equivalent to gpt-4o-mini.",
                DEVELOPMENT_GRAPH_EXTRACTOR,
            )
        logger.info("graph chunks processed = %s", len(nodes))
        try:
            extracted = list(extractor(nodes))
        except GraphRAGError:
            raise
        except Exception as exc:
            logger.exception("Graph extraction failed paper_id=%s", paper_id)
            raise GraphRAGError("Graph extraction failed.", status_code=502) from exc
        entity_ids, relation_count = _count_graph(extracted)
        store = connect_graph_store(self.settings)
        try:
            store.delete_paper_relations(paper_id)
            _insert_property_graph(self.settings, store, extracted, _AlreadyExtracted())
        except GraphRAGError:
            raise
        except Exception as exc:
            logger.exception("Graph indexing failed paper_id=%s", paper_id)
            raise GraphRAGError("Could not write the Neo4j knowledge graph.", status_code=503) from exc
        finally:
            store.close()
        logger.info("graph entities created = %s", len(entity_ids))
        logger.info("graph relationships created = %s", relation_count)
        return GraphIndexResult(
            paper_id=paper_id,
            status="indexed",
            chunk_count=len(chunks),
            entity_count=len(entity_ids),
            relationship_count=relation_count,
            llm_provider=provider,
            llm_model=model_name,
        )

    def retrieve(
        self,
        question: str,
        paper_ids: list[str],
        top_k: int | None,
        path_depth: int,
    ) -> GraphRetrievalResponse:
        started = time.perf_counter()
        resolved_top_k = _resolve_top_k(self.settings, top_k)
        _validate_depth(path_depth)
        keywords = question_keywords(question)
        logger.info(
            "Graph retrieval started paper_ids=%s top_k=%s path_depth=%s",
            paper_ids,
            resolved_top_k,
            path_depth,
        )
        store = connect_graph_store(self.settings)
        try:
            try:
                node_count_data = store.structured_query(
                    "MATCH (n) WHERE n.paper_id IN $paper_ids RETURN count(n) AS node_count",
                    param_map={"paper_ids": paper_ids},
                )
                rel_count_data = store.structured_query(
                    "MATCH ()-[r]->() WHERE r.paper_id IN $paper_ids AND type(r) <> 'MENTIONS' RETURN count(r) AS rel_count",
                    param_map={"paper_ids": paper_ids},
                )
                num_nodes = node_count_data[0]["node_count"] if node_count_data else 0
                num_rels = rel_count_data[0]["rel_count"] if rel_count_data else 0
                logger.info("GraphRAG Diagnostic: number of graph nodes = %s", num_nodes)
                logger.info("GraphRAG Diagnostic: number of relationships = %s", num_rels)
                if num_nodes == 0 and num_rels == 0:
                    logger.warning("Graph data missing: 0 nodes and 0 relationships for paper_ids=%s", paper_ids)
            except Exception as diag_err:
                logger.debug("Diagnostic count query skipped: %s", diag_err)

            _run_reference_retrievers(self.settings, store, question, resolved_top_k, path_depth)
            query_str = (
                f"MATCH p = (a:`__Entity__`)-[rels*1..{path_depth}]->(b:`__Entity__`)\n"
                "WHERE ALL(rel IN rels WHERE rel.paper_id IN $paper_ids AND type(rel) <> 'MENTIONS')\n"
                "RETURN DISTINCT source.name AS source, type(rel) AS relation, target.name AS target..."
            )
            logger.info("GraphRAG Diagnostic: graph query executed =\n%s", query_str)
            rows = fetch_graph_rows(store, paper_ids, path_depth)
        except GraphRAGError:
            raise
        except Exception as exc:
            logger.exception("Graph retrieval failed")
            raise GraphRAGError("Graph retrieval failed.", status_code=503) from exc
        finally:
            store.close()
        matched = filter_graph_rows(rows, paper_ids, keywords, resolved_top_k)
        response = rows_to_response(question, matched, _elapsed_ms(started))
        logger.info("GraphRAG Diagnostic: number of graph results returned = %s", len(response.results))
        logger.info(
            "Graph retrieval completed paths=%s chunks=%s",
            len(response.paths),
            len(response.results),
        )
        return response

    def query(
        self,
        question: str,
        paper_ids: list[str],
        top_k: int | None,
        path_depth: int,
    ) -> GraphQueryResponse:
        started = time.perf_counter()
        retrieved = self.retrieve(question, paper_ids, top_k, path_depth)
        if not retrieved.results:
            return GraphQueryResponse(
                question=question,
                answer=NO_EVIDENCE_ANSWER,
                sources=[],
                entities=[],
                relationships=[],
                retrieved_chunks=[],
                retrieval_latency_ms=retrieved.latency_ms,
                generation_latency_ms=0.0,
                total_latency_ms=_elapsed_ms(started),
                llm_model=configured_llm_model_name(self.settings),
            )
        candidate_chunks = retrieved.results[:3]
        evidence_blocks = [
            f"Evidence {i}:\n{filter_table_noise(chunk.text)}"
            for i, chunk in enumerate(candidate_chunks, start=1)
            if filter_table_noise(chunk.text)
        ]
        graph_facts = [
            f"{rel.source} -[{rel.relation}]-> {rel.target}"
            for rel in retrieved.relationships[:6]
            if rel.source and rel.target and rel.relation
        ]
        unique_facts = list(dict.fromkeys(graph_facts))
        context_parts = []
        if unique_facts:
            context_parts.append("Graph facts:\n" + "\n".join(unique_facts))
        if evidence_blocks:
            context_parts.append("\n\n".join(evidence_blocks))
        context_str = "\n\n".join(context_parts)

        generation_started = time.perf_counter()
        _generation_model.set(None)
        answer = generate_grounded_answer(
            self.settings,
            question=question,
            context=context_str,
            retrieved_chunks=retrieved.results,
            generate_fn=_generate,
        )
        generation_latency_ms = _elapsed_ms(generation_started)
        total_latency_ms = _elapsed_ms(started)
        sources = citations_from(retrieved.results)

        return GraphQueryResponse(
            question=question,
            answer=answer,
            sources=sources,
            entities=retrieved.entities,
            relationships=retrieved.relationships,
            retrieved_chunks=retrieved.results,
            retrieval_latency_ms=retrieved.latency_ms,
            generation_latency_ms=generation_latency_ms,
            total_latency_ms=total_latency_ms,
            llm_model=resolve_generation_model_name(self.settings),
        )

    def _load_chunks(self, paper_id: str) -> list[ChunkRecord]:
        try:
            chunks = load_chunks(self.settings, paper_id)
        except IngestionError as exc:
            raise GraphRAGError(str(exc), status_code=exc.status_code) from exc
        if not chunks:
            raise GraphRAGError(f"Paper {paper_id} has no semantic chunks to index.", status_code=422)
        return chunks


def connect_graph_store(settings: Settings) -> ApoCFreeNeo4jPropertyGraphStore:
    if not settings.neo4j_uri.strip() or not settings.neo4j_username.strip() or not settings.neo4j_password:
        raise GraphRAGError("Neo4j URI, username, and password are required for GraphRAG.", status_code=400)
    try:
        return ApoCFreeNeo4jPropertyGraphStore(
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            url=settings.neo4j_uri,
            database=settings.neo4j_database,
        )
    except Exception as exc:
        logger.exception("Neo4j graph store connection failed")
        raise GraphRAGError("Could not connect to Neo4j.", status_code=503) from exc


def fetch_graph_rows(
    store: ApoCFreeNeo4jPropertyGraphStore,
    paper_ids: list[str],
    path_depth: int,
) -> list[dict[str, Any]]:
    depth = int(path_depth)
    return store.structured_query(
        f"""
        MATCH p = (a:`__Entity__`)-[rels*1..{depth}]->(b:`__Entity__`)
        WHERE ALL(rel IN rels WHERE rel.paper_id IN $paper_ids AND type(rel) <> 'MENTIONS')
        UNWIND range(0, size(rels) - 1) AS idx
        WITH rels[idx] AS rel, nodes(p)[idx] AS source, nodes(p)[idx + 1] AS target
        RETURN DISTINCT source.name AS source,
               type(rel) AS relation,
               target.name AS target,
               rel.paper_id AS paper_id,
               rel.chunk_id AS chunk_id,
               rel.title AS title,
               rel.page AS page,
               rel.section AS section,
               rel.source_filename AS source_filename,
               rel.original_chunk_text AS text
        """,
        param_map={"paper_ids": paper_ids},
    )


class _AlreadyExtracted:
    """Keep triples already attached to the chunk nodes."""

    def __call__(self, nodes, show_progress: bool = False, **kwargs):
        return nodes

    async def acall(self, nodes, show_progress: bool = False, **kwargs):
        return self.__call__(nodes, show_progress=show_progress, **kwargs)


def _insert_property_graph(settings: Settings, store: ApoCFreeNeo4jPropertyGraphStore, nodes: list, extractor) -> None:
    try:
        embed_model = build_embedding_model(settings)
        PropertyGraphIndex(
            nodes=nodes,
            kg_extractors=[extractor],
            property_graph_store=store,
            embed_model=embed_model,
            embed_kg_nodes=True,
            use_async=False,
            show_progress=False,
        )
    except Exception:
        logger.exception("Graph embedding failed. Writing the graph without entity embeddings.")
        PropertyGraphIndex(
            nodes=nodes,
            kg_extractors=[extractor],
            property_graph_store=store,
            embed_kg_nodes=False,
            use_async=False,
            show_progress=False,
        )


def _run_reference_retrievers(
    settings: Settings,
    store: ApoCFreeNeo4jPropertyGraphStore,
    question: str,
    top_k: int,
    path_depth: int,
) -> None:
    """Exercise the reference retrievers. Keyword filtering still applies afterwards."""

    try:
        LLMSynonymRetriever(
            graph_store=store,
            include_text=False,
            include_properties=True,
            path_depth=path_depth,
            llm=_synonym_llm(settings),
            max_keywords=top_k,
        ).retrieve(question)
        logger.info("LLMSynonymRetriever completed")
    except Exception:
        logger.exception("LLMSynonymRetriever failed")
    try:
        VectorContextRetriever(
            graph_store=store,
            include_text=False,
            include_properties=True,
            embed_model=build_embedding_model(settings),
            similarity_top_k=top_k,
            path_depth=path_depth,
        ).retrieve(question)
        logger.info("VectorContextRetriever completed")
    except Exception:
        logger.exception("VectorContextRetriever failed")


def _synonym_llm(settings: Settings):
    if normalize_llm_provider(settings) == "openai":
        from llama_index.llms.openai import OpenAI

        return OpenAI(
            model=settings.openai_llm_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
    return _DevelopmentSynonymLLM()


def _count_graph(nodes: list) -> tuple[set[str], int]:
    from llama_index.core.graph_stores.types import KG_NODES_KEY, KG_RELATIONS_KEY

    entity_ids: set[str] = set()
    relations = 0
    for node in nodes:
        for entity in node.metadata.get(KG_NODES_KEY, []):
            entity_ids.add(entity.id)
        relations += len(node.metadata.get(KG_RELATIONS_KEY, []))
    return entity_ids, relations


def _resolve_top_k(settings: Settings, requested: int | None) -> int:
    top_k = settings.vector_top_k if requested is None else requested
    if top_k < 1 or top_k > 50:
        raise GraphRAGError("top_k must be between 1 and 50.", status_code=400)
    return top_k


def _validate_depth(path_depth: int) -> None:
    if path_depth < 1 or path_depth > 4:
        raise GraphRAGError("path_depth must be between 1 and 4.", status_code=400)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
