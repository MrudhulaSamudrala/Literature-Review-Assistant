"""HybridRAG composes dense vector search, BM25 lexical retrieval, RRF fusion, and graph evidence."""

import logging
import math
import re
import time
from collections import Counter

from app.config import Settings
from app.rag.graph_rag.models import GraphEntity, GraphRelationship
from app.rag.graph_rag.service import GraphRAGService
from app.rag.hybrid_rag.models import HybridEvidence, HybridQueryResponse
from app.rag.vector_rag.models import Citation, RetrievedChunk
from app.rag.vector_rag.prompts import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    NO_EVIDENCE_ANSWER,
    clean_and_validate_answer,
    filter_table_noise,
)
from app.rag.vector_rag.service import (
    VectorRAGService,
    _generate,
    _generation_model,
    configured_llm_model_name,
    generate_grounded_answer,
    resolve_generation_model_name,
)

logger = logging.getLogger(__name__)


class BM25Retriever:
    """Okapi BM25 lexical retriever for paper chunks."""

    def __init__(self, corpus: list[RetrievedChunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_len = [len(self._tokenize(doc.text)) for doc in corpus]
        self.avgdl = sum(self.doc_len) / max(1, len(corpus))
        self.doc_freqs: Counter[str] = Counter()
        self.doc_term_counts: list[Counter[str]] = []
        for doc in corpus:
            term_counts = Counter(self._tokenize(doc.text))
            self.doc_term_counts.append(term_counts)
            for term in term_counts:
                self.doc_freqs[term] += 1
        self.idf = {
            term: math.log(1 + (len(corpus) - count + 0.5) / (count + 0.5))
            for term, count in self.doc_freqs.items()
        }

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def get_top_n(self, query: str, n: int) -> list[RetrievedChunk]:
        query_terms = self._tokenize(query)
        scores: list[tuple[float, RetrievedChunk]] = []
        for idx, (doc, doc_len, term_counts) in enumerate(
            zip(self.corpus, self.doc_len, self.doc_term_counts)
        ):
            score = 0.0
            for term in query_terms:
                if term not in term_counts:
                    continue
                tf = term_counts[term]
                idf = self.idf.get(term, 0.0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                score += idf * (numerator / denominator)
            if score > 0:
                scores.append((score, doc))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [
            doc.model_copy(update={"rank": i + 1, "score": round(sc, 4)})
            for i, (sc, doc) in enumerate(scores[:n])
        ]


def reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    bm25_results: list[RetrievedChunk],
    k: int = 60,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """Fuse dense vector and BM25 lexical rankings via Reciprocal Rank Fusion."""
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}

    for rank, chunk in enumerate(dense_results, start=1):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (1.0 / (k + rank))
        chunk_map[chunk.chunk_id] = chunk

    for rank, chunk in enumerate(bm25_results, start=1):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (1.0 / (k + rank))
        chunk_map[chunk.chunk_id] = chunk

    ranked = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    fused: list[RetrievedChunk] = []
    for rank, (chunk_id, score) in enumerate(ranked[:top_k], start=1):
        original = chunk_map[chunk_id]
        fused.append(
            original.model_copy(update={"rank": rank, "score": round(score, 6)})
        )
    return fused


def fuse_evidence(
    vector_chunks: list[RetrievedChunk],
    graph_chunks: list[RetrievedChunk],
) -> list[HybridEvidence]:
    """Deduplicate chunks by paper_id and chunk_id and record where each came from."""

    combined: dict[tuple[str, str], HybridEvidence] = {}
    for chunk in vector_chunks:
        combined[(chunk.paper_id, chunk.chunk_id)] = _evidence(chunk, "vector")
    for chunk in graph_chunks:
        key = (chunk.paper_id, chunk.chunk_id)
        existing = combined.get(key)
        if existing is None:
            combined[key] = _evidence(chunk, "graph")
            continue
        combined[key] = existing.model_copy(update={"origin": "both"})
    return list(combined.values())


def build_hybrid_prompt(
    question: str,
    evidence: list[HybridEvidence],
    entities: list[GraphEntity],
    relationships: list[GraphRelationship],
) -> str:
    """Ground the answer in fused vector and graph evidence."""
    graph_lines = [
        f"{item.source} -[{item.relation}]-> {item.target}"
        for item in relationships[:6]
        if item.source and item.relation and item.target
    ]
    unique_graph = list(dict.fromkeys(graph_lines))
    graph_block = "\n".join(unique_graph) if unique_graph else "None"
    evidence_blocks = [
        f"Evidence {i}:\n{filter_table_noise(item.text)}"
        for i, item in enumerate(evidence[:3], start=1)
        if filter_table_noise(item.text)
    ]
    chunks = "\n\n".join(evidence_blocks)
    return (
        "Answer the question using the evidence.\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Graph facts:\n{graph_block}\n\n"
        f"Evidence:\n{chunks}\n\n"
        "Give a short, direct answer in 1-3 sentences.\n"
        "- Use only the evidence.\n"
        "- Do not copy the evidence verbatim.\n"
        "- Do not invent facts.\n"
        "- Do not invent citations.\n"
        "- Do not invent graph relationships.\n"
        "- If the evidence is insufficient, explicitly say that the retrieved evidence is insufficient.\n"
        "- Do not mention the evidence, instructions, sources, chunk IDs, or model.\n"
        f'- If the available evidence does not provide enough information to answer this question, say: "{INSUFFICIENT_EVIDENCE_ANSWER}"'
    )


class HybridRAGService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.vector_service = VectorRAGService(settings)
        self.graph_service = GraphRAGService(settings)

    def bm25_retrieve(self, question: str, paper_ids: list[str], top_k: int) -> list[RetrievedChunk]:
        """Perform BM25 lexical search over candidate paper chunks."""
        all_chunks: list[RetrievedChunk] = []
        for paper_id in paper_ids:
            try:
                from app.rag.vector_rag.indexer import load_paper_chunks

                chunks = load_paper_chunks(self.settings, paper_id)
                for chunk in chunks:
                    all_chunks.append(
                        RetrievedChunk(
                            rank=1,
                            score=None,
                            chunk_id=chunk.chunk_id,
                            paper_id=chunk.paper_id,
                            title=chunk.title,
                            page=chunk.page,
                            section=chunk.section,
                            source_filename=chunk.source_filename,
                            text=chunk.text,
                        )
                    )
            except Exception:
                continue
        if not all_chunks:
            return []
        retriever = BM25Retriever(all_chunks)
        return retriever.get_top_n(question, top_k)

    def query(
        self,
        question: str,
        paper_ids: list[str],
        top_k: int | None,
        path_depth: int,
    ) -> HybridQueryResponse:
        started = time.perf_counter()
        logger.info("HybridRAG query started paper_ids=%s top_k=%s path_depth=%s", paper_ids, top_k, path_depth)
        resolved_top_k = self.vector_service.resolve_top_k(top_k)

        # 1. Dense Vector Retrieval
        vector = self.vector_service.retrieve(question, paper_ids, top_k)
        dense_results = vector.results

        # 2. BM25 Lexical Retrieval
        bm25_results = self.bm25_retrieve(question, paper_ids, resolved_top_k)

        # 3. Reciprocal Rank Fusion (RRF)
        if bm25_results:
            rrf_results = reciprocal_rank_fusion(dense_results, bm25_results, top_k=resolved_top_k)
        else:
            rrf_results = dense_results

        # Diagnostic logging for HybridRAG execution
        logger.info("HybridRAG dense result count: %d", len(dense_results))
        logger.info("HybridRAG BM25 result count: %d", len(bm25_results))
        logger.info("HybridRAG RRF result count: %d", len(rrf_results))
        logger.info("HybridRAG final top-K chunk IDs: %s", [c.chunk_id for c in rrf_results])

        # 4. Graph Retrieval
        try:
            graph = self.graph_service.retrieve(question, paper_ids, top_k, path_depth)
        except Exception as exc:
            logger.warning("Graph retrieval unavailable or skipped in HybridRAG: %s", exc)
            graph = None

        vector_evidence = [_evidence(chunk, "vector") for chunk in rrf_results]
        graph_evidence = [_evidence(chunk, "graph") for chunk in (graph.results if graph else [])]
        combined = fuse_evidence(rrf_results, graph.results if graph else [])
        logger.info(
            "Hybrid evidence fused vector=%s graph=%s combined=%s",
            len(vector_evidence),
            len(graph_evidence),
            len(combined),
        )
        if not combined:
            return HybridQueryResponse(
                question=question,
                answer=NO_EVIDENCE_ANSWER,
                sources=[],
                vector_evidence=vector_evidence,
                graph_evidence=graph_evidence,
                combined_evidence=[],
                entities=graph.entities if graph else [],
                relationships=graph.relationships if graph else [],
                vector_retrieval_latency_ms=vector.latency_ms,
                graph_retrieval_latency_ms=graph.latency_ms if graph else 0.0,
                generation_latency_ms=0.0,
                total_latency_ms=_elapsed_ms(started),
                embedding_model=vector.embedding_model,
                llm_model=configured_llm_model_name(self.settings),
            )

        candidate_chunks = combined[:3]
        evidence_blocks = [
            f"Evidence {i}:\n{filter_table_noise(item.text)}"
            for i, item in enumerate(candidate_chunks, start=1)
            if filter_table_noise(item.text)
        ]
        graph_facts = [
            f"{rel.source} -[{rel.relation}]-> {rel.target}"
            for rel in (graph.relationships if graph else [])[:6]
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
            retrieved_chunks=[
                RetrievedChunk(
                    rank=i + 1,
                    score=item.score,
                    chunk_id=item.chunk_id,
                    paper_id=item.paper_id,
                    title=item.title,
                    page=item.page,
                    section=item.section,
                    source_filename=item.source_filename,
                    text=item.text,
                )
                for i, item in enumerate(combined)
            ],
            generate_fn=_generate,
        )
        generation_latency_ms = _elapsed_ms(generation_started)
        total_latency_ms = _elapsed_ms(started)
        sources = _citations(combined)

        return HybridQueryResponse(
            question=question,
            answer=answer,
            sources=sources,
            vector_evidence=vector_evidence,
            graph_evidence=graph_evidence,
            combined_evidence=combined,
            entities=graph.entities if graph else [],
            relationships=graph.relationships if graph else [],
            vector_retrieval_latency_ms=vector.latency_ms,
            graph_retrieval_latency_ms=graph.latency_ms if graph else 0.0,
            generation_latency_ms=generation_latency_ms,
            total_latency_ms=total_latency_ms,
            embedding_model=vector.embedding_model,
            llm_model=resolve_generation_model_name(self.settings),
        )


def _evidence(chunk: RetrievedChunk, origin: str) -> HybridEvidence:
    return HybridEvidence(
        paper_id=chunk.paper_id,
        title=chunk.title,
        page=chunk.page,
        section=chunk.section,
        chunk_id=chunk.chunk_id,
        source_filename=chunk.source_filename,
        text=chunk.text,
        score=chunk.score,
        origin=origin,
    )


def _format_evidence(item: HybridEvidence) -> str:
    return filter_table_noise(item.text)


def _citations(evidence: list[HybridEvidence]) -> list[Citation]:
    return [
        Citation(
            paper_id=item.paper_id,
            title=item.title,
            page=item.page,
            chunk_id=item.chunk_id,
            source_filename=item.source_filename,
        )
        for item in evidence
    ]


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
