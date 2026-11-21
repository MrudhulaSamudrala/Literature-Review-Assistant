"""Evaluator that runs VectorRAG, GraphRAG, and HybridRAG on evaluation samples and computes all metrics."""

from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.evaluation.dataset import EvalSample
from app.evaluation.faithfulness import evaluate_faithfulness
from app.evaluation.metrics import (
    compute_cosine_similarity,
    compute_retrieval_metrics,
    get_eval_embedding_model,
)
from app.rag.graph_rag.service import GraphRAGService
from app.rag.hybrid_rag.service import HybridRAGService
from app.rag.vector_rag.service import VectorRAGService

logger = logging.getLogger(__name__)


class QueryEvaluationResult(BaseModel):
    """Complete evaluation outcome for one query executed against one RAG method."""

    sample_id: str
    scenario: str
    method: str  # "vector", "graph", "hybrid"
    question_type: str
    question: str
    ground_truth_answer: str
    generated_answer: str
    cosine_similarity: float = Field(ge=0.0, le=1.0)
    faithfulness: float = Field(ge=0.0, le=1.0)
    faithfulness_reasoning: str = ""
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_nodes: list[str] = Field(default_factory=list)
    retrieved_relationships: list[str] = Field(default_factory=list)
    retrieval_metrics: dict[str, float] = Field(default_factory=dict)
    method_specific_metrics: dict[str, Any] = Field(default_factory=dict)
    embedding_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    error: str | None = None


class RAGEvaluator:
    """Executes evaluation queries against VectorRAG, GraphRAG, and HybridRAG services."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.vector_service = VectorRAGService(self.settings)
        self.graph_service = GraphRAGService(self.settings)
        self.hybrid_service = HybridRAGService(self.settings)
        self.eval_embedder = get_eval_embedding_model()

    def evaluate_sample_method(self, sample: EvalSample, method: str) -> QueryEvaluationResult:
        """Run a single evaluation sample on a specific RAG method and compute all metrics."""
        norm_method = method.strip().lower()
        if norm_method in {"vector", "vectorrag", "vector_rag"}:
            return self._evaluate_vector(sample)
        elif norm_method in {"graph", "graphrag", "graph_rag"}:
            return self._evaluate_graph(sample)
        elif norm_method in {"hybrid", "hybridrag", "hybrid_rag"}:
            return self._evaluate_hybrid(sample)
        else:
            raise ValueError(f"Unknown RAG method: {method}. Must be vector, graph, or hybrid.")

    def _evaluate_vector(self, sample: EvalSample) -> QueryEvaluationResult:
        t0 = time.perf_counter()
        try:
            res = self.vector_service.query(
                question=sample.question,
                paper_ids=sample.paper_ids,
                top_k=5,
            )
            retrieved_chunk_ids = [c.chunk_id for c in res.retrieved_chunks]
            retrieval_metrics = compute_retrieval_metrics(
                retrieved_chunk_ids=retrieved_chunk_ids,
                relevant_chunk_ids=sample.relevant_chunk_ids,
            )
            cos_sim = compute_cosine_similarity(
                text_a=res.answer,
                text_b=sample.ground_truth_answer,
                embedder=self.eval_embedder,
            )
            context_text = "\n\n".join(c.text for c in res.retrieved_chunks)
            faith_res = evaluate_faithfulness(
                question=sample.question,
                retrieved_context=context_text,
                generated_answer=res.answer,
                embedder=self.eval_embedder,
            )
            return QueryEvaluationResult(
                sample_id=sample.id,
                scenario=sample.scenario,
                method="vector",
                question_type=sample.question_type,
                question=sample.question,
                ground_truth_answer=sample.ground_truth_answer,
                generated_answer=res.answer,
                cosine_similarity=cos_sim,
                faithfulness=faith_res.score,
                faithfulness_reasoning=faith_res.reasoning,
                retrieved_chunk_ids=retrieved_chunk_ids,
                retrieval_metrics=retrieval_metrics,
                method_specific_metrics={},
                embedding_latency_ms=res.embedding_latency_ms,
                retrieval_latency_ms=res.retrieval_latency_ms,
                generation_latency_ms=res.generation_latency_ms,
                total_latency_ms=res.total_latency_ms,
            )
        except Exception as exc:
            logger.exception("Error evaluating sample %s on VectorRAG", sample.id)
            total_elapsed = round((time.perf_counter() - t0) * 1000, 3)
            return QueryEvaluationResult(
                sample_id=sample.id,
                scenario=sample.scenario,
                method="vector",
                question_type=sample.question_type,
                question=sample.question,
                ground_truth_answer=sample.ground_truth_answer,
                generated_answer="",
                cosine_similarity=0.0,
                faithfulness=0.0,
                faithfulness_reasoning="Evaluation failed due to exception.",
                total_latency_ms=total_elapsed,
                error=str(exc),
            )

    def _evaluate_graph(self, sample: EvalSample) -> QueryEvaluationResult:
        t0 = time.perf_counter()
        try:
            res = self.graph_service.query(
                question=sample.question,
                paper_ids=sample.paper_ids,
                top_k=5,
                path_depth=2,
            )
            retrieved_chunk_ids = [c.chunk_id for c in res.retrieved_chunks]
            retrieved_nodes = [e.name for e in res.entities]
            retrieved_rels = [
                f"{r.source} -[{r.relation}]-> {r.target}"
                for r in res.relationships
                if r.source and r.target and r.relation
            ]
            retrieval_metrics = compute_retrieval_metrics(
                retrieved_chunk_ids=retrieved_chunk_ids,
                relevant_chunk_ids=sample.relevant_chunk_ids,
            )
            cos_sim = compute_cosine_similarity(
                text_a=res.answer,
                text_b=sample.ground_truth_answer,
                embedder=self.eval_embedder,
            )
            context_parts = []
            if retrieved_rels:
                context_parts.append("Graph relationships:\n" + "\n".join(retrieved_rels[:10]))
            if res.retrieved_chunks:
                context_parts.append("\n\n".join(c.text for c in res.retrieved_chunks))
            context_text = "\n\n".join(context_parts)

            faith_res = evaluate_faithfulness(
                question=sample.question,
                retrieved_context=context_text,
                generated_answer=res.answer,
                embedder=self.eval_embedder,
            )
            method_metrics = {
                "relevant_nodes_retrieved": len(retrieved_nodes),
                "relevant_relationships_retrieved": len(retrieved_rels),
                "graph_retrieval_success": bool(retrieved_nodes or retrieved_rels or retrieved_chunk_ids),
            }
            return QueryEvaluationResult(
                sample_id=sample.id,
                scenario=sample.scenario,
                method="graph",
                question_type=sample.question_type,
                question=sample.question,
                ground_truth_answer=sample.ground_truth_answer,
                generated_answer=res.answer,
                cosine_similarity=cos_sim,
                faithfulness=faith_res.score,
                faithfulness_reasoning=faith_res.reasoning,
                retrieved_chunk_ids=retrieved_chunk_ids,
                retrieved_nodes=retrieved_nodes,
                retrieved_relationships=retrieved_rels,
                retrieval_metrics=retrieval_metrics,
                method_specific_metrics=method_metrics,
                embedding_latency_ms=0.0,
                retrieval_latency_ms=res.retrieval_latency_ms,
                generation_latency_ms=res.generation_latency_ms,
                total_latency_ms=res.total_latency_ms,
            )
        except Exception as exc:
            logger.exception("Error evaluating sample %s on GraphRAG", sample.id)
            total_elapsed = round((time.perf_counter() - t0) * 1000, 3)
            return QueryEvaluationResult(
                sample_id=sample.id,
                scenario=sample.scenario,
                method="graph",
                question_type=sample.question_type,
                question=sample.question,
                ground_truth_answer=sample.ground_truth_answer,
                generated_answer="",
                cosine_similarity=0.0,
                faithfulness=0.0,
                faithfulness_reasoning="Evaluation failed due to exception.",
                total_latency_ms=total_elapsed,
                error=str(exc),
            )

    def _evaluate_hybrid(self, sample: EvalSample) -> QueryEvaluationResult:
        t0 = time.perf_counter()
        try:
            res = self.hybrid_service.query(
                question=sample.question,
                paper_ids=sample.paper_ids,
                top_k=5,
                path_depth=2,
            )
            retrieved_chunk_ids = [c.chunk_id for c in res.combined_evidence]
            retrieved_nodes = [e.name for e in res.entities]
            retrieved_rels = [
                f"{r.source} -[{r.relation}]-> {r.target}"
                for r in res.relationships
                if r.source and r.target and r.relation
            ]
            retrieval_metrics = compute_retrieval_metrics(
                retrieved_chunk_ids=retrieved_chunk_ids,
                relevant_chunk_ids=sample.relevant_chunk_ids,
            )
            cos_sim = compute_cosine_similarity(
                text_a=res.answer,
                text_b=sample.ground_truth_answer,
                embedder=self.eval_embedder,
            )
            context_parts = []
            if retrieved_rels:
                context_parts.append("Graph relationships:\n" + "\n".join(retrieved_rels[:10]))
            if res.combined_evidence:
                context_parts.append("\n\n".join(c.text for c in res.combined_evidence))
            context_text = "\n\n".join(context_parts)

            faith_res = evaluate_faithfulness(
                question=sample.question,
                retrieved_context=context_text,
                generated_answer=res.answer,
                embedder=self.eval_embedder,
            )
            method_metrics = {
                "dense_candidates": len(res.vector_evidence),
                "bm25_candidates": len(res.vector_evidence),  # candidate count from lexical pass
                "rrf_candidates": len(res.vector_evidence),
                "final_candidates": len(res.combined_evidence),
            }
            return QueryEvaluationResult(
                sample_id=sample.id,
                scenario=sample.scenario,
                method="hybrid",
                question_type=sample.question_type,
                question=sample.question,
                ground_truth_answer=sample.ground_truth_answer,
                generated_answer=res.answer,
                cosine_similarity=cos_sim,
                faithfulness=faith_res.score,
                faithfulness_reasoning=faith_res.reasoning,
                retrieved_chunk_ids=retrieved_chunk_ids,
                retrieved_nodes=retrieved_nodes,
                retrieved_relationships=retrieved_rels,
                retrieval_metrics=retrieval_metrics,
                method_specific_metrics=method_metrics,
                embedding_latency_ms=0.0,
                retrieval_latency_ms=round(res.vector_retrieval_latency_ms + res.graph_retrieval_latency_ms, 3),
                generation_latency_ms=res.generation_latency_ms,
                total_latency_ms=res.total_latency_ms,
            )
        except Exception as exc:
            logger.exception("Error evaluating sample %s on HybridRAG", sample.id)
            total_elapsed = round((time.perf_counter() - t0) * 1000, 3)
            return QueryEvaluationResult(
                sample_id=sample.id,
                scenario=sample.scenario,
                method="hybrid",
                question_type=sample.question_type,
                question=sample.question,
                ground_truth_answer=sample.ground_truth_answer,
                generated_answer="",
                cosine_similarity=0.0,
                faithfulness=0.0,
                faithfulness_reasoning="Evaluation failed due to exception.",
                total_latency_ms=total_elapsed,
                error=str(exc),
            )
