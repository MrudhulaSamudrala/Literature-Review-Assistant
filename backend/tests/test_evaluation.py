"""Unit tests for the RAG evaluation framework."""

import json
from pathlib import Path

import pytest

from app.evaluation.dataset import (
    EvalDataset,
    EvalSample,
    get_default_evaluation_dataset,
)
from app.evaluation.evaluator import QueryEvaluationResult
from app.evaluation.faithfulness import (
    evaluate_faithfulness,
    extract_claims,
    is_claim_supported,
)
from app.evaluation.metrics import (
    compute_cosine_similarity,
    compute_mean_and_std,
    compute_retrieval_metrics,
    format_mean_std,
)
from app.evaluation.report import EvaluationReporter


def test_dataset_loading_and_filtering(tmp_path: Path) -> None:
    ds = get_default_evaluation_dataset()
    assert len(ds.samples) == 20
    single = ds.filter_by_scenario("single_paper")
    multi = ds.filter_by_scenario("multi_paper")
    assert len(single) == 10
    assert len(multi) == 10

    for s in ds.samples:
        assert s.id
        assert s.question
        assert s.ground_truth_answer
        assert len(s.paper_ids) >= 1
        assert len(s.relevant_chunk_ids) >= 1

    # Test serialization
    dump_path = tmp_path / "test_eval_dataset.json"
    ds.save(dump_path)
    loaded = EvalDataset.load(dump_path)
    assert len(loaded.samples) == 20
    assert loaded.samples[0].id == ds.samples[0].id


def test_retrieval_metrics() -> None:
    retrieved = ["C1", "C2", "C3", "C4", "C5", "C6"]
    relevant = ["C3", "C5"]

    metrics = compute_retrieval_metrics(retrieved, relevant)
    assert metrics["hit_at_1"] == 0.0
    assert metrics["hit_at_3"] == 1.0
    assert metrics["hit_at_5"] == 1.0
    assert metrics["recall_at_5"] == 1.0
    assert metrics["mrr"] == round(1.0 / 3, 4)

    # Empty relevant set
    empty_metrics = compute_retrieval_metrics(retrieved, [])
    assert empty_metrics["hit_at_1"] == 0.0
    assert empty_metrics["mrr"] == 0.0


def test_mean_and_std() -> None:
    values = [0.8, 0.85, 0.9, 0.95]
    mean_val, std_val = compute_mean_and_std(values)
    assert mean_val == 0.875
    assert std_val > 0.0
    formatted = format_mean_std(mean_val, std_val)
    assert "0.875" in formatted

    empty_mean, empty_std = compute_mean_and_std([])
    assert empty_mean == 0.0
    assert empty_std == 0.0


def test_faithfulness_logic() -> None:
    # Insufficient evidence refusal should be 1.0
    refusal_res = evaluate_faithfulness(
        question="What is X?",
        retrieved_context="",
        generated_answer="The available evidence does not provide enough information to answer this question.",
    )
    assert refusal_res.score == 1.0

    # Empty context with factual claim should be 0.0
    unsupported_res = evaluate_faithfulness(
        question="What is X?",
        retrieved_context="",
        generated_answer="X is a fast vector database.",
    )
    assert unsupported_res.score == 0.0

    # Grounded answer
    context = "VectorRAG uses FAISS for vector indexing and retrieval."
    answer = "VectorRAG uses FAISS for vector indexing."
    grounded_res = evaluate_faithfulness(
        question="What does VectorRAG use?",
        retrieved_context=context,
        generated_answer=answer,
    )
    assert grounded_res.score == 1.0


def test_extract_claims() -> None:
    text = "VectorRAG uses FAISS. GraphRAG uses Neo4j! HybridRAG combines them."
    claims = extract_claims(text)
    assert len(claims) == 3


def test_reporter_aggregation_and_exports(tmp_path: Path) -> None:
    reporter = EvaluationReporter(output_dir=tmp_path)
    sample_res = [
        QueryEvaluationResult(
            sample_id="SP_001",
            scenario="single_paper",
            method="vector",
            question_type="methodology",
            question="What is VectorRAG?",
            ground_truth_answer="VectorRAG is vector retrieval.",
            generated_answer="VectorRAG is vector retrieval.",
            cosine_similarity=1.0,
            faithfulness=1.0,
            retrieved_chunk_ids=["P017_C009"],
            retrieval_metrics={"hit_at_1": 1.0, "hit_at_5": 1.0, "mrr": 1.0},
            retrieval_latency_ms=10.0,
            generation_latency_ms=50.0,
            total_latency_ms=60.0,
        ),
        QueryEvaluationResult(
            sample_id="SP_001",
            scenario="single_paper",
            method="hybrid",
            question_type="methodology",
            question="What is HybridRAG?",
            ground_truth_answer="HybridRAG is hybrid retrieval.",
            generated_answer="HybridRAG combines vector and graph.",
            cosine_similarity=0.85,
            faithfulness=1.0,
            retrieved_chunk_ids=["P017_C009"],
            retrieval_metrics={"hit_at_1": 1.0, "hit_at_5": 1.0, "mrr": 1.0},
            retrieval_latency_ms=15.0,
            generation_latency_ms=60.0,
            total_latency_ms=75.0,
        ),
    ]

    aggregated = reporter.aggregate_results(sample_res)
    assert "single_paper_vector" in aggregated
    assert "single_paper_hybrid" in aggregated
    assert aggregated["single_paper_vector"]["cosine_similarity_mean"] == 1.0

    json_path = reporter.export_json(sample_res, aggregated)
    assert json_path.exists()

    csv_path = reporter.export_csv(sample_res)
    assert csv_path.exists()

    charts = reporter.generate_charts(aggregated)
    assert len(charts) == 5
    for c in charts:
        assert c.exists()

    md_path = reporter.generate_markdown_report(sample_res, aggregated)
    assert md_path.exists()
    content = md_path.read_text(encoding="utf-8")
    assert "Executive Summary" in content
    assert "Comparison Against Base Paper" in content
