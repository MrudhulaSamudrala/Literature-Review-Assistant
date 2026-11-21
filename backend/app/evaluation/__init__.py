"""Evaluation framework for VectorRAG, GraphRAG, and HybridRAG."""

from app.evaluation.dataset import EvalDataset, EvalSample, get_default_evaluation_dataset
from app.evaluation.evaluator import QueryEvaluationResult, RAGEvaluator
from app.evaluation.faithfulness import FaithfulnessResult, evaluate_faithfulness
from app.evaluation.metrics import (
    compute_cosine_similarity,
    compute_mean_and_std,
    compute_retrieval_metrics,
    format_mean_std,
)
from app.evaluation.report import EvaluationReporter
from app.evaluation.runner import run_evaluation

__all__ = [
    "EvalSample",
    "EvalDataset",
    "get_default_evaluation_dataset",
    "RAGEvaluator",
    "QueryEvaluationResult",
    "evaluate_faithfulness",
    "FaithfulnessResult",
    "compute_cosine_similarity",
    "compute_retrieval_metrics",
    "compute_mean_and_std",
    "format_mean_std",
    "EvaluationReporter",
    "run_evaluation",
]
