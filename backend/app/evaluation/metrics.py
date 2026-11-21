"""Evaluation metrics for RAG benchmark: Cosine Similarity, Retrieval (Hit@K, Recall@K, MRR), and Statistics."""

from __future__ import annotations

import logging
import math
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Dedicated evaluation embedding model
EVALUATION_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_eval_embedder = None


def get_eval_embedding_model():
    """Lazily load and cache the dedicated evaluation embedding model."""
    global _eval_embedder
    if _eval_embedder is None:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        logger.info("Initializing dedicated evaluation embedding model: %s", EVALUATION_EMBEDDING_MODEL)
        _eval_embedder = HuggingFaceEmbedding(model_name=EVALUATION_EMBEDDING_MODEL)
    return _eval_embedder


def compute_cosine_similarity(text_a: str, text_b: str, embedder=None) -> float:
    """Calculate cosine similarity between text_a and text_b using the evaluation embedder."""
    if not text_a or not text_a.strip() or not text_b or not text_b.strip():
        return 0.0

    emb = embedder or get_eval_embedding_model()
    try:
        vec_a = np.array(emb.get_text_embedding(text_a.strip()), dtype=np.float32)
        vec_b = np.array(emb.get_text_embedding(text_b.strip()), dtype=np.float32)
    except Exception as exc:
        logger.error("Failed to compute text embeddings for cosine similarity: %s", exc)
        return 0.0

    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    return max(0.0, min(1.0, round(sim, 4)))


def compute_retrieval_metrics(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
) -> dict[str, float]:
    """Compute Hit@1, Hit@3, Hit@5, Recall@5, and MRR for retrieved chunks.

    Args:
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs (rank 1 to N).
        relevant_chunk_ids: Ground-truth chunk IDs that contain relevant evidence.
    """
    rel_set = set(relevant_chunk_ids)
    if not rel_set:
        return {
            "hit_at_1": 0.0,
            "hit_at_3": 0.0,
            "hit_at_5": 0.0,
            "recall_at_5": 0.0,
            "mrr": 0.0,
        }

    # Hit@K
    hit_1 = 1.0 if any(cid in rel_set for cid in retrieved_chunk_ids[:1]) else 0.0
    hit_3 = 1.0 if any(cid in rel_set for cid in retrieved_chunk_ids[:3]) else 0.0
    hit_5 = 1.0 if any(cid in rel_set for cid in retrieved_chunk_ids[:5]) else 0.0

    # Recall@5
    top5_set = set(retrieved_chunk_ids[:5])
    overlap_count = len(top5_set.intersection(rel_set))
    recall_5 = round(overlap_count / len(rel_set), 4)

    # MRR (Mean Reciprocal Rank of first relevant item)
    mrr = 0.0
    for rank, cid in enumerate(retrieved_chunk_ids, start=1):
        if cid in rel_set:
            mrr = round(1.0 / rank, 4)
            break

    return {
        "hit_at_1": hit_1,
        "hit_at_3": hit_3,
        "hit_at_5": hit_5,
        "recall_at_5": recall_5,
        "mrr": mrr,
    }


def compute_mean_and_std(values: Sequence[float]) -> tuple[float, float]:
    """Compute sample mean and standard deviation (ddof=1)."""
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return round(float(values[0]), 4), 0.0

    n = len(values)
    mean_val = sum(values) / n
    variance = sum((x - mean_val) ** 2 for x in values) / (n - 1)
    std_val = math.sqrt(variance)
    return round(mean_val, 4), round(std_val, 4)


def format_mean_std(mean_val: float, std_val: float) -> str:
    """Format mean and std as 'X.XXX ± Y.YYY'."""
    return f"{mean_val:.3f} ± {std_val:.3f}"
