"""Faithfulness evaluation assessing whether generated answers are grounded in retrieved evidence."""

from __future__ import annotations

import logging
import re
from typing import Sequence

from pydantic import BaseModel, Field

from app.evaluation.metrics import compute_cosine_similarity, compute_mean_and_std
from app.rag.vector_rag.prompts import INSUFFICIENT_EVIDENCE_ANSWER, NO_EVIDENCE_ANSWER

logger = logging.getLogger(__name__)

# Common English stopwords to ignore in content-word grounding
_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "did", "do", "does", "doing", "don", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she",
    "should", "so", "some", "such", "t", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "would", "you", "your", "yours", "yourself",
}


class ClaimSupport(BaseModel):
    """Evaluation result for an individual factual claim."""

    claim: str
    is_supported: bool
    support_score: float = Field(ge=0.0, le=1.0)
    reasoning: str


class FaithfulnessResult(BaseModel):
    """Overall faithfulness evaluation for a question-answer pair."""

    score: float = Field(ge=0.0, le=1.0)
    claims: list[ClaimSupport] = Field(default_factory=list)
    reasoning: str


def extract_claims(text: str) -> list[str]:
    """Segment generated answer text into atomic sentences / factual claims."""
    clean = text.strip()
    if not clean:
        return []

    # Split by sentence boundaries: period, exclamation, question mark, or newlines
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", clean)
    claims: list[str] = []
    for s in raw_sentences:
        s_clean = s.strip()
        # Filter out trivial fragments or isolated punctuation
        if len(s_clean) > 5 and re.search(r"[A-Za-z0-9]", s_clean):
            claims.append(s_clean)

    return claims if claims else [clean]


def _extract_content_tokens(text: str) -> set[str]:
    """Extract lowercased alphanumeric tokens excluding stopwords."""
    words = re.findall(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*", text.lower())
    return {w for w in words if len(w) > 1 and w not in _STOPWORDS}


def is_claim_supported(claim: str, context: str, embedder=None) -> tuple[bool, float, str]:
    """Assess whether a specific factual claim is supported by the context."""
    if not context or not context.strip():
        return False, 0.0, "Context is empty; claim cannot be supported."

    claim_tokens = _extract_content_tokens(claim)
    if not claim_tokens:
        # Trivial claim with no content words
        return True, 1.0, "Claim contains no specific factual assertions."

    context_tokens = _extract_content_tokens(context)
    overlap = claim_tokens.intersection(context_tokens)
    token_precision = len(overlap) / len(claim_tokens)

    # Context sentence split for fine-grained semantic match
    context_sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", context) if len(s.strip()) > 10
    ]
    if not context_sentences:
        context_sentences = [context.strip()]

    # Semantic similarity against best matching sentence in context
    max_sim = 0.0
    for cs in context_sentences:
        sim = compute_cosine_similarity(claim, cs, embedder=embedder)
        if sim > max_sim:
            max_sim = sim

    # Combined support heuristic:
    # 1. High semantic similarity to a context sentence (>= 0.70)
    # 2. Or high token precision (>= 0.60) and moderate similarity (>= 0.50)
    # 3. Or full token containment (token_precision >= 0.85)
    if max_sim >= 0.70:
        supported = True
        score = max_sim
        reason = f"High semantic entailment (max_sim={max_sim:.3f}, token_prec={token_precision:.2f})"
    elif token_precision >= 0.60 and max_sim >= 0.48:
        supported = True
        score = max_sim
        reason = f"Grounded keywords & semantic match (token_prec={token_precision:.2f}, sim={max_sim:.3f})"
    elif token_precision >= 0.80:
        supported = True
        score = max(max_sim, token_precision)
        reason = f"Strong lexical overlap with context (token_prec={token_precision:.2f})"
    else:
        supported = False
        score = max_sim
        reason = f"Insufficient support (max_sim={max_sim:.3f}, token_prec={token_precision:.2f})"

    return supported, round(score, 4), reason


def evaluate_faithfulness(
    question: str,
    retrieved_context: str,
    generated_answer: str,
    embedder=None,
) -> FaithfulnessResult:
    """Evaluate whether the generated answer is faithful to the retrieved context.

    Returns a normalized faithfulness score between 0.0 and 1.0.
    """
    clean_ans = generated_answer.strip()
    clean_ctx = retrieved_context.strip()

    # Case 1: Refusal / insufficient evidence fallback
    # If the system explicitly states that evidence is insufficient, it makes zero
    # unsupported claims. This is perfectly faithful to the absence/insufficiency of evidence.
    refusal_markers = [
        INSUFFICIENT_EVIDENCE_ANSWER.lower(),
        NO_EVIDENCE_ANSWER.lower(),
        "does not provide enough information",
        "insufficient evidence",
        "evidence does not provide",
        "no evidence found",
    ]
    if any(marker in clean_ans.lower() for marker in refusal_markers):
        return FaithfulnessResult(
            score=1.0,
            claims=[
                ClaimSupport(
                    claim=clean_ans,
                    is_supported=True,
                    support_score=1.0,
                    reasoning="System correctly refrained from fabricating facts on insufficient evidence.",
                )
            ],
            reasoning="Valid refusal on insufficient evidence: faithfulness = 1.0",
        )

    # Case 2: Generated answer is empty
    if not clean_ans:
        return FaithfulnessResult(
            score=0.0,
            claims=[],
            reasoning="Generated answer is empty.",
        )

    # Case 3: Context is empty but system gave an answer
    if not clean_ctx:
        return FaithfulnessResult(
            score=0.0,
            claims=[
                ClaimSupport(
                    claim=clean_ans,
                    is_supported=False,
                    support_score=0.0,
                    reasoning="No context was retrieved to support this answer.",
                )
            ],
            reasoning="Context is empty while answer makes factual claims: faithfulness = 0.0",
        )

    # Case 4: Standard factual answer with context
    claims = extract_claims(clean_ans)
    claim_evals: list[ClaimSupport] = []
    supported_count = 0

    for claim in claims:
        supported, score, reason = is_claim_supported(claim, clean_ctx, embedder=embedder)
        if supported:
            supported_count += 1
        claim_evals.append(
            ClaimSupport(
                claim=claim,
                is_supported=supported,
                support_score=score,
                reasoning=reason,
            )
        )

    overall_score = round(supported_count / max(1, len(claims)), 4)
    summary_reason = f"{supported_count}/{len(claims)} claims supported by retrieved evidence."

    return FaithfulnessResult(
        score=overall_score,
        claims=claim_evals,
        reasoning=summary_reason,
    )
