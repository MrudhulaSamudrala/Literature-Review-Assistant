"""Evaluation dataset schema and curated benchmark dataset for RAG comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class EvalSample(BaseModel):
    """Single evaluation sample representing a test question and ground truth."""

    id: str = Field(description="Unique sample identifier (e.g. SP_001, MP_001)")
    question: str = Field(description="Evaluation question")
    ground_truth_answer: str = Field(description="Reference / ground-truth factual answer")
    paper_ids: list[str] = Field(description="List of paper IDs needed for context")
    reference_passage: str = Field(description="Context or reference excerpt from the source text")
    question_type: str = Field(
        description="Type of question (e.g. factoid, methodology, comparative, architecture, evaluation)"
    )
    scenario: str = Field(description="Evaluation scenario: 'single_paper' or 'multi_paper'")
    relevant_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Ground-truth chunk IDs containing the necessary evidence for Hit@K/MRR",
    )
    is_synthetic: bool = Field(
        default=False,
        description="Flag indicating if the question/answer was generated synthetically",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional custom metadata")


class EvalDataset(BaseModel):
    """Collection of evaluation samples with serialization and filtering."""

    name: str = "Smart Literature Review RAG Benchmark"
    version: str = "1.0.0"
    description: str = (
        "Evaluation benchmark reproducing the base paper methodology across VectorRAG, "
        "GraphRAG, and HybridRAG in Single-Paper and Multi-Paper scenarios."
    )
    samples: list[EvalSample] = Field(default_factory=list)

    def filter_by_scenario(self, scenario: str) -> list[EvalSample]:
        norm = scenario.strip().lower()
        if norm in {"single", "single_paper", "single-paper"}:
            return [s for s in self.samples if s.scenario == "single_paper"]
        if norm in {"multi", "multi_paper", "multi-paper", "database"}:
            return [s for s in self.samples if s.scenario == "multi_paper"]
        return list(self.samples)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> EvalDataset:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)


# -----------------------------------------------------------------------------
# Curated Milestone 1 Benchmark: 10 Single-Paper (P017) + 10 Multi-Paper (P017, P022)
# -----------------------------------------------------------------------------

SINGLE_PAPER_SAMPLES: list[EvalSample] = [
    EvalSample(
        id="SP_001",
        question="What are the three RAG methods evaluated in the paper?",
        ground_truth_answer="The paper evaluates VectorRAG, GraphRAG, and Hybrid RAG.",
        paper_ids=["P017"],
        reference_passage="These three methods form the basis of our experimental framework: VectorRAG, GraphRAG, and Hybrid RAG.",
        question_type="methodology",
        scenario="single_paper",
        relevant_chunk_ids=["P017_C009", "P017_C010"],
        is_synthetic=False,
    ),
    EvalSample(
        id="SP_002",
        question="What is the default value for maxPathsPerChunk in the prototype implementation?",
        ground_truth_answer="In the prototype, the default value for maxPathsPerChunk is set to 10.",
        paper_ids=["P017"],
        reference_passage="In the prototype, the default value for maxPathsPerChunk is set to 10.",
        question_type="factoid",
        scenario="single_paper",
        relevant_chunk_ids=["P017_C011"],
        is_synthetic=False,
    ),
    EvalSample(
        id="SP_003",
        question="What are the two primary evaluation scenarios examined in the study?",
        ground_truth_answer="The two primary evaluation scenarios are RAG on a Single Paper and RAG Applied to a Database.",
        paper_ids=["P017"],
        reference_passage="To evaluate the effectiveness of our system, we examine two primary use-cases: Single Paper RAG and Database RAG.",
        question_type="evaluation",
        scenario="single_paper",
        relevant_chunk_ids=["P017_C004", "P017_C017"],
        is_synthetic=False,
    ),
    EvalSample(
        id="SP_004",
        question="Which three paradigms characterize the evolution of RAG according to the literature review?",
        ground_truth_answer="The paper details RAG's evolution through three paradigms: Naive RAG, Advanced RAG, and Modular RAG.",
        paper_ids=["P017"],
        reference_passage="The paper thoroughly details RAG's evolution through three paradigms: Naive RAG, Advanced RAG, and Modular RAG.",
        question_type="methodology",
        scenario="single_paper",
        relevant_chunk_ids=["P017_C006"],
        is_synthetic=False,
    ),
    EvalSample(
        id="SP_005",
        question="What does Hybrid RAG combine to improve accuracy and contextual relevance?",
        ground_truth_answer="Hybrid RAG combines both VectorRAG and GraphRAG to improve accuracy and contextual relevance.",
        paper_ids=["P017"],
        reference_passage="Hybrid RAG is an advanced framework that combines both VectorRAG and GraphRAG to improve the accuracy and contextual relevance.",
        question_type="architecture",
        scenario="single_paper",
        relevant_chunk_ids=["P017_C009"],
        is_synthetic=False,
    ),
    EvalSample(
        id="SP_006",
        question="What are the main evaluation metrics used to assess RAG performance in the paper?",
        ground_truth_answer="The main evaluation metrics are Cosine Similarity and Faithfulness, evaluated by Mean and Standard Deviation.",
        paper_ids=["P017"],
        reference_passage="Methods/Metrics Cosine Similarity Faithfulness Cosine Similarity Faithfulness Mean SD.",
        question_type="evaluation",
        scenario="single_paper",
        relevant_chunk_ids=["P017_C019", "P017_C021"],
        is_synthetic=False,
    ),
    EvalSample(
        id="SP_007",
        question="What response options were available to annotators when evaluating whether an answer was derived from context?",
        ground_truth_answer="The annotator selected responses from Yes, No, or Unsure for each question.",
        paper_ids=["P017"],
        reference_passage="Is the answer derived from the context? The annotator selected responses from Yes, No, or Unsure for each question.",
        question_type="evaluation",
        scenario="single_paper",
        relevant_chunk_ids=["P017_C016"],
        is_synthetic=False,
    ),
    EvalSample(
        id="SP_008",
        question="What challenge is observed regarding multi-document or database retrieval?",
        ground_truth_answer="The lower performance in multi-paper retrieval underscores the need for improved retrieval ranking methods.",
        paper_ids=["P017"],
        reference_passage="Challenges in multi-document retrieval: The lower performance in multi-paper retrieval underscores the need for improved retrieval ranking methods.",
        question_type="comparative",
        scenario="single_paper",
        relevant_chunk_ids=["P017_C022"],
        is_synthetic=False,
    ),
    EvalSample(
        id="SP_009",
        question="What framework was introduced by prior work to address challenges with dynamic knowledge sources?",
        ground_truth_answer="The authors introduced a framework called Auepora to address the challenge of dynamic knowledge sources.",
        paper_ids=["P017"],
        reference_passage="The authors introduced a framework called the Auepora to address the challenges of dynamic knowledge sources.",
        question_type="factoid",
        scenario="single_paper",
        relevant_chunk_ids=["P017_C007"],
        is_synthetic=False,
    ),
    EvalSample(
        id="SP_010",
        question="What model was used to generate benchmark test sets in the base paper?",
        ground_truth_answer="Benchmark test sets were generated using a GPT model, with selected outputs annotated for evaluation.",
        paper_ids=["P017"],
        reference_passage="Benchmark test sets were generated using a GPT model, with selected outputs annotated for evaluation.",
        question_type="methodology",
        scenario="single_paper",
        relevant_chunk_ids=["P017_C002"],
        is_synthetic=False,
    ),
]

MULTI_PAPER_SAMPLES: list[EvalSample] = [
    EvalSample(
        id="MP_001",
        question="How does GraphRAG contrast with traditional VectorRAG when indexing source documents?",
        ground_truth_answer="GraphRAG contrasts with VectorRAG by generating a graph index of entities and relationships from source data, rather than relying solely on unstructured vector chunk embeddings.",
        paper_ids=["P017", "P022"],
        reference_passage="GraphRAG contrasts with these approaches by generating a graph index from the source data, then applying graph-based retrieval.",
        question_type="comparative",
        scenario="multi_paper",
        relevant_chunk_ids=["P017_C009", "P022_C008", "P022_C010"],
        is_synthetic=False,
    ),
    EvalSample(
        id="MP_002",
        question="What high-level workflow connects source documents to query responses in GraphRAG?",
        ground_truth_answer="The workflow extracts text chunks from source documents, identifies entities and relationships to build a knowledge graph, and uses graph structures to synthesize responses.",
        paper_ids=["P017", "P022"],
        reference_passage="Figure 1 illustrates the high-level data flow: Source Documents -> Text Chunks -> Entities & Relationships -> Knowledge Graph.",
        question_type="architecture",
        scenario="multi_paper",
        relevant_chunk_ids=["P022_C009", "P022_C010"],
        is_synthetic=False,
    ),
    EvalSample(
        id="MP_003",
        question="How does GraphRAG handle duplicate or co-referent entities across different documents?",
        ground_truth_answer="GraphRAG is resilient to duplicate entities because duplicates are clustered together during community detection and summarization.",
        paper_ids=["P017", "P022"],
        reference_passage="GraphRAG is generally resilient to duplicate entities since duplicates are typically clustered together for summarization.",
        question_type="architecture",
        scenario="multi_paper",
        relevant_chunk_ids=["P022_C013"],
        is_synthetic=False,
    ),
    EvalSample(
        id="MP_004",
        question="What evaluation metrics are used to compare response relevance and retrieval quality across papers?",
        ground_truth_answer="Evaluation utilizes cosine similarity to measure semantic match with reference answers, faithfulness to evaluate context groundedness, and retrieval rank metrics.",
        paper_ids=["P017", "P022"],
        reference_passage="The comparative analysis emphasizes retrieval accuracy and response relevance using Cosine Similarity and Faithfulness.",
        question_type="evaluation",
        scenario="multi_paper",
        relevant_chunk_ids=["P017_C019", "P017_C021", "P022_C003"],
        is_synthetic=False,
    ),
    EvalSample(
        id="MP_005",
        question="Why does multi-paper retrieval show lower accuracy compared to single-paper retrieval?",
        ground_truth_answer="Multi-paper retrieval faces larger corpus distraction, conflicting contexts, and inter-document ambiguity, which underscores the need for improved retrieval ranking.",
        paper_ids=["P017", "P022"],
        reference_passage="The lower performance in multi-paper retrieval underscores the need for improved retrieval ranking methods due to multi-document complexity.",
        question_type="comparative",
        scenario="multi_paper",
        relevant_chunk_ids=["P017_C022"],
        is_synthetic=False,
    ),
    EvalSample(
        id="MP_006",
        question="What role does hierarchical community partition play in graph-based summarization?",
        ground_truth_answer="Each level of the hierarchy provides a community partition covering the nodes in a mutually exclusive manner, enabling global multi-document summarization.",
        paper_ids=["P017", "P022"],
        reference_passage="Each level of this hierarchy provides a community partition that covers the nodes of the graph in a mutually exclusive, collectively exhaustive way.",
        question_type="architecture",
        scenario="multi_paper",
        relevant_chunk_ids=["P022_C014"],
        is_synthetic=False,
    ),
    EvalSample(
        id="MP_007",
        question="How does Hybrid RAG combine multiple retrieval mechanisms to overcome single-mode weaknesses?",
        ground_truth_answer="Hybrid RAG fuses dense vector similarity search, lexical BM25 retrieval, and graph structural relationships to balance precision and recall.",
        paper_ids=["P017", "P022"],
        reference_passage="Hybrid RAG is an advanced framework that combines both VectorRAG and GraphRAG to improve accuracy and contextual relevance.",
        question_type="methodology",
        scenario="multi_paper",
        relevant_chunk_ids=["P017_C009", "P022_C005"],
        is_synthetic=False,
    ),
    EvalSample(
        id="MP_008",
        question="What scoring or rating range does GraphRAG generate to evaluate how helpful intermediate answers are?",
        ground_truth_answer="The model is asked to generate a score between 0 and 100 indicating how helpful the intermediate answer is.",
        paper_ids=["P017", "P022"],
        reference_passage="The LLM is also asked to generate a score between 0-100 indicating how helpful the intermediate answer is.",
        question_type="factoid",
        scenario="multi_paper",
        relevant_chunk_ids=["P022_C015"],
        is_synthetic=False,
    ),
    EvalSample(
        id="MP_009",
        question="What are the limitations of pure vector similarity search when answering corpus-wide or global questions?",
        ground_truth_answer="Pure vector similarity search struggles with global query-focused summarization because it only retrieves top-k isolated text chunks rather than aggregating structural relationships across documents.",
        paper_ids=["P017", "P022"],
        reference_passage="GraphRAG contrasts with baseline RAG by generating a graph index for global summarization across large sections of data where standard vector retrieval fails.",
        question_type="comparative",
        scenario="multi_paper",
        relevant_chunk_ids=["P017_C022", "P022_C006", "P022_C008"],
        is_synthetic=False,
    ),
    EvalSample(
        id="MP_010",
        question="What foundational data structures are established during the indexing phase of scientific literature RAG?",
        ground_truth_answer="The indexing phase establishes vector embeddings for chunk similarity retrieval and property graph nodes and relationships for entity connectivity.",
        paper_ids=["P017", "P022"],
        reference_passage="The indexing phase establishes the foundational data structures: vector stores for chunk retrieval and property graph stores for entity-relationship traversal.",
        question_type="methodology",
        scenario="multi_paper",
        relevant_chunk_ids=["P017_C012", "P022_C009"],
        is_synthetic=False,
    ),
]


def get_default_evaluation_dataset() -> EvalDataset:
    """Return the curated benchmark dataset (10 single-paper + 10 multi-paper)."""
    return EvalDataset(
        name="Smart Literature Review RAG Benchmark",
        version="1.0.0",
        description="Curated benchmark reproducing base paper evaluation categories.",
        samples=SINGLE_PAPER_SAMPLES + MULTI_PAPER_SAMPLES,
    )
