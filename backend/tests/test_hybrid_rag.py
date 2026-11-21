from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app
from app.rag.graph_rag.models import (
    GraphEntity,
    GraphPath,
    GraphRelationship,
    GraphRetrievalResponse,
)
from app.rag.hybrid_rag.service import HybridRAGService, build_hybrid_prompt, fuse_evidence
from app.rag.vector_rag.models import RetrievedChunk, VectorRetrievalResponse
from app.rag.vector_rag.prompts import NO_EVIDENCE_ANSWER


def _settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "papers_dir": tmp_path / "papers",
        "processed_dir": tmp_path / "processed",
        "chunks_dir": tmp_path / "chunks",
        "openai_api_key": "",
        "openai_embedding_model": "text-embedding-ada-002",
        "openai_llm_model": "gpt-4o-mini",
        "llm_provider": "local",
        "embedding_provider": "local",
        "local_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "neo4j_password": "secret",
        "vector_top_k": 5,
    }
    values.update(overrides)
    return Settings(**values)


def _chunk(
    chunk_id: str,
    paper_id: str,
    text: str,
    score: float | None = None,
    page: int = 3,
) -> RetrievedChunk:
    return RetrievedChunk(
        rank=1,
        score=score,
        chunk_id=chunk_id,
        paper_id=paper_id,
        title="Example Research Paper" if paper_id == "P001" else f"Paper {paper_id}",
        page=page,
        section="Methodology",
        source_filename=f"{paper_id}.pdf",
        text=text,
    )


def _vector_response(chunks: list[RetrievedChunk]) -> VectorRetrievalResponse:
    return VectorRetrievalResponse(
        question="What method is discussed?",
        results=chunks,
        evidence_found=bool(chunks),
        embedding_latency_ms=1.0,
        retrieval_latency_ms=2.0,
        latency_ms=3.0,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )


def _graph_response(chunks: list[RetrievedChunk]) -> GraphRetrievalResponse:
    entities = [
        GraphEntity(name="Chunking", paper_id=chunk.paper_id, chunk_id=chunk.chunk_id) for chunk in chunks
    ]
    relationships = [
        GraphRelationship(
            source="Chunking",
            relation="MENTIONED_IN",
            target=chunk.paper_id,
            paper_id=chunk.paper_id,
            chunk_id=chunk.chunk_id,
        )
        for chunk in chunks
    ]
    paths = [
        GraphPath(
            path=f"Chunking -[MENTIONED_IN]-> {chunk.paper_id}",
            source="Chunking",
            relation="MENTIONED_IN",
            target=chunk.paper_id,
            paper_id=chunk.paper_id,
            chunk_id=chunk.chunk_id,
        )
        for chunk in chunks
    ]
    return GraphRetrievalResponse(
        question="What method is discussed?",
        entities=entities,
        relationships=relationships,
        paths=paths,
        results=chunks,
        evidence_found=bool(chunks),
        latency_ms=4.0,
    )


def _patch_retrieval(monkeypatch: pytest.MonkeyPatch, vector_chunks, graph_chunks, calls: dict) -> None:
    def vector_retrieve(self, question, paper_ids, top_k):
        calls["vector"] = {"question": question, "paper_ids": list(paper_ids), "top_k": top_k}
        return _vector_response(vector_chunks)

    def graph_retrieve(self, question, paper_ids, top_k, path_depth):
        calls["graph"] = {
            "question": question,
            "paper_ids": list(paper_ids),
            "top_k": top_k,
            "path_depth": path_depth,
        }
        return _graph_response(graph_chunks)

    monkeypatch.setattr("app.rag.hybrid_rag.service.VectorRAGService.retrieve", vector_retrieve)
    monkeypatch.setattr("app.rag.hybrid_rag.service.GraphRAGService.retrieve", graph_retrieve)


def test_fusion_deduplicates_and_marks_origin() -> None:
    shared = _chunk("P001_C001", "P001", "Semantic chunking groups sentences.", score=0.9)
    vector_only = _chunk("P002_C001", "P002", "Vectors compare embeddings.", score=0.4, page=4)
    graph_only = _chunk("P003_C001", "P003", "Graphs store entities.", score=None, page=5)
    combined = fuse_evidence([shared, vector_only], [shared, graph_only])
    by_id = {item.chunk_id: item for item in combined}
    assert list(by_id) == ["P001_C001", "P002_C001", "P003_C001"]
    assert by_id["P001_C001"].origin == "both"
    assert by_id["P001_C001"].score == 0.9
    assert by_id["P001_C001"].paper_id == "P001"
    assert by_id["P002_C001"].origin == "vector"
    assert by_id["P003_C001"].origin == "graph"
    assert by_id["P003_C001"].source_filename == "P003.pdf"


def test_hybrid_query_fuses_evidence_and_attributes_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict = {}
    shared = _chunk("P001_C001", "P001", "Only this retrieved sentence should reach the model.", score=0.8)
    graph_only = _chunk("P002_C004", "P002", "The second paper uses a graph index.", page=4)
    _patch_retrieval(monkeypatch, [shared], [shared, graph_only], calls)
    prompts: list[str] = []

    def generate(settings, prompt: str) -> str:
        prompts.append(prompt)
        from app.rag.vector_rag.service import _generation_model

        _generation_model.set("development-fallback:extractive")
        return "Both papers are cited [P001_C001] [P002_C004]."

    monkeypatch.setattr("app.rag.hybrid_rag.service._generate", generate)
    response = HybridRAGService(_settings(tmp_path)).query(
        "How do the approaches differ?",
        ["P001", "P002", "P003"],
        5,
        2,
    )
    assert calls["vector"]["paper_ids"] == ["P001", "P002", "P003"]
    assert calls["graph"]["paper_ids"] == ["P001", "P002", "P003"]
    assert calls["graph"]["path_depth"] == 2
    assert response.retrieval_method == "HybridRAG"
    assert response.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert response.llm_model == "development-fallback:extractive"
    assert [item.chunk_id for item in response.combined_evidence] == ["P001_C001", "P002_C004"]
    assert response.combined_evidence[0].origin == "both"
    assert response.combined_evidence[1].origin == "graph"
    assert response.sources[0].paper_id == "P001"
    assert response.sources[1].paper_id == "P002"
    assert response.sources[1].page == 4
    assert response.entities[0].paper_id == "P001"
    assert response.relationships
    assert "Do not invent graph relationships." in prompts[0]
    assert "Only this retrieved sentence" in prompts[0]
    assert "second paper uses a graph index" in prompts[0]
    assert response.vector_retrieval_latency_ms == 3.0
    assert response.graph_retrieval_latency_ms == 4.0
    assert response.generation_latency_ms >= 0


def test_hybrid_insufficient_evidence_does_not_generate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"generated": False}
    _patch_retrieval(monkeypatch, [], [], {})
    monkeypatch.setattr(
        "app.rag.hybrid_rag.service._generate",
        lambda settings, prompt: calls.__setitem__("generated", True),
    )
    response = HybridRAGService(_settings(tmp_path)).query("Unknown topic", ["P001"], 5, 2)
    assert calls["generated"] is False
    assert response.answer == NO_EVIDENCE_ANSWER
    assert response.sources == []
    assert response.combined_evidence == []
    assert response.generation_latency_ms == 0


def test_local_fallback_answers_without_openai(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = _chunk("P001_C001", "P001", "Semantic chunking groups sentences.", score=0.5)
    _patch_retrieval(monkeypatch, [chunk], [], {})

    def unavailable(settings, prompt: str) -> str:
        raise RuntimeError("local model is not cached")

    monkeypatch.setattr("app.rag.vector_rag.service._generate_local", unavailable)
    response = HybridRAGService(_settings(tmp_path, llm_provider="local")).query(
        "What method is discussed?",
        ["P001"],
        5,
        2,
    )
    assert "not equivalent to gpt-4o-mini" in response.answer
    assert "Semantic chunking groups sentences." in response.answer
    assert response.llm_model == "development-fallback:extractive"
    assert response.combined_evidence[0].origin == "vector"


def test_hybrid_prompt_forbids_invented_relationships() -> None:
    evidence = fuse_evidence([_chunk("P001_C001", "P001", "Chunk text.", score=0.2)], [])
    prompt = build_hybrid_prompt("What method is discussed?", evidence, [], [])
    assert "Do not invent facts." in prompt
    assert "Do not invent citations." in prompt
    assert "Do not invent graph relationships." in prompt
    assert "explicitly say that the retrieved evidence is insufficient" in prompt


def test_hybrid_api_validation(tmp_path: Path) -> None:
    app.dependency_overrides[get_settings] = lambda: _settings(tmp_path)
    response = TestClient(app).post(
        "/api/rag/hybrid/query",
        json={"question": "", "paper_ids": ["P001"], "top_k": 5, "path_depth": 2},
    )
    app.dependency_overrides.clear()
    assert response.status_code == 422
