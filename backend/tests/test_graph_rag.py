import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from llama_index.core.graph_stores.types import Relation
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor

from app.config import Settings, get_settings
from app.core.exceptions import GraphRAGError
from app.ingestion.models import ChunkRecord
from app.main import app
from app.rag.graph_rag.extractor import (
    DEVELOPMENT_GRAPH_EXTRACTOR,
    MAX_PATHS_PER_CHUNK,
    build_graph_extractor,
    deterministic_triples,
)
from app.rag.graph_rag.service import (
    GraphRAGService,
    filter_graph_rows,
    question_keywords,
)
from app.rag.graph_rag.store import ApoCFreeNeo4jPropertyGraphStore, relation_type
from app.rag.vector_rag.prompts import NO_EVIDENCE_ANSWER
from app.rag.vector_rag.service import _generation_model


def _settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "papers_dir": tmp_path / "papers",
        "processed_dir": tmp_path / "processed",
        "chunks_dir": tmp_path / "chunks",
        "openai_api_key": "sk-test",
        "openai_embedding_model": "text-embedding-ada-002",
        "openai_llm_model": "gpt-4o-mini",
        "llm_provider": "local",
        "embedding_provider": "local",
        "neo4j_password": "secret",
        "vector_top_k": 5,
    }
    values.update(overrides)
    return Settings(**values)


def _chunk(chunk_id: str, paper_id: str, text: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        paper_id=paper_id,
        title="Example Research Paper" if paper_id == "P001" else "Second Paper",
        section="Methodology",
        page=3 if paper_id == "P001" else 4,
        source_filename=f"{paper_id}.pdf",
        text=text,
    )


def _store_chunks(settings: Settings, chunks: list[ChunkRecord]) -> None:
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    settings.chunks_dir.mkdir(parents=True, exist_ok=True)
    for chunk in chunks:
        (settings.processed_dir / f"{chunk.paper_id}.json").write_text("{}", encoding="utf-8")
        path = settings.chunks_dir / f"{chunk.paper_id}.json"
        existing = []
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))["chunks"]
        existing.append(chunk.model_dump())
        path.write_text(json.dumps({"paper_id": chunk.paper_id, "chunks": existing}), encoding="utf-8")


class _Store:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.closed = False

    def delete_paper_relations(self, paper_id: str) -> None:
        self.deleted.append(paper_id)

    def close(self) -> None:
        self.closed = True

    def structured_query(self, query: str, param_map=None):
        return []


def test_deterministic_extractor_is_stable_and_traceable(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    extractor, provider, model_name = build_graph_extractor(settings)
    assert provider == "local"
    assert model_name == DEVELOPMENT_GRAPH_EXTRACTOR
    metadata = {
        "paper_id": "P001",
        "chunk_id": "P001_C001",
        "title": "Example Research Paper",
        "section": "Methodology",
        "page": 3,
        "source_filename": "P001.pdf",
        "original_chunk_text": "Semantic chunking groups related sentences.",
    }
    first = deterministic_triples(metadata["original_chunk_text"], metadata)
    second = deterministic_triples(metadata["original_chunk_text"], metadata)
    assert first == second
    assert len(first) <= MAX_PATHS_PER_CHUNK
    from llama_index.core.schema import TextNode

    nodes = extractor([TextNode(id_="P001_C001", text=metadata["original_chunk_text"], metadata=metadata)])
    relation = nodes[0].metadata["relations"][0]
    assert relation.properties["chunk_id"] == "P001_C001"
    assert relation.properties["paper_id"] == "P001"
    assert relation.properties["page"] == 3
    assert relation.properties["original_chunk_text"] == metadata["original_chunk_text"]
    assert relation.properties["extraction"] == DEVELOPMENT_GRAPH_EXTRACTOR


def test_openai_graph_extractor_keeps_gpt_4o_mini(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_local(*args, **kwargs):
        raise AssertionError("local extractor must not replace gpt-4o-mini")

    monkeypatch.setattr(
        "app.rag.graph_rag.extractor.DeterministicPathExtractor",
        reject_local,
    )
    extractor, provider, model_name = build_graph_extractor(
        _settings(tmp_path, llm_provider="openai", embedding_provider="openai")
    )
    assert provider == "openai"
    assert model_name == "gpt-4o-mini"
    assert isinstance(extractor, SimpleLLMPathExtractor)
    assert extractor.max_paths_per_chunk == 10
    assert extractor.llm.model == "gpt-4o-mini"


def test_evaluation_refuses_local_graph_extractor(tmp_path: Path) -> None:
    settings = _settings(tmp_path, evaluation_mode=True, openai_api_key="")
    with pytest.raises(GraphRAGError, match="development graph extractor"):
        build_graph_extractor(settings)


def test_relation_upsert_uses_merge_without_apoc() -> None:
    store = ApoCFreeNeo4jPropertyGraphStore.__new__(ApoCFreeNeo4jPropertyGraphStore)
    queries: list[str] = []

    def record(query: str, param_map=None):
        queries.append(query)
        return []

    store.structured_query = record
    store.upsert_relations(
        [
            Relation(
                label="mentioned in",
                source_id="Semantic",
                target_id="P001",
                properties={"paper_id": "P001", "chunk_id": "P001_C001"},
            )
        ]
    )
    assert relation_type("mentioned in") == "mentioned_in"
    assert "MERGE" in queries[0]
    assert "apoc" not in queries[0].lower()


def test_graph_index_is_idempotent_and_logs_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = _settings(tmp_path, openai_api_key="")
    _store_chunks(settings, [_chunk("P001_C001", "P001", "Semantic chunking groups related sentences.")])
    store = _Store()
    captured: dict = {}
    monkeypatch.setattr("app.rag.graph_rag.service.connect_graph_store", lambda current: store)
    monkeypatch.setattr("app.rag.graph_rag.service.build_embedding_model", lambda current: object())
    monkeypatch.setattr(
        "app.rag.graph_rag.service.PropertyGraphIndex",
        lambda **kwargs: captured.setdefault("calls", []).append(kwargs["nodes"][0].metadata["chunk_id"]),
    )
    service = GraphRAGService(settings)
    with caplog.at_level("INFO"):
        first = service.index_paper("P001")
        second = service.index_paper("P001")
    assert first.status == "indexed"
    assert first.chunk_count == 1
    assert first.entity_count > 0
    assert first.relationship_count > 0
    assert first.llm_provider == "local"
    assert first.llm_model == DEVELOPMENT_GRAPH_EXTRACTOR
    assert second.entity_count == first.entity_count
    assert store.deleted == ["P001", "P001"]
    assert captured["calls"] == ["P001_C001", "P001_C001"]
    assert "graph llm_provider = local" in caplog.text
    assert "graph llm_model = development-fallback:graph-extract" in caplog.text
    assert "not equivalent to gpt-4o-mini" in caplog.text
    assert "sk-" not in caplog.text
    assert store.closed is True


def test_missing_paper_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(GraphRAGError, match="not found") as raised:
        GraphRAGService(_settings(tmp_path, neo4j_password="")).index_paper("P001")
    assert raised.value.status_code == 404


def test_multi_paper_retrieval_keeps_source_attribution() -> None:
    rows = [
        {
            "source": "Chunking",
            "relation": "MENTIONED_IN",
            "target": "P001",
            "paper_id": "P001",
            "chunk_id": "P001_C001",
            "title": "Example Research Paper",
            "page": 3,
            "section": "Methodology",
            "source_filename": "P001.pdf",
            "text": "Semantic chunking groups sentences.",
        },
        {
            "source": "Graphs",
            "relation": "MENTIONED_IN",
            "target": "P002",
            "paper_id": "P002",
            "chunk_id": "P002_C001",
            "title": "Second Paper",
            "page": 4,
            "section": "Methodology",
            "source_filename": "P002.pdf",
            "text": "Graphs store entities.",
        },
        {
            "source": "Unrelated",
            "relation": "MENTIONED_IN",
            "target": "P009",
            "paper_id": "P009",
            "chunk_id": "P009_C001",
            "title": "Other",
            "page": 1,
            "section": "Intro",
            "source_filename": "P009.pdf",
            "text": "Should stay outside the selected papers.",
        },
    ]
    keywords = question_keywords("How do chunking and graphs differ?")
    matched = filter_graph_rows(rows, ["P001", "P002", "P003"], keywords, limit=5)
    papers = {row["paper_id"] for row in matched}
    assert papers == {"P001", "P002"}
    assert all(row["chunk_id"].startswith(row["paper_id"]) for row in matched)


def test_graph_retrieve_and_empty_query(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    rows = [
        {
            "source": "Chunking",
            "relation": "MENTIONED_IN",
            "target": "P001",
            "paper_id": "P001",
            "chunk_id": "P001_C001",
            "title": "Example Research Paper",
            "page": 3,
            "section": "Methodology",
            "source_filename": "P001.pdf",
            "text": "Semantic chunking groups sentences.",
        }
    ]
    monkeypatch.setattr("app.rag.graph_rag.service.connect_graph_store", lambda current: _Store())
    monkeypatch.setattr("app.rag.graph_rag.service._run_reference_retrievers", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.rag.graph_rag.service.fetch_graph_rows", lambda *args, **kwargs: rows)
    service = GraphRAGService(settings)
    found = service.retrieve("How does chunking work?", ["P001"], 5, 2)
    assert found.retrieval_method == "GraphRAG"
    assert found.evidence_found is True
    assert found.results[0].chunk_id == "P001_C001"
    assert found.results[0].page == 3
    assert found.results[0].text == "Semantic chunking groups sentences."
    assert found.paths[0].path == "Chunking -[MENTIONED_IN]-> P001"
    assert found.entities[0].paper_id == "P001"

    monkeypatch.setattr("app.rag.graph_rag.service.fetch_graph_rows", lambda *args, **kwargs: [])
    generated = {"called": False}
    monkeypatch.setattr(
        "app.rag.graph_rag.service._generate",
        lambda settings, prompt: generated.__setitem__("called", True),
    )
    empty = service.query("Unknown topic", ["P001"], 5, 2)
    assert generated["called"] is False
    assert empty.answer == NO_EVIDENCE_ANSWER
    assert empty.sources == []
    assert empty.retrieved_chunks == []
    assert empty.generation_latency_ms == 0
    assert empty.retrieval_method == "GraphRAG"


def test_graph_query_uses_only_retrieved_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path, llm_provider="local")
    rows = [
        {
            "source": "Chunking",
            "relation": "MENTIONED_IN",
            "target": "P001",
            "paper_id": "P001",
            "chunk_id": "P001_C001",
            "title": "Example Research Paper",
            "page": 3,
            "section": "Methodology",
            "source_filename": "P001.pdf",
            "text": "Only this retrieved sentence should reach the model.",
        }
    ]
    prompts: list[str] = []
    monkeypatch.setattr("app.rag.graph_rag.service.connect_graph_store", lambda current: _Store())
    monkeypatch.setattr("app.rag.graph_rag.service._run_reference_retrievers", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.rag.graph_rag.service.fetch_graph_rows", lambda *args, **kwargs: rows)

    def generate(current, prompt: str) -> str:
        prompts.append(prompt)
        _generation_model.set("development-fallback:extractive")
        return "The paper discusses chunking [P001_C001]."

    monkeypatch.setattr("app.rag.graph_rag.service._generate", generate)
    response = GraphRAGService(settings).query("How does chunking work?", ["P001", "P002"], 5, 2)
    assert "Only this retrieved sentence" in prompts[0]
    assert "Do not invent citations." in prompts[0]
    assert response.sources[0].chunk_id == "P001_C001"
    assert response.sources[0].page == 3
    assert response.llm_model == "development-fallback:extractive"
    assert response.retrieval_method == "GraphRAG"
    assert response.entities


def test_graph_api_missing_paper_and_validation(tmp_path: Path) -> None:
    settings = _settings(tmp_path, neo4j_password="")
    app.dependency_overrides[get_settings] = lambda: settings
    client = TestClient(app)
    missing = client.post("/api/rag/graph/index/P001")
    assert missing.status_code == 404
    invalid = client.post(
        "/api/rag/graph/retrieve",
        json={"question": "", "paper_ids": ["P001"], "path_depth": 2},
    )
    assert invalid.status_code == 422
    app.dependency_overrides.clear()
