import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from llama_index.embeddings.openai import OpenAIEmbedding
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.core.exceptions import IngestionError
from app.ingestion.models import ChunkRecord, PaperMetadata
from app.ingestion.persistence import allocate_paper_path
from app.ingestion.pipeline import IngestionService
from app.ingestion.semantic_chunker import (
    BREAKPOINT_PERCENTILE_THRESHOLD,
    BUFFER_SIZE,
    REFERENCE_EMBEDDING_MODEL,
    DeterministicTextEmbedding,
    SemanticChunker,
)
from app.main import app
from tests.pdf_samples import pdf_bytes

PAPER_TEXT = [
    "Introduction\nSemantic chunking groups related sentences together. Graphs store entities and relations.",
    "Methods\nVector search compares embeddings. Hybrid retrieval combines vector search with graph traversal.",
]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        papers_dir=tmp_path / "papers",
        processed_dir=tmp_path / "processed",
        chunks_dir=tmp_path / "chunks",
        openai_api_key="",
    )


@pytest.fixture
def client(settings: Settings):
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health_still_ok(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "smart-literature-review-assistant"


def test_rejects_non_pdf(client: TestClient) -> None:
    response = client.post(
        "/api/papers/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF files are accepted."
    assert "Traceback" not in response.text


def test_rejects_empty_pdf(client: TestClient) -> None:
    response = client.post(
        "/api/papers/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_rejects_corrupted_pdf(client: TestClient, settings: Settings) -> None:
    response = client.post(
        "/api/papers/upload",
        files={"file": ("broken.pdf", b"%PDF-1.4\nnot-a-real-pdf", "application/pdf")},
    )
    assert response.status_code == 422
    assert "readable" in response.json()["detail"].lower()
    assert "Traceback" not in response.text
    assert list(settings.papers_dir.glob("*.pdf")) == []


def test_paper_metadata_validation() -> None:
    paper = PaperMetadata(
        paper_id="P001",
        filename="example.pdf",
        title=None,
        authors=None,
        year=None,
        page_count=2,
        source_path="data/papers/P001.pdf",
    )
    assert paper.title is None
    with pytest.raises(ValidationError):
        PaperMetadata(
            paper_id="paper-1",
            filename="example.pdf",
            page_count=0,
            source_path="data/papers/P001.pdf",
        )


def test_paper_id_generation_does_not_overwrite(settings: Settings) -> None:
    settings.papers_dir.mkdir()
    (settings.papers_dir / "P001.pdf").write_bytes(b"%PDF")
    (settings.papers_dir / "P003.pdf").write_bytes(b"%PDF")
    paper_id, path = allocate_paper_path(settings, b"%PDF-1.4 sample")
    assert paper_id == "P004"
    assert path.name == "P004.pdf"
    assert (settings.papers_dir / "P001.pdf").read_bytes() == b"%PDF"


def test_semantic_chunking_configuration() -> None:
    chunker = SemanticChunker(embed_model=DeterministicTextEmbedding())
    assert chunker.parser.buffer_size == BUFFER_SIZE == 1
    assert chunker.parser.breakpoint_percentile_threshold == BREAKPOINT_PERCENTILE_THRESHOLD == 95


def test_configured_openai_model_is_used_for_semantic_splits() -> None:
    settings = Settings(
        embedding_provider="openai",
        openai_api_key="sk-test",
        openai_embedding_model=REFERENCE_EMBEDDING_MODEL,
        evaluation_mode=False,
    )
    chunker = SemanticChunker(settings=settings)
    assert isinstance(chunker.parser.embed_model, OpenAIEmbedding)
    assert chunker.parser.embed_model.model_name == "text-embedding-ada-002"
    assert chunker.parser.embed_model._text_engine == "text-embedding-ada-002"
    assert chunker.embedding_status.mode == "openai"
    assert chunker.embedding_status.development_fallback is False
    assert chunker.embedding_status.reference_equivalent is True
    assert chunker.embedding_status.vectors_persisted is False

    other = Settings(
        embedding_provider="openai",
        openai_api_key="sk-test",
        openai_embedding_model="text-embedding-3-small",
    )
    other_chunker = SemanticChunker(settings=other)
    assert other_chunker.parser.embed_model.model_name == "text-embedding-3-small"
    assert other_chunker.embedding_status.reference_equivalent is False


def test_local_semantic_chunking_uses_huggingface(caplog: pytest.LogCaptureFixture) -> None:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    settings = Settings(
        embedding_provider="local",
        local_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        openai_api_key="",
        evaluation_mode=False,
    )
    with caplog.at_level(logging.INFO):
        chunker = SemanticChunker(settings=settings)
    assert isinstance(chunker.parser.embed_model, HuggingFaceEmbedding)
    assert chunker.embedding_status.mode == "local"
    assert chunker.embedding_status.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert chunker.embedding_status.development_fallback is False
    assert "Semantic chunking is using local embedding model sentence-transformers/all-MiniLM-L6-v2" in caplog.text


def test_development_fallback_is_explicit(caplog: pytest.LogCaptureFixture) -> None:
    settings = Settings(
        embedding_provider="openai",
        openai_api_key="",
        evaluation_mode=False,
        app_env="development",
    )
    with caplog.at_level(logging.WARNING):
        chunker = SemanticChunker(settings=settings)
    assert isinstance(chunker.embed_model, DeterministicTextEmbedding)
    assert chunker.embed_model.model_name == "development-fallback:deterministic"
    assert chunker.embedding_status.mode == "development_fallback"
    assert chunker.embedding_status.development_fallback is True
    assert chunker.embedding_status.reference_equivalent is False
    assert "DEVELOPMENT FALLBACK" in caplog.text
    assert "not equivalent to the reference-paper setup" in caplog.text
    assert "text-embedding-ada-002" in caplog.text


def test_evaluation_run_refuses_fallback(caplog: pytest.LogCaptureFixture) -> None:
    settings = Settings(
        embedding_provider="openai",
        openai_api_key="",
        evaluation_mode=True,
        app_env="development",
    )
    with caplog.at_level(logging.ERROR):
        with pytest.raises(IngestionError, match="development fallback"):
            SemanticChunker(settings=settings)
    assert "refused the development fallback" in caplog.text

    labeled = Settings(
        embedding_provider="openai",
        openai_api_key="",
        evaluation_mode=False,
        app_env="evaluation",
    )
    with pytest.raises(IngestionError, match="development fallback"):
        SemanticChunker(settings=labeled)

    with_key = Settings(
        embedding_provider="openai",
        openai_api_key="sk-test",
        openai_embedding_model="text-embedding-ada-002",
        evaluation_mode=True,
    )
    chunker = SemanticChunker(settings=with_key)
    assert isinstance(chunker.parser.embed_model, OpenAIEmbedding)
    assert chunker.embedding_status.model == "text-embedding-ada-002"


def test_chunk_ids_are_deterministic_and_keep_page_metadata(settings: Settings) -> None:
    service = IngestionService(
        settings,
        chunker=SemanticChunker(embed_model=DeterministicTextEmbedding()),
    )
    content = pdf_bytes(PAPER_TEXT, title="Example Research Paper", author="Ada Lovelace")
    first = service.ingest_upload("example.pdf", content)
    second = service.ingest_upload("example.pdf", content)

    assert first.paper_id == "P001"
    assert second.paper_id == "P002"
    assert first.page_count == 2
    assert first.chunk_count >= 1
    assert first.status == "ingested"
    assert first.title == "Example Research Paper"
    assert first.authors == ["Ada Lovelace"]

    stored = json.loads((settings.chunks_dir / "P001.json").read_text(encoding="utf-8"))
    processed = json.loads((settings.processed_dir / "P001.json").read_text(encoding="utf-8"))
    assert [page["page"] for page in processed["pages"]] == [1, 2]
    assert processed["semantic_chunking"]["buffer_size"] == 1
    assert processed["semantic_chunking"]["breakpoint_percentile_threshold"] == 95
    assert processed["semantic_chunking"]["embedding"] == {
        "mode": "development_fallback",
        "model": "development-fallback:deterministic",
        "development_fallback": True,
        "reference_equivalent": False,
        "vectors_persisted": False,
    }
    assert "embedding" not in stored["chunks"][0]
    assert "vector" not in stored["chunks"][0]

    chunks = [ChunkRecord.model_validate(item) for item in stored["chunks"]]
    assert [chunk.chunk_id for chunk in chunks] == [
        f"P001_C{index:03d}" for index in range(1, len(chunks) + 1)
    ]
    assert len(chunks) == first.chunk_count
    pages = {page["page"]: page["text"] for page in processed["pages"]}
    for chunk in chunks:
        assert chunk.paper_id == "P001"
        assert chunk.source_filename == "example.pdf"
        assert chunk.page in pages
        assert chunk.text in pages[chunk.page]
        assert chunk.title == "Example Research Paper"

    repeated = service.ingest_upload("example.pdf", content)
    repeated_chunks = json.loads(
        (settings.chunks_dir / f"{repeated.paper_id}.json").read_text(encoding="utf-8")
    )
    assert [item["text"] for item in repeated_chunks["chunks"]] == [chunk.text for chunk in chunks]
    assert [item["chunk_id"] for item in repeated_chunks["chunks"]] == [
        chunk.chunk_id.replace("P001", repeated.paper_id) for chunk in chunks
    ]


def test_upload_api_lists_paper_and_chunks(client: TestClient, settings: Settings) -> None:
    content = pdf_bytes(PAPER_TEXT)
    response = client.post(
        "/api/papers/upload",
        files={"file": ("example.pdf", content, "application/pdf")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["paper_id"] == "P001"
    assert body["filename"] == "example.pdf"
    assert body["status"] == "ingested"
    assert body["chunk_count"] >= 1
    assert (settings.papers_dir / "P001.pdf").is_file()
    assert (settings.processed_dir / "P001.json").is_file()
    assert (settings.chunks_dir / "P001.json").is_file()

    listed = client.get("/api/papers")
    assert listed.status_code == 200
    assert listed.json()["papers"][0]["paper_id"] == "P001"

    detail = client.get("/api/papers/P001")
    assert detail.status_code == 200
    assert detail.json()["page_count"] == 2

    chunks = client.get("/api/papers/P001/chunks?offset=0&limit=1")
    assert chunks.status_code == 200
    payload = chunks.json()
    assert payload["total"] == body["chunk_count"]
    assert payload["limit"] == 1
    assert len(payload["chunks"]) == 1
    assert payload["chunks"][0]["chunk_id"] == "P001_C001"

    missing = client.get("/api/papers/P999")
    assert missing.status_code == 404
