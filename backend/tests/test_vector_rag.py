from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.embeddings.openai import OpenAIEmbedding

from app.config import Settings, get_settings
from app.core.exceptions import VectorRAGError
from app.ingestion.models import ChunkRecord
from app.main import app
from app.rag.vector_rag.indexer import (
    REQUIRED_EMBEDDING_MODEL,
    build_embedding_model,
    chunks_needing_embedding,
    chunks_to_nodes,
    connect_vector_store,
    embed_and_store,
    embedding_dimension,
    log_embedding_configuration,
    require_openai_embeddings,
    require_vector_embeddings,
)
from app.rag.vector_rag.models import RetrievedChunk
from app.rag.vector_rag.prompts import NO_EVIDENCE_ANSWER, build_grounded_prompt
from app.rag.vector_rag.retriever import citations_from, retrieve_chunks
from app.api.vector_rag import _call
from app.rag.vector_rag.service import (
    DEVELOPMENT_FALLBACK_LLM,
    VectorRAGService,
    _generate,
    _generate_local,
    _local_generators,
    log_llm_configuration,
)


def _chunk(chunk_id: str, paper_id: str, text: str, page: int = 1) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        paper_id=paper_id,
        title="Example Research Paper" if paper_id == "P001" else "Second Paper",
        section="Methodology",
        page=page,
        source_filename=f"{paper_id}.pdf",
        text=text,
    )


def _settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "papers_dir": tmp_path / "papers",
        "processed_dir": tmp_path / "processed",
        "chunks_dir": tmp_path / "chunks",
        "openai_api_key": "sk-test",
        "openai_embedding_model": "text-embedding-ada-002",
        "openai_llm_model": "gpt-4o-mini",
        "llm_provider": "openai",
        "embedding_provider": "openai",
        "neo4j_password": "secret",
        "vector_top_k": 5,
    }
    values.update(overrides)
    return Settings(**values)


def test_embedding_model_must_be_ada_002() -> None:
    settings = _settings(Path("."))
    model = build_embedding_model(settings)
    assert isinstance(model, OpenAIEmbedding)
    assert model.model_name == REQUIRED_EMBEDDING_MODEL == "text-embedding-ada-002"
    assert type(model).__name__ != "DeterministicTextEmbedding"

    missing = _settings(Path("."), openai_api_key="")
    with pytest.raises(VectorRAGError, match="development embedding fallback"):
        require_openai_embeddings(missing)

    wrong_model = _settings(Path("."), openai_embedding_model="text-embedding-3-small")
    with pytest.raises(VectorRAGError, match="text-embedding-ada-002"):
        require_openai_embeddings(wrong_model)


def test_nodes_preserve_chunk_metadata() -> None:
    nodes = chunks_to_nodes([_chunk("P001_C001", "P001", "Semantic chunking groups sentences.", page=3)])
    assert nodes[0].node_id == "P001_C001"
    assert nodes[0].metadata["paper_id"] == "P001"
    assert nodes[0].metadata["chunk_id"] == "P001_C001"
    assert nodes[0].metadata["title"] == "Example Research Paper"
    assert nodes[0].metadata["section"] == "Methodology"
    assert nodes[0].metadata["page"] == 3
    assert nodes[0].metadata["source_filename"] == "P001.pdf"
    assert nodes[0].get_content() == "Semantic chunking groups sentences."


def test_duplicate_indexing_skips_unchanged_text() -> None:
    first = _chunk("P001_C001", "P001", "unchanged")
    second = _chunk("P001_C002", "P001", "new text", page=2)
    pending = chunks_needing_embedding({"P001_C001": "unchanged"}, [first, second])
    assert [chunk.chunk_id for chunk in pending] == ["P001_C002"]


def test_index_embeds_only_new_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = [
        _chunk("P001_C001", "P001", "unchanged"),
        _chunk("P001_C002", "P001", "new text", page=2),
    ]
    captured: dict = {}

    monkeypatch.setattr("app.rag.vector_rag.service.load_paper_chunks", lambda settings, paper_id: chunks)
    monkeypatch.setattr(
        "app.rag.vector_rag.service.connect_vector_store",
        lambda settings, embedding_dimension: object(),
    )
    monkeypatch.setattr(
        "app.rag.vector_rag.service.read_indexed_chunks",
        lambda store, settings, paper_id: {"P001_C001": "unchanged", "P001_C009": "stale"},
    )
    monkeypatch.setattr(
        "app.rag.vector_rag.service.delete_chunk_ids",
        lambda store, settings, chunk_ids: captured.setdefault("deleted", chunk_ids),
    )

    def capture_store(store, embed_model, nodes):
        captured["model"] = embed_model.model_name
        captured["ids"] = [node.node_id for node in nodes]
        captured["metadata"] = nodes[0].metadata

    monkeypatch.setattr("app.rag.vector_rag.service.embed_and_store", capture_store)
    result = VectorRAGService(_settings(tmp_path)).index_paper("P001")
    assert result.status == "indexed"
    assert result.chunk_count == 2
    assert result.embedded_chunk_count == 1
    assert result.embedding_model == "text-embedding-ada-002"
    assert captured["deleted"] == ["P001_C009"]
    assert captured["ids"] == ["P001_C002"]
    assert captured["model"] == "text-embedding-ada-002"
    assert captured["metadata"]["page"] == 2


def test_missing_paper_is_rejected_before_neo4j(tmp_path: Path) -> None:
    with pytest.raises(VectorRAGError, match="not found") as raised:
        VectorRAGService(_settings(tmp_path, neo4j_password="")).index_paper("P001")
    assert raised.value.status_code == 404


def test_embed_and_store_uses_vector_store_index(monkeypatch: pytest.MonkeyPatch) -> None:
    inserted: dict = {}

    class FakeIndex:
        def insert_nodes(self, nodes):
            inserted["nodes"] = list(nodes)

    monkeypatch.setattr(
        "app.rag.vector_rag.indexer.VectorStoreIndex.from_vector_store",
        lambda store, embed_model: FakeIndex(),
    )
    node = chunks_to_nodes([_chunk("P001_C001", "P001", "stored text")])[0]
    embed_and_store(object(), object(), [node])
    assert inserted["nodes"][0].metadata["chunk_id"] == "P001_C001"


def test_retrieve_uses_configured_top_k_for_one_or_many_papers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    class FakeEmbed:
        model_name = "text-embedding-ada-002"

        def get_query_embedding(self, text: str):
            captured["question"] = text
            return [0.2, 0.4]

    monkeypatch.setattr("app.rag.vector_rag.service.build_embedding_model", lambda settings: FakeEmbed())
    monkeypatch.setattr(
        "app.rag.vector_rag.service.connect_vector_store",
        lambda settings, embedding_dimension: "neo4j-store",
    )

    def fake_retrieve(store, embed_model, question, query_embedding, paper_ids, top_k):
        captured.update(
            store=store,
            paper_ids=list(paper_ids),
            top_k=top_k,
            embedding=query_embedding,
        )
        return [
            RetrievedChunk(
                rank=1,
                score=0.91,
                chunk_id="P001_C007",
                paper_id="P001",
                title="Example Research Paper",
                page=3,
                section="Methodology",
                source_filename="P001.pdf",
                text="Vector search compares embeddings.",
            )
        ]

    monkeypatch.setattr("app.rag.vector_rag.service.retrieve_chunks", fake_retrieve)
    service = VectorRAGService(_settings(tmp_path, vector_top_k=4))
    single = service.retrieve("What is the contribution?", ["P001"], None)
    assert captured["top_k"] == 4
    assert captured["paper_ids"] == ["P001"]
    assert captured["store"] == "neo4j-store"
    assert single.results[0].score == 0.91
    assert single.results[0].chunk_id == "P001_C007"
    assert single.embedding_model == "text-embedding-ada-002"
    assert single.embedding_latency_ms >= 0
    assert single.retrieval_latency_ms >= 0

    service.retrieve("Compare the papers", ["P001", "P002", "P003"], 2)
    assert captured["paper_ids"] == ["P001", "P002", "P003"]
    assert captured["top_k"] == 2


def test_empty_retrieval_does_not_generate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.rag.vector_rag.service.build_embedding_model",
        lambda settings: type("Embed", (), {"get_query_embedding": lambda self, text: [0.1]})(),
    )
    monkeypatch.setattr(
        "app.rag.vector_rag.service.connect_vector_store",
        lambda settings, embedding_dimension: object(),
    )
    monkeypatch.setattr(
        "app.rag.vector_rag.service.retrieve_chunks",
        lambda *args, **kwargs: [],
    )
    generated = {"called": False}
    monkeypatch.setattr(
        "app.rag.vector_rag.service._generate",
        lambda settings, prompt: generated.__setitem__("called", True),
    )
    response = VectorRAGService(_settings(tmp_path)).query("Unknown topic", ["P001"], 5)
    assert generated["called"] is False
    assert response.answer == NO_EVIDENCE_ANSWER
    assert response.sources == []
    assert response.retrieved_chunks == []
    assert response.generation_latency_ms == 0
    assert response.llm_model == "gpt-4o-mini"
    assert response.total_latency_ms >= response.retrieval_latency_ms


def test_query_sends_only_retrieved_evidence_and_citations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kept = RetrievedChunk(
        rank=1,
        score=0.8,
        chunk_id="P001_C007",
        paper_id="P001",
        title="Example Research Paper",
        page=3,
        section="Methodology",
        source_filename="example.pdf",
        text="Only this retrieved sentence should reach the model.",
    )
    monkeypatch.setattr(
        "app.rag.vector_rag.service.build_embedding_model",
        lambda settings: type("Embed", (), {"get_query_embedding": lambda self, text: [0.1]})(),
    )
    monkeypatch.setattr(
        "app.rag.vector_rag.service.connect_vector_store",
        lambda settings, embedding_dimension: object(),
    )
    monkeypatch.setattr(
        "app.rag.vector_rag.service.retrieve_chunks",
        lambda *args, **kwargs: [kept],
    )
    prompts: list[str] = []

    def generate(settings, prompt: str) -> str:
        prompts.append(prompt)
        assert settings.openai_llm_model == "gpt-4o-mini"
        return "The paper compares vector search [P001_C007]."

    monkeypatch.setattr("app.rag.vector_rag.service._generate", generate)
    response = VectorRAGService(_settings(tmp_path)).query("What does the paper compare?", ["P001"], 5)
    assert "Only this retrieved sentence" in prompts[0]
    assert "Do not invent citations." in prompts[0]
    assert "unretrieved sentence" not in prompts[0]
    assert response.sources[0].chunk_id == "P001_C007"
    assert response.sources[0].page == 3
    assert response.sources[0].paper_id == "P001"
    assert response.sources[0].source_filename == "example.pdf"
    assert response.embedding_model == "text-embedding-ada-002"
    assert response.llm_model == "gpt-4o-mini"
    assert response.retrieval_latency_ms >= 0
    assert response.generation_latency_ms >= 0
    assert response.total_latency_ms >= 0


def test_grounded_prompt_and_citation_shape() -> None:
    chunk = RetrievedChunk(
        rank=1,
        score=0.5,
        chunk_id="P002_C003",
        paper_id="P002",
        title="Second Paper",
        page=4,
        section=None,
        source_filename="second.pdf",
        text="Graphs store entities.",
    )
    prompt = build_grounded_prompt("How are papers combined?", [chunk])
    assert "chunk_id=P002_C003" in prompt
    assert "paper_id=P002" in prompt
    assert "page=4" in prompt
    assert "source_filename=second.pdf" in prompt
    citation = citations_from([chunk])[0]
    assert citation.model_dump() == {
        "paper_id": "P002",
        "title": "Second Paper",
        "page": 4,
        "chunk_id": "P002_C003",
        "source_filename": "second.pdf",
    }


def test_retriever_keeps_real_score_and_metadata() -> None:
    node = TextNode(
        id_="P001_C007",
        text="Retrieved text",
        metadata={
            "paper_id": "P001",
            "chunk_id": "P001_C007",
            "title": "Example Research Paper",
            "section": "Methodology",
            "page": 3,
            "source_filename": "example.pdf",
        },
    )
    captured = {}

    class FakeRetriever:
        def __init__(self, index, similarity_top_k, filters, embed_model):
            captured["top_k"] = similarity_top_k
            captured["papers"] = filters.filters[0].value

        def retrieve(self, bundle):
            captured["embedding"] = bundle.embedding
            return [NodeWithScore(node=node, score=0.77)]

    class FakeIndex:
        @staticmethod
        def from_vector_store(store, embed_model):
            return object()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("app.rag.vector_rag.retriever.VectorStoreIndex", FakeIndex)
    monkeypatch.setattr("app.rag.vector_rag.retriever.VectorIndexRetriever", FakeRetriever)
    results = retrieve_chunks(object(), object(), "question", [0.3], ["P001", "P002"], 5)
    monkeypatch.undo()
    assert captured["top_k"] == 5
    assert captured["papers"] == ["P001", "P002"]
    assert captured["embedding"] == [0.3]
    assert results[0].score == 0.77
    assert results[0].page == 3
    assert results[0].chunk_id == "P001_C007"


def test_vector_api_validation(tmp_path: Path) -> None:
    settings = _settings(tmp_path, openai_api_key="")
    app.dependency_overrides[get_settings] = lambda: settings
    client = TestClient(app)
    missing_key = client.post("/api/rag/vector/index/P001")
    assert missing_key.status_code == 400
    assert "development embedding fallback" in missing_key.json()["detail"]

    app.dependency_overrides[get_settings] = lambda: _settings(tmp_path)
    invalid = TestClient(app).post(
        "/api/rag/vector/retrieve",
        json={"question": "", "paper_ids": ["P001"]},
    )
    assert invalid.status_code == 422
    app.dependency_overrides.clear()


class _LocalEmbed:
    def __init__(self) -> None:
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.embed_dim = 384

    def get_query_embedding(self, text: str) -> list[float]:
        return [0.1] * self.embed_dim


def test_local_provider_does_not_call_openai_embeddings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    created: dict = {}

    def reject_openai(*args, **kwargs):
        raise AssertionError("OpenAIEmbedding must not be constructed for the local provider")

    def local_model(model_name: str) -> _LocalEmbed:
        created["model_name"] = model_name
        return _LocalEmbed()

    monkeypatch.setattr("app.rag.vector_rag.indexer.OpenAIEmbedding", reject_openai)
    monkeypatch.setattr("app.rag.vector_rag.indexer._build_local_embedding", local_model)
    settings = _settings(
        tmp_path,
        openai_api_key="",
        embedding_provider="local",
        vector_embedding_dimension=1536,
    )
    with caplog.at_level("INFO"):
        model = build_embedding_model(settings)
        log_embedding_configuration(settings)
    assert created["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert model.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert embedding_dimension(settings, model) == 384
    assert "embedding provider = local" in caplog.text
    assert "embedding model = sentence-transformers/all-MiniLM-L6-v2" in caplog.text
    assert "sk-" not in caplog.text


def test_local_index_uses_model_dimension_and_separate_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    class FakeStore:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.embedding_dimension = kwargs["embedding_dimension"]
            self.index_name = kwargs["index_name"]
            self.node_label = kwargs["node_label"]

    monkeypatch.setattr("app.rag.vector_rag.indexer.Neo4jVectorStore", FakeStore)
    monkeypatch.setattr("app.rag.vector_rag.indexer._build_local_embedding", lambda model_name: _LocalEmbed())
    settings = _settings(tmp_path, openai_api_key="", embedding_provider="local")
    service = VectorRAGService(settings)
    monkeypatch.setattr(
        "app.rag.vector_rag.service.load_paper_chunks",
        lambda current_settings, paper_id: [_chunk("P001_C001", "P001", "local text")],
    )
    monkeypatch.setattr(
        "app.rag.vector_rag.service.read_indexed_chunks",
        lambda store, current_settings, paper_id: {},
    )
    monkeypatch.setattr(
        "app.rag.vector_rag.service.delete_chunk_ids",
        lambda store, current_settings, chunk_ids: None,
    )
    monkeypatch.setattr(
        "app.rag.vector_rag.service.embed_and_store",
        lambda store, embed_model, nodes: captured.setdefault("embedded", embed_model.model_name),
    )
    result = service.index_paper("P001")
    assert result.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert captured["embedding_dimension"] == 384
    assert captured["index_name"] == "literature_vector_dev"
    assert captured["node_label"] == "LiteratureChunkDev"
    assert captured["embedded"] == "sentence-transformers/all-MiniLM-L6-v2"


def test_local_dimension_uses_sentence_transformer_not_openai_size(tmp_path: Path) -> None:
    class SentenceModel:
        def get_sentence_embedding_dimension(self) -> int:
            return 384

    class Wrapper:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        _model = SentenceModel()

    settings = _settings(tmp_path, embedding_provider="local", vector_embedding_dimension=1536)
    assert embedding_dimension(settings, Wrapper()) == 384


def test_local_dimension_uses_sentence_transformer_not_openai_size(tmp_path: Path) -> None:
    class SentenceModel:
        def get_sentence_embedding_dimension(self) -> int:
            return 384

    class Wrapper:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        _model = SentenceModel()

    settings = _settings(tmp_path, embedding_provider="local", vector_embedding_dimension=1536)
    assert embedding_dimension(settings, Wrapper()) == 384


def test_incompatible_neo4j_index_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class ExistingStore:
        def __init__(self, **kwargs):
            self.embedding_dimension = 1536
            self.index_name = kwargs["index_name"]
            self.node_label = kwargs["node_label"]

    monkeypatch.setattr("app.rag.vector_rag.indexer.Neo4jVectorStore", ExistingStore)
    settings = _settings(tmp_path, embedding_provider="local")
    with pytest.raises(VectorRAGError, match="Refusing to reuse an incompatible index") as raised:
        connect_vector_store(settings, 384)
    assert raised.value.status_code == 409


def test_openai_provider_keeps_ada_002(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_local(model_name: str):
        raise AssertionError("local embedding must not replace text-embedding-ada-002")

    monkeypatch.setattr("app.rag.vector_rag.indexer._build_local_embedding", reject_local)
    settings = _settings(tmp_path, embedding_provider="openai", openai_embedding_model="text-embedding-ada-002")
    model = build_embedding_model(settings)
    assert isinstance(model, OpenAIEmbedding)
    assert model.model_name == "text-embedding-ada-002"
    assert embedding_dimension(settings, model) == 1536


def test_evaluation_refuses_local_embeddings(tmp_path: Path) -> None:
    settings = _settings(tmp_path, embedding_provider="local", evaluation_mode=True, openai_api_key="")
    with pytest.raises(VectorRAGError, match="development only"):
        require_vector_embeddings(settings)


def test_local_index_endpoint_does_not_require_openai_embeddings(tmp_path: Path) -> None:
    settings = _settings(tmp_path, openai_api_key="", embedding_provider="local")
    app.dependency_overrides[get_settings] = lambda: settings
    response = TestClient(app).post("/api/rag/vector/index/P001")
    app.dependency_overrides.clear()
    assert response.status_code == 404
    assert "OPENAI_API_KEY" not in response.json()["detail"]


def _stub_retrieval(monkeypatch: pytest.MonkeyPatch, text: str = "Only this retrieved sentence should reach the model.") -> None:
    kept = RetrievedChunk(
        rank=1,
        score=0.8,
        chunk_id="P001_C007",
        paper_id="P001",
        title="Example Research Paper",
        page=3,
        section="Methodology",
        source_filename="example.pdf",
        text=text,
    )
    monkeypatch.setattr(
        "app.rag.vector_rag.service.build_embedding_model",
        lambda settings: type("Embed", (), {"embed_dim": 384, "get_query_embedding": lambda self, value: [0.1]})(),
    )
    monkeypatch.setattr(
        "app.rag.vector_rag.service.connect_vector_store",
        lambda settings, embedding_dimension: object(),
    )
    monkeypatch.setattr(
        "app.rag.vector_rag.service.retrieve_chunks",
        lambda *args, **kwargs: [kept],
    )


def test_local_llm_fallback_answers_from_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_openai(*args, **kwargs):
        raise AssertionError("OpenAI must not be called when LLM_PROVIDER=local")

    def unavailable(settings, prompt: str) -> str:
        raise RuntimeError("local model is not cached")

    monkeypatch.setattr("app.rag.vector_rag.service.OpenAI", reject_openai)
    monkeypatch.setattr("app.rag.vector_rag.service._generate_local", unavailable)
    _stub_retrieval(monkeypatch)
    settings = _settings(
        tmp_path,
        llm_provider="local",
        embedding_provider="local",
        openai_api_key="",
    )
    response = VectorRAGService(settings).query("What does the paper compare?", ["P001"], 5)
    assert "not equivalent to gpt-4o-mini" in response.answer
    assert "Only this retrieved sentence" in response.answer
    assert response.llm_model == DEVELOPMENT_FALLBACK_LLM
    assert response.llm_model != "gpt-4o-mini"
    assert response.sources[0].chunk_id == "P001_C007"
    assert response.retrieval_method == "VectorRAG"


def test_local_llm_uses_huggingface_pipeline_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def local_answer(settings, prompt: str) -> str:
        captured["model"] = settings.local_llm_model
        captured["prompt"] = prompt
        return "The paper compares vector search [P001_C007]."

    monkeypatch.setattr("app.rag.vector_rag.service._generate_local", local_answer)
    _stub_retrieval(monkeypatch)
    settings = _settings(tmp_path, llm_provider="local", embedding_provider="local", openai_api_key="")
    response = VectorRAGService(settings).query("What does the paper compare?", ["P001"], 5)
    assert response.answer.startswith("The paper compares")
    assert response.llm_model == "google/flan-t5-small"
    assert "Only this retrieved sentence" in captured["prompt"]
    assert captured["model"] == "google/flan-t5-small"


def test_generate_local_does_not_duplicate_local_files_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pipeline() already supplies local_files_only; model_kwargs must not repeat it."""

    captured: dict = {}

    def fake_pipeline(*args, **kwargs):
        model_kwargs = kwargs.get("model_kwargs") or {}
        if "local_files_only" in model_kwargs:
            raise TypeError(
                "AutoConfig.from_pretrained() got multiple values for keyword argument 'local_files_only'"
            )
        captured["args"] = args
        captured["kwargs"] = kwargs

        def generate(prompt, max_new_tokens, do_sample):
            captured["prompt"] = prompt
            captured["max_new_tokens"] = max_new_tokens
            captured["do_sample"] = do_sample
            return [{"generated_text": "A local Flan answer."}]

        return generate

    import transformers

    monkeypatch.setattr(transformers, "pipeline", fake_pipeline)
    monkeypatch.setattr("transformers.pipeline", fake_pipeline)
    _local_generators.clear()
    settings = _settings(
        tmp_path,
        llm_provider="local",
        local_llm_model="google/flan-t5-small",
        embedding_provider="local",
        openai_api_key="",
    )
    try:
        text = _generate_local(settings, "Question with evidence")
    finally:
        _local_generators.clear()
    assert text == "A local Flan answer."
    assert captured["args"][0] == "text2text-generation"
    assert captured["kwargs"]["model"] == "google/flan-t5-small"
    assert captured["kwargs"]["tokenizer"] == "google/flan-t5-small"
    assert captured["kwargs"]["local_files_only"] is True
    assert "local_files_only" not in (captured["kwargs"].get("model_kwargs") or {})
    assert captured["max_new_tokens"] == 256
    assert captured["do_sample"] is False


def test_openai_llm_provider_keeps_gpt_4o_mini(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCompletion:
        text = "Reference answer."

    class FakeOpenAI:
        def __init__(self, model, api_key, temperature):
            assert model == "gpt-4o-mini"
            assert api_key == "sk-test"
            assert temperature == 0

        def complete(self, prompt):
            return FakeCompletion()

    monkeypatch.setattr("app.rag.vector_rag.service.OpenAI", FakeOpenAI)
    settings = _settings(tmp_path, llm_provider="openai")
    assert _generate(settings, "Answer from evidence.") == "Reference answer."


def test_evaluation_refuses_local_llm(tmp_path: Path) -> None:
    settings = _settings(tmp_path, llm_provider="local", evaluation_mode=True)
    with pytest.raises(VectorRAGError, match="development only"):
        _generate(settings, "prompt")


def test_llm_startup_log_has_no_secrets(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    settings = _settings(tmp_path, llm_provider="local", openai_api_key="sk-test", neo4j_password="secret")
    with caplog.at_level("INFO"):
        log_llm_configuration(settings)
    assert "llm_provider = local" in caplog.text
    assert "llm_model = google/flan-t5-small" in caplog.text
    assert "sk-test" not in caplog.text
    assert "secret" not in caplog.text


def test_development_errors_include_the_logged_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = _settings(tmp_path, app_env="development", openai_api_key="sk-test")
    monkeypatch.setattr("app.api.vector_rag.get_settings", lambda: settings)

    def boom():
        raise RuntimeError("quota sk-test exceeded")

    with caplog.at_level("ERROR"):
        with pytest.raises(HTTPException) as raised:
            _call(boom)
    assert "error_type=RuntimeError" in caplog.text
    assert "quota [redacted] exceeded" in caplog.text
    assert raised.value.status_code == 500
    assert "RuntimeError" in raised.value.detail
    assert "sk-test" not in raised.value.detail
