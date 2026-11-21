"""FastAPI entrypoint for the Smart Literature Review Assistant."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import documents, graph_rag, health, hybrid_rag, papers, query, vector_rag
from app.config import get_settings
from app.core.logging import configure_logging
from app.rag.vector_rag.indexer import log_embedding_configuration
from app.rag.vector_rag.service import log_llm_configuration
from app.utils.files import ensure_directory


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings)
    log_embedding_configuration(settings)
    log_llm_configuration(settings)
    ensure_directory(settings.papers_dir)
    ensure_directory(settings.processed_dir)
    ensure_directory(settings.chunks_dir)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        description=(
            "Web-based Smart Literature Review Assistant comparing VectorRAG, "
            "GraphRAG, and HybridRAG with LlamaIndex and Neo4j."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(papers.router)
    app.include_router(vector_rag.router)
    app.include_router(graph_rag.router)
    app.include_router(hybrid_rag.router)
    app.include_router(query.router)
    return app


app = create_app()
