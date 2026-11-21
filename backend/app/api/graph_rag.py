"""GraphRAG index, retrieval, and grounded query endpoints."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.config import Settings, get_settings
from app.core.exceptions import GraphRAGError
from app.rag.graph_rag.models import (
    GraphIndexResult,
    GraphQueryResponse,
    GraphRetrievalRequest,
    GraphRetrievalResponse,
)
from app.rag.graph_rag.service import GraphRAGService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/rag/graph", tags=["graph-rag"])


def get_graph_service(settings: Settings = Depends(get_settings)) -> GraphRAGService:
    return GraphRAGService(settings)


@router.post("/index/{paper_id}", response_model=GraphIndexResult)
def index_paper(paper_id: str, service: GraphRAGService = Depends(get_graph_service)) -> GraphIndexResult:
    return _call(lambda: service.index_paper(paper_id))


@router.post("/retrieve", response_model=GraphRetrievalResponse)
def retrieve(
    payload: GraphRetrievalRequest,
    service: GraphRAGService = Depends(get_graph_service),
) -> GraphRetrievalResponse:
    return _call(
        lambda: service.retrieve(payload.question, payload.paper_ids, payload.top_k, payload.path_depth)
    )


@router.post("/query", response_model=GraphQueryResponse)
def query(
    payload: GraphRetrievalRequest,
    service: GraphRAGService = Depends(get_graph_service),
) -> GraphQueryResponse:
    return _call(
        lambda: service.query(payload.question, payload.paper_ids, payload.top_k, payload.path_depth)
    )


def _call(action):
    try:
        return action()
    except GraphRAGError as exc:
        logger.warning("GraphRAG request rejected status=%s detail=%s", exc.status_code, exc)
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception as exc:
        safe_error = _safe_exception_text(exc)
        logger.exception(
            "GraphRAG request failed error_type=%s error=%s",
            type(exc).__name__,
            safe_error,
        )
        detail = "GraphRAG request failed."
        if get_settings().app_env.strip().lower() == "development":
            detail = f"GraphRAG request failed: {type(exc).__name__}: {safe_error}"
        raise HTTPException(status_code=500, detail=detail) from exc


def _safe_exception_text(exc: Exception) -> str:
    settings = get_settings()
    text = str(exc).splitlines()[0][:500]
    for secret in (settings.openai_api_key, settings.neo4j_password):
        if secret:
            text = text.replace(secret, "[redacted]")
    return text or type(exc).__name__
