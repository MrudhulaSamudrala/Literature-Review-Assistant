"""VectorRAG index, retrieval, and grounded query endpoints."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.config import Settings, get_settings
from app.core.exceptions import VectorRAGError
from app.rag.vector_rag.models import (
    IndexResult,
    IndexStatus,
    VectorQueryResponse,
    VectorRetrievalRequest,
    VectorRetrievalResponse,
)
from app.rag.vector_rag.service import VectorRAGService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/rag/vector", tags=["vector-rag"])


def get_vector_service(settings: Settings = Depends(get_settings)) -> VectorRAGService:
    return VectorRAGService(settings)


@router.post("/index/{paper_id}", response_model=IndexResult)
def index_paper(
    paper_id: str,
    service: VectorRAGService = Depends(get_vector_service),
) -> IndexResult:
    return _call(lambda: service.index_paper(paper_id))


@router.get("/index/{paper_id}", response_model=IndexStatus)
def index_status(
    paper_id: str,
    service: VectorRAGService = Depends(get_vector_service),
) -> IndexStatus:
    return _call(lambda: service.index_status(paper_id))


@router.post("/retrieve", response_model=VectorRetrievalResponse)
def retrieve(
    payload: VectorRetrievalRequest,
    service: VectorRAGService = Depends(get_vector_service),
) -> VectorRetrievalResponse:
    return _call(
        lambda: service.retrieve(payload.question, payload.paper_ids, payload.top_k)
    )


@router.post("/query", response_model=VectorQueryResponse)
def query(
    payload: VectorRetrievalRequest,
    service: VectorRAGService = Depends(get_vector_service),
) -> VectorQueryResponse:
    return _call(lambda: service.query(payload.question, payload.paper_ids, payload.top_k))


def _call(action):
    try:
        return action()
    except VectorRAGError as exc:
        logger.warning("VectorRAG request rejected status=%s detail=%s", exc.status_code, exc)
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception as exc:
        safe_error = _safe_exception_text(exc)
        logger.exception(
            "VectorRAG request failed error_type=%s error=%s",
            type(exc).__name__,
            safe_error,
        )
        detail = "VectorRAG request failed."
        if get_settings().app_env.strip().lower() == "development":
            detail = f"VectorRAG request failed: {type(exc).__name__}: {safe_error}"
        raise HTTPException(status_code=500, detail=detail) from exc


def _safe_exception_text(exc: Exception) -> str:
    settings = get_settings()
    text = str(exc).splitlines()[0][:500]
    for secret in (settings.openai_api_key, settings.neo4j_password):
        if secret:
            text = text.replace(secret, "[redacted]")
    return text or type(exc).__name__
