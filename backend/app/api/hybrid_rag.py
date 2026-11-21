"""HybridRAG query endpoint."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.config import Settings, get_settings
from app.core.exceptions import GraphRAGError, HybridRAGError, VectorRAGError
from app.rag.hybrid_rag.models import HybridQueryRequest, HybridQueryResponse
from app.rag.hybrid_rag.service import HybridRAGService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/rag/hybrid", tags=["hybrid-rag"])


def get_hybrid_service(settings: Settings = Depends(get_settings)) -> HybridRAGService:
    return HybridRAGService(settings)


@router.post("/query", response_model=HybridQueryResponse)
def query(
    payload: HybridQueryRequest,
    service: HybridRAGService = Depends(get_hybrid_service),
) -> HybridQueryResponse:
    return _call(
        lambda: service.query(payload.question, payload.paper_ids, payload.top_k, payload.path_depth)
    )


def _call(action):
    try:
        return action()
    except (HybridRAGError, VectorRAGError, GraphRAGError) as exc:
        logger.warning("HybridRAG request rejected status=%s detail=%s", exc.status_code, exc)
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception as exc:
        safe_error = _safe_exception_text(exc)
        logger.exception(
            "HybridRAG request failed error_type=%s error=%s",
            type(exc).__name__,
            safe_error,
        )
        detail = "HybridRAG request failed."
        if get_settings().app_env.strip().lower() == "development":
            detail = f"HybridRAG request failed: {type(exc).__name__}: {safe_error}"
        raise HTTPException(status_code=500, detail=detail) from exc


def _safe_exception_text(exc: Exception) -> str:
    settings = get_settings()
    text = str(exc).splitlines()[0][:500]
    for secret in (settings.openai_api_key, settings.neo4j_password):
        if secret:
            text = text.replace(secret, "[redacted]")
    return text or type(exc).__name__
