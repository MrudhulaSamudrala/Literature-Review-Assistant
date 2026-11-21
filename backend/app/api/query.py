"""Question-answering route. Generation is not implemented in Stage 1."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.core.exceptions import PipelineNotImplementedError
from app.models.schemas import QueryRequest, QueryResponse
from app.services.rag_service import RAGService, get_rag_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/query", tags=["query"])


@router.post("", response_model=QueryResponse)
def ask_question(
    payload: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> QueryResponse:
    logger.info("Received query for unimplemented pipeline %s", payload.mode.value)
    try:
        return rag_service.query(payload)
    except PipelineNotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
