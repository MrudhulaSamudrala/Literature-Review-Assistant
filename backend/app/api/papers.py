"""Paper upload and ingestion inspection endpoints."""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from app.config import Settings, get_settings
from app.core.exceptions import IngestionError
from app.ingestion import persistence
from app.ingestion.models import ChunkListResponse, IngestionResult, PaperListResponse, PaperSummary
from app.ingestion.pipeline import IngestionService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/papers", tags=["papers"])


def get_ingestion_service(settings: Settings = Depends(get_settings)) -> IngestionService:
    try:
        return IngestionService(settings)
    except IngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.post("/upload", response_model=IngestionResult)
async def upload_paper(
    file: UploadFile = File(...),
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestionResult:
    content = await file.read()
    try:
        return service.ingest_upload(file.filename, content)
    except IngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Ingestion failed for filename=%s", file.filename)
        raise HTTPException(status_code=500, detail="Ingestion failed.") from exc


@router.get("", response_model=PaperListResponse)
def list_papers(settings: Settings = Depends(get_settings)) -> PaperListResponse:
    try:
        return PaperListResponse(papers=persistence.list_papers(settings))
    except IngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.get("/{paper_id}", response_model=PaperSummary)
def get_paper(paper_id: str, settings: Settings = Depends(get_settings)) -> PaperSummary:
    try:
        return persistence.load_paper(settings, paper_id)
    except IngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.get("/{paper_id}/chunks", response_model=ChunkListResponse)
def get_chunks(
    paper_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    settings: Settings = Depends(get_settings),
) -> ChunkListResponse:
    try:
        chunks = persistence.load_chunks(settings, paper_id)
    except IngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    page = chunks[offset : offset + limit]
    return ChunkListResponse(
        paper_id=paper_id,
        total=len(chunks),
        offset=offset,
        limit=limit,
        chunks=page,
    )
