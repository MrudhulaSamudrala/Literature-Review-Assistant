"""Document upload endpoints. Ingestion logic is deferred to a later step."""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.config import Settings, get_settings
from app.models.schemas import DocumentUploadResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
) -> DocumentUploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="A filename is required.")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    settings.papers_dir.mkdir(parents=True, exist_ok=True)
    destination = settings.papers_dir / Path(file.filename).name
    content = await file.read()
    destination.write_bytes(content)
    logger.info("Stored uploaded paper at %s", destination)

    return DocumentUploadResponse(
        filename=destination.name,
        stored_path=str(destination),
        message="PDF stored. Ingestion and indexing will be implemented in a later step.",
        implemented=False,
    )
