"""Orchestrate PDF upload, page extraction, semantic chunking, and persistence."""

import logging
import time

from app.config import Settings
from app.core.exceptions import IngestionError
from app.ingestion import persistence
from app.ingestion.document_processor import build_paper_metadata, pages_to_documents
from app.ingestion.models import IngestionResult, ProcessedPaper, SemanticChunkingConfig
from app.ingestion.pdf_ingestor import extract_pages, read_pdf_metadata, validate_pdf_upload
from app.ingestion.semantic_chunker import BUFFER_SIZE, BREAKPOINT_PERCENTILE_THRESHOLD, SemanticChunker

logger = logging.getLogger(__name__)


class IngestionService:
    """Upload a paper and persist page text plus semantic chunks."""

    def __init__(self, settings: Settings, chunker: SemanticChunker | None = None) -> None:
        self.settings = settings
        self.chunker = chunker or SemanticChunker(settings=settings)

    def ingest_upload(self, filename: str | None, content: bytes) -> IngestionResult:
        started = time.perf_counter()
        logger.info("PDF upload started filename=%s", filename)
        safe_name = validate_pdf_upload(filename, content)
        paper_id, source_path = persistence.allocate_paper_path(self.settings, content)
        try:
            pages = extract_pages(paper_id, safe_name, content)
            title, authors, year = read_pdf_metadata(content)
            paper = build_paper_metadata(
                paper_id=paper_id,
                filename=safe_name,
                source_path=str(source_path),
                page_count=len(pages),
                title=title,
                authors=authors,
                year=year,
            )
            documents = pages_to_documents(paper, pages)
            chunks = self.chunker.chunk_documents(documents, paper)
            processed = ProcessedPaper(
                **paper.model_dump(),
                status="ingested",
                chunk_count=len(chunks),
                breakpoint_embed_model=self.chunker.embed_model_name,
                semantic_chunking=SemanticChunkingConfig(
                    buffer_size=BUFFER_SIZE,
                    breakpoint_percentile_threshold=BREAKPOINT_PERCENTILE_THRESHOLD,
                    embedding=self.chunker.embedding_status,
                ),
                pages=pages,
            )
            persistence.write_processed_paper(self.settings, processed)
            persistence.write_chunks(self.settings, paper_id, chunks)
        except Exception:
            persistence.remove_paper_files(self.settings, paper_id)
            raise

        elapsed = round(time.perf_counter() - started, 3)
        logger.info(
            "Ingestion completed paper_id=%s pages=%s chunks=%s seconds=%s",
            paper_id,
            len(pages),
            len(chunks),
            elapsed,
        )
        return IngestionResult(
            paper_id=paper_id,
            filename=safe_name,
            status="ingested",
            title=title,
            authors=authors,
            year=year,
            page_count=len(pages),
            chunk_count=len(chunks),
            processing_time_seconds=elapsed,
            source_path=str(source_path),
        )


def get_ingestion_service(settings: Settings) -> IngestionService:
    return IngestionService(settings)
