"""Extract page text from research-paper PDFs."""

import logging
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import EmptyFileError, PdfReadError, PyPdfError

from app.core.exceptions import IngestionError
from app.ingestion.models import PageText

logger = logging.getLogger(__name__)


def validate_pdf_upload(filename: str | None, content: bytes) -> str:
    """Reject uploads that are not non-empty PDF files. Returns the base filename."""

    logger.info("PDF validation filename=%s", filename)
    if not filename or not filename.strip():
        raise IngestionError("A filename is required.", status_code=400)
    safe_name = Path(filename).name
    if not safe_name.lower().endswith(".pdf"):
        raise IngestionError("Only PDF files are accepted.", status_code=400)
    if not content:
        raise IngestionError("The uploaded PDF is empty.", status_code=400)
    if not content.lstrip().startswith(b"%PDF"):
        raise IngestionError("The file is not a readable PDF.", status_code=422)
    _open_reader(content)
    return safe_name


def extract_pages(paper_id: str, source_filename: str, content: bytes) -> list[PageText]:
    """Extract text from each page without flattening the document."""

    logger.info("Text extraction started paper_id=%s", paper_id)
    reader = _open_reader(content)
    pages: list[PageText] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except PyPdfError as exc:
            logger.exception("Text extraction failed paper_id=%s page=%s", paper_id, index)
            raise IngestionError(
                "Text could not be extracted from the PDF.",
                status_code=422,
            ) from exc
        pages.append(
            PageText(
                paper_id=paper_id,
                page=index,
                source_filename=source_filename,
                text=text,
            )
        )
    if not any(page.text.strip() for page in pages):
        raise IngestionError("The PDF contains no extractable text.", status_code=422)
    logger.info("Pages extracted paper_id=%s page_count=%s", paper_id, len(pages))
    return pages


def read_pdf_metadata(content: bytes) -> tuple[str | None, list[str] | None, int | None]:
    """Read title, authors, and year only when the PDF metadata states them."""

    reader = _open_reader(content)
    info = reader.metadata
    if info is None:
        return None, None, None
    title = _clean_text(getattr(info, "title", None))
    authors = _parse_authors(getattr(info, "author", None))
    year = _read_year(info)
    return title, authors, year


def _open_reader(content: bytes) -> PdfReader:
    try:
        reader = PdfReader(BytesIO(content))
    except EmptyFileError as exc:
        raise IngestionError("The uploaded PDF is empty.", status_code=400) from exc
    except PdfReadError as exc:
        logger.exception("Rejected an unreadable PDF")
        raise IngestionError("The file is not a readable PDF.", status_code=422) from exc
    except PyPdfError as exc:
        logger.exception("Rejected a PDF that could not be parsed")
        raise IngestionError("The file is not a readable PDF.", status_code=422) from exc
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            logger.exception("Rejected an encrypted PDF")
            raise IngestionError("The PDF is encrypted and cannot be read.", status_code=422) from exc
    try:
        page_count = len(reader.pages)
    except PdfReadError as exc:
        logger.exception("Rejected a PDF whose pages could not be read")
        raise IngestionError("The file is not a readable PDF.", status_code=422) from exc
    if page_count < 1:
        raise IngestionError("The PDF has no pages.", status_code=422)
    return reader


def _clean_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _parse_authors(value: object) -> list[str] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    import re

    parts = re.split(r"\s*(?:,|;|\band\b)\s*", value.strip())
    authors = [part.strip() for part in parts if part.strip()]
    return authors or None


def _read_year(info: object) -> int | None:
    raw = None
    if hasattr(info, "get"):
        raw = info.get("/Year")  # type: ignore[union-attr]
    if not isinstance(raw, str) or not raw.strip().isdigit():
        return None
    year = int(raw.strip())
    if 1000 <= year <= 9999:
        return year
    return None
