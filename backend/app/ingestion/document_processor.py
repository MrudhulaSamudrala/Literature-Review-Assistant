"""Turn extracted pages into LlamaIndex documents and normalized metadata."""

import logging

from llama_index.core import Document

from app.ingestion.models import PageText, PaperMetadata

logger = logging.getLogger(__name__)

_KNOWN_SECTIONS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "methods",
    "methodology",
    "materials and methods",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "references",
    "acknowledgements",
    "acknowledgments",
}


def build_paper_metadata(
    *,
    paper_id: str,
    filename: str,
    source_path: str,
    page_count: int,
    title: str | None,
    authors: list[str] | None,
    year: int | None,
) -> PaperMetadata:
    return PaperMetadata(
        paper_id=paper_id,
        filename=filename,
        title=title,
        authors=authors,
        year=year,
        page_count=page_count,
        source_path=source_path,
    )


def pages_to_documents(paper: PaperMetadata, pages: list[PageText]) -> list[Document]:
    """Create one LlamaIndex document per page so page numbers stay attached."""

    documents: list[Document] = []
    for page in pages:
        text = page.text.strip()
        if not text:
            continue
        documents.append(
            Document(
                text=page.text,
                metadata={
                    "paper_id": paper.paper_id,
                    "page": page.page,
                    "source_filename": page.source_filename,
                    "title": paper.title,
                    "section": infer_section(page.text),
                },
            )
        )
    logger.info(
        "LlamaIndex documents created paper_id=%s documents=%s",
        paper.paper_id,
        len(documents),
    )
    return documents


def infer_section(text: str) -> str | None:
    """Use a heading only when the text starts with a recognizable section label."""

    for line in text.splitlines():
        candidate = line.strip().strip(":").strip()
        if not candidate:
            continue
        if candidate.lower() in _KNOWN_SECTIONS:
            return candidate
        return None
    return None
