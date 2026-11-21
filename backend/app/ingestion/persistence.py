"""Save papers, extracted pages, and semantic chunks as JSON."""

import json
import logging
import os
import re
import threading
from pathlib import Path

from app.config import Settings
from app.core.exceptions import IngestionError
from app.ingestion.models import ChunkRecord, PaperSummary, ProcessedPaper

logger = logging.getLogger(__name__)

_PAPER_ID = re.compile(r"^P(\d+)$")
_LOCK = threading.Lock()


def allocate_paper_path(settings: Settings, content: bytes) -> tuple[str, Path]:
    """Store the original PDF as {paper_id}.pdf without overwriting an existing file."""

    settings.papers_dir.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        number = _next_number(settings)
        while True:
            paper_id = f"P{number:03d}"
            destination = settings.papers_dir / f"{paper_id}.pdf"
            try:
                fd = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                number += 1
                continue
            try:
                os.write(fd, content)
            except OSError as exc:
                logger.exception("Persistence failure while saving PDF paper_id=%s", paper_id)
                raise IngestionError("The PDF could not be saved.", status_code=500) from exc
            finally:
                os.close(fd)
            logger.info("PDF saved paper_id=%s path=%s", paper_id, destination)
            return paper_id, destination


def write_processed_paper(settings: Settings, paper: ProcessedPaper) -> Path:
    path = settings.processed_dir / f"{paper.paper_id}.json"
    _write_json(path, paper.model_dump())
    logger.info("Processed document persisted paper_id=%s path=%s", paper.paper_id, path)
    return path


def write_chunks(settings: Settings, paper_id: str, chunks: list[ChunkRecord]) -> Path:
    path = settings.chunks_dir / f"{paper_id}.json"
    payload = {"paper_id": paper_id, "chunks": [chunk.model_dump() for chunk in chunks]}
    _write_json(path, payload)
    logger.info("Chunks persisted paper_id=%s path=%s count=%s", paper_id, path, len(chunks))
    return path


def list_papers(settings: Settings) -> list[PaperSummary]:
    if not settings.processed_dir.exists():
        return []
    papers: list[PaperSummary] = []
    for path in sorted(settings.processed_dir.glob("P*.json")):
        papers.append(_summary_from_path(path))
    return papers


def load_paper(settings: Settings, paper_id: str) -> PaperSummary:
    path = _processed_path(settings, paper_id)
    return _summary_from_path(path)


def load_chunks(settings: Settings, paper_id: str) -> list[ChunkRecord]:
    _processed_path(settings, paper_id)
    path = settings.chunks_dir / f"{paper_id}.json"
    if not path.is_file():
        raise IngestionError(f"No chunks are stored for {paper_id}.", status_code=404)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [ChunkRecord.model_validate(item) for item in payload["chunks"]]
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.exception("Could not read chunks paper_id=%s", paper_id)
        raise IngestionError("Stored chunks could not be read.", status_code=500) from exc


def remove_paper_files(settings: Settings, paper_id: str) -> None:
    for path in (
        settings.papers_dir / f"{paper_id}.pdf",
        settings.processed_dir / f"{paper_id}.json",
        settings.chunks_dir / f"{paper_id}.json",
    ):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.exception("Could not remove partial ingestion file %s", path)


def _summary_from_path(path: Path) -> PaperSummary:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return PaperSummary.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.exception("Could not read processed paper %s", path)
        raise IngestionError("Stored paper metadata could not be read.", status_code=500) from exc


def _processed_path(settings: Settings, paper_id: str) -> Path:
    if _PAPER_ID.match(paper_id) is None:
        raise IngestionError("Unknown paper id.", status_code=404)
    path = settings.processed_dir / f"{paper_id}.json"
    if not path.is_file():
        raise IngestionError(f"Paper {paper_id} was not found.", status_code=404)
    return path


def _next_number(settings: Settings) -> int:
    highest = 0
    for directory in (settings.papers_dir, settings.processed_dir, settings.chunks_dir):
        if not directory.exists():
            continue
        for path in directory.iterdir():
            match = _PAPER_ID.match(path.stem)
            if match:
                highest = max(highest, int(match.group(1)))
    return highest + 1


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        logger.exception("Persistence failure path=%s", path)
        temporary.unlink(missing_ok=True)
        raise IngestionError("Ingestion output could not be saved.", status_code=500) from exc
