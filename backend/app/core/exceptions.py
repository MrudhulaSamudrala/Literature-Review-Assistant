"""Shared application exceptions."""


class LiteratureReviewError(Exception):
    """Base exception for the Smart Literature Review Assistant."""


class PipelineNotImplementedError(LiteratureReviewError):
    """Raised when a RAG pipeline has not been implemented yet."""


class VectorRAGError(LiteratureReviewError):
    """Raised when VectorRAG indexing or retrieval cannot continue."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class HybridRAGError(LiteratureReviewError):
    """Raised when HybridRAG cannot combine the existing retrieval services."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class GraphRAGError(LiteratureReviewError):
    """Raised when GraphRAG indexing or retrieval cannot continue."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class IngestionError(LiteratureReviewError):
    """Raised when a PDF cannot be ingested. The message is safe to return to clients."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code
