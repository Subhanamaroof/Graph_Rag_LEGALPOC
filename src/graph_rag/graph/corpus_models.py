"""Schemas for the full-corpus graph build (Story 1.6.2)."""

from typing import Literal

from pydantic import BaseModel, Field

JobStatus = Literal[
    "pending",  # created or reset for retry, not running yet
    "running",
    "paused",  # stopped on request (or at shutdown) between chunks; resumable
    "interrupted",  # stopped by an error or a process restart; resumable
    "completed",  # every chunk extracted and persisted
    "completed_with_failures",  # finished, but some chunks failed; see /failures and /retry
]
DocumentStatus = Literal["pending", "processing", "completed", "partially_completed", "failed"]
ChunkStatus = Literal["pending", "completed", "failed"]
FailureStage = Literal["load", "extraction", "graph"]


class BuildOptions(BaseModel):
    """Effective processing options of a job (request values override settings)."""

    batch_size: int = Field(ge=1, le=100)
    delay_seconds: float = Field(ge=0, le=60)
    max_retries: int = Field(ge=0, le=10)
    retry_delay_seconds: float = Field(ge=0, le=300)
    max_consecutive_failures: int = Field(ge=1)


class BuildAllRequest(BaseModel):
    batch_size: int | None = Field(default=None, ge=1, le=100)
    delay_seconds: float | None = Field(default=None, ge=0, le=60)
    max_retries: int | None = Field(default=None, ge=0, le=10)
    resume: bool = True  # continue the latest job instead of starting from scratch


class DiscoverySummary(BaseModel):
    total_chunk_records: int
    total_documents: int
    records_missing_document_id: int
    invalid_record_ids: list[str] = Field(default_factory=list)  # first few only
    documents_eligible: int
    chunks_eligible: int


class DryRunResponse(BaseModel):
    """What a start request would do. Nothing is written and Groq is not called."""

    discovery: DiscoverySummary
    resumable_job_id: str | None  # latest job, which ``resume: true`` would continue
    resumable_job_status: JobStatus | None
    chunks_already_completed: int  # in that job; skipped on resume
    chunks_to_process: int  # with ``resume: true``
    chunks_to_process_without_resume: int


class JobActionResponse(BaseModel):
    job_id: str
    status: JobStatus
    resumed: bool = False
    chunks_to_process: int
    message: str
    discovery: DiscoverySummary | None = None


class JobStatusResponse(BaseModel):
    """Counts are aggregated from the per-chunk records in the job database, not estimated.

    ``chunks_attempted`` >= ``chunks_extracted`` >= ``chunks_processed``: a chunk is
    *processed* only when its extraction succeeded AND its graph write committed.
    """

    job_id: str
    status: JobStatus
    error: str | None
    stop_requested: bool
    options: BuildOptions
    records_missing_document_id: int
    total_documents: int
    documents_completed: int
    documents_partially_completed: int
    documents_failed: int
    documents_pending: int  # pending or processing
    total_chunks: int
    chunks_attempted: int
    chunks_extracted: int
    chunks_processed: int
    chunks_failed: int
    chunks_pending: int
    entities_extracted: int
    relationships_extracted: int
    nodes_created: int
    nodes_merged: int
    relationships_created: int
    relationships_merged: int
    documents_progress: str  # "completed/total"
    chunks_progress: str  # "processed/total"
    progress_percentage: float  # processed chunks / total chunks * 100
    created_at: str
    started_at: str | None
    updated_at: str
    completed_at: str | None


class ChunkFailure(BaseModel):
    chunk_id: str
    stage: FailureStage
    error: str
    attempts: int


class DocumentFailure(BaseModel):
    document_id: str
    status: DocumentStatus
    total_chunks: int
    chunks_processed: int
    chunks_failed: int
    error: str | None
    failed_chunks: list[ChunkFailure]


class FailuresResponse(BaseModel):
    job_id: str
    total_failed_documents: int
    offset: int
    limit: int
    documents: list[DocumentFailure]
