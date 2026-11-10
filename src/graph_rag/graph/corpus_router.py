"""HTTP endpoints for the full-corpus graph build (Story 1.6.2).

Nothing starts automatically: a build begins only with ``POST /v1/graph/build-all``. The
request returns at once (202) with a job ID; a background thread does the work and the job's
progress is read back from the SQLite job store.
"""

from functools import partial
from typing import Annotated

from chromadb.api.models.Collection import Collection
from fastapi import APIRouter, Depends, HTTPException, Query, status

from graph_rag.core.config import Settings, get_settings
from graph_rag.core.llm import JSONCompletionClient
from graph_rag.graph.build_job_store import BuildJobStore
from graph_rag.graph.corpus_builder import BuildRunner, JobAlreadyRunningError, run_job, runner
from graph_rag.graph.corpus_discovery import INVALID_ID_SAMPLE, CorpusDiscovery, discover_corpus
from graph_rag.graph.corpus_models import (
    BuildAllRequest,
    BuildOptions,
    DiscoverySummary,
    DryRunResponse,
    FailuresResponse,
    JobActionResponse,
    JobStatusResponse,
)
from graph_rag.graph.graph_router import get_graph_client
from graph_rag.graph.graph_schema import ensure_schema
from graph_rag.graph.neo4j_client import GraphStoreError, Neo4jClient
from graph_rag.graph.router import get_extraction_llm
from graph_rag.retrieval.router import get_chroma_collection
from graph_rag.vector_store.chroma_store import VectorStoreError

router = APIRouter(prefix="/v1/graph/build-all", tags=["graph"])


def get_job_store(settings: Annotated[Settings, Depends(get_settings)]) -> BuildJobStore:
    return BuildJobStore(settings.graph_build_jobs_db_path)


def get_build_runner() -> BuildRunner:
    return runner


Store = Annotated[BuildJobStore, Depends(get_job_store)]
Runner = Annotated[BuildRunner, Depends(get_build_runner)]


def _discover(collection: Collection) -> CorpusDiscovery:
    try:
        return discover_corpus(collection)
    except VectorStoreError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


def _summary(discovery: CorpusDiscovery) -> DiscoverySummary:
    return DiscoverySummary(
        total_chunk_records=discovery.total_chunk_records,
        total_documents=discovery.total_documents,
        records_missing_document_id=discovery.records_missing_document_id,
        invalid_record_ids=discovery.invalid_record_ids[:INVALID_ID_SAMPLE],
        documents_eligible=discovery.total_documents,
        chunks_eligible=discovery.eligible_chunks,
    )


def _options(request: BuildAllRequest, settings: Settings) -> BuildOptions:
    return BuildOptions(
        batch_size=request.batch_size or settings.graph_build_batch_size,
        delay_seconds=(
            settings.graph_build_delay_seconds
            if request.delay_seconds is None
            else request.delay_seconds
        ),
        max_retries=(
            settings.graph_build_max_retries if request.max_retries is None else request.max_retries
        ),
        retry_delay_seconds=settings.graph_build_retry_delay_seconds,
        max_consecutive_failures=settings.graph_build_max_consecutive_failures,
    )


def _ensure_idle(store: BuildJobStore, build_runner: BuildRunner) -> None:
    active = build_runner.active_job_id() or store.running_job_id()
    if active:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=f"Graph build job '{active}' is already running. Stop it or wait for it.",
        )


def _require_job(store: BuildJobStore, job_id: str) -> None:
    if store.get_job(job_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' was not found.")


def _launch(store, build_runner, job_id, collection, client, llm) -> None:
    work = partial(run_job, store, job_id, collection=collection, client=client, llm=llm)
    store.set_job_status(job_id, "running")  # visible to other requests before the thread runs
    try:
        build_runner.start(job_id, lambda stop: work(stop=stop))
    except JobAlreadyRunningError as exc:
        store.set_job_status(job_id, "interrupted", "Another job was already running.")
        raise HTTPException(
            status.HTTP_409_CONFLICT, detail=f"Graph build job '{exc}' is already running."
        ) from exc


@router.get("/dry-run", response_model=DryRunResponse)
def dry_run(
    store: Store, collection: Annotated[Collection, Depends(get_chroma_collection)]
) -> DryRunResponse:
    """Discover the corpus and report what a build would process. Writes nothing, no Groq."""
    discovery = _discover(collection)
    job_id = store.latest_job_id()
    done = 0
    job_status = None
    if job_id is not None:
        job_status = store.get_job(job_id)["status"]
        done = store.chunk_counts(job_id)["completed"]
    return DryRunResponse(
        discovery=_summary(discovery),
        resumable_job_id=job_id,
        resumable_job_status=job_status,
        chunks_already_completed=done,
        chunks_to_process=discovery.eligible_chunks - done,
        chunks_to_process_without_resume=discovery.eligible_chunks,
    )


@router.post("", response_model=JobActionResponse, status_code=status.HTTP_202_ACCEPTED)
def start_build(
    request: BuildAllRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    store: Store,
    build_runner: Runner,
    collection: Annotated[Collection, Depends(get_chroma_collection)],
    client: Annotated[Neo4jClient, Depends(get_graph_client)],
    llm: Annotated[JSONCompletionClient, Depends(get_extraction_llm)],
) -> JobActionResponse:
    """Start (or, with ``resume: true``, continue the latest) full-corpus build job.

    409 if a job is running; 503 if ChromaDB, Neo4j or Groq is unavailable. Resuming skips
    every chunk already completed; failed chunks are retried only via ``/{job_id}/retry``.
    """
    _ensure_idle(store, build_runner)
    discovery = _discover(collection)
    try:
        ensure_schema(client)
    except GraphStoreError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    options = _options(request, settings)
    job_id = store.latest_job_id() if request.resume else None
    resumed = job_id is not None
    if resumed:
        store.sync_job(job_id, discovery, options)
    else:
        job_id = store.create_job(discovery, options)
    pending = store.chunk_counts(job_id)["pending"]
    _launch(store, build_runner, job_id, collection, client, llm)
    return JobActionResponse(
        job_id=job_id,
        status="running",
        resumed=resumed,
        chunks_to_process=pending,
        message=f"Processing {pending} pending chunk(s) in the background.",
        discovery=_summary(discovery),
    )


@router.get("/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str, store: Store) -> JobStatusResponse:
    data = store.status(job_id)
    if data is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' was not found.")
    job, documents, chunks = data["job"], data["documents"], data["chunks"]
    total = chunks["total"]
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        error=job["error"],
        stop_requested=bool(job["stop_requested"]),
        options=BuildOptions.model_validate_json(job["options"]),
        records_missing_document_id=job["records_missing_document_id"],
        total_documents=documents["total"],
        documents_completed=documents["completed"],
        documents_partially_completed=documents["partial"],
        documents_failed=documents["failed"],
        documents_pending=documents["pending"],
        total_chunks=total,
        chunks_attempted=chunks["attempted"],
        chunks_extracted=chunks["extracted"],
        chunks_processed=chunks["processed"],
        chunks_failed=chunks["failed"],
        chunks_pending=chunks["pending"],
        entities_extracted=chunks["entities"],
        relationships_extracted=chunks["relationships"],
        nodes_created=chunks["nodes_created"],
        nodes_merged=chunks["nodes_merged"],
        relationships_created=chunks["relationships_created"],
        relationships_merged=chunks["relationships_merged"],
        documents_progress=f"{documents['completed']}/{documents['total']}",
        chunks_progress=f"{chunks['processed']}/{total}",
        progress_percentage=round(100 * chunks["processed"] / total, 2) if total else 0.0,
        created_at=job["created_at"],
        started_at=job["started_at"],
        updated_at=job["updated_at"],
        completed_at=job["completed_at"],
    )


@router.get("/{job_id}/failures", response_model=FailuresResponse)
def job_failures(
    job_id: str,
    store: Store,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> FailuresResponse:
    """Documents that have failed chunks, with each chunk's stage, error and attempt count."""
    _require_job(store, job_id)
    total, documents = store.failures(job_id, offset, limit)
    return FailuresResponse(
        job_id=job_id,
        total_failed_documents=total,
        offset=offset,
        limit=limit,
        documents=documents,
    )


@router.post(
    "/{job_id}/retry", response_model=JobActionResponse, status_code=status.HTTP_202_ACCEPTED
)
def retry_failures(
    job_id: str,
    store: Store,
    build_runner: Runner,
    collection: Annotated[Collection, Depends(get_chroma_collection)],
    client: Annotated[Neo4jClient, Depends(get_graph_client)],
    llm: Annotated[JSONCompletionClient, Depends(get_extraction_llm)],
) -> JobActionResponse:
    """Re-run only failed chunks (plus any still pending). Completed chunks are not touched.

    A chunk that failed at the graph stage reuses its stored extraction: no new Groq call.
    """
    _require_job(store, job_id)
    _ensure_idle(store, build_runner)
    try:
        ensure_schema(client)
    except GraphStoreError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    reset = store.reset_failures(job_id)
    pending = store.chunk_counts(job_id)["pending"]
    if pending == 0:
        store.set_job_status(job_id, "completed")
        return JobActionResponse(
            job_id=job_id, status="completed", chunks_to_process=0, message="Nothing to retry."
        )
    _launch(store, build_runner, job_id, collection, client, llm)
    return JobActionResponse(
        job_id=job_id,
        status="running",
        resumed=True,
        chunks_to_process=pending,
        message=f"Retrying {reset} failed chunk(s); {pending} chunk(s) pending in total.",
    )


@router.post(
    "/{job_id}/stop", response_model=JobActionResponse, status_code=status.HTTP_202_ACCEPTED
)
def stop_build(job_id: str, store: Store, build_runner: Runner) -> JobActionResponse:
    """Ask the running job to pause after its current chunk. Resume with ``POST /build-all``."""
    _require_job(store, job_id)
    if not build_runner.stop(job_id):
        raise HTTPException(status.HTTP_409_CONFLICT, detail=f"Job '{job_id}' is not running.")
    store.request_stop(job_id)
    return JobActionResponse(
        job_id=job_id,
        status="running",
        chunks_to_process=store.chunk_counts(job_id)["pending"],
        message="Stop requested; the job pauses after the current chunk.",
    )
