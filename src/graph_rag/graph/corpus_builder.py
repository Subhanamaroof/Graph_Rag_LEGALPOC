"""Full-corpus graph construction (Story 1.6.2): ChromaDB -> Story 1.5 -> Story 1.6 -> Neo4j.

Nothing here extracts or writes the graph in a new way. Per chunk the pipeline calls:

1. ``load_chunks`` (Story 1.5) to read every chunk of one judgment, without a chunk limit;
2. ``extract_from_chunk`` (Story 1.5) for validated entities and relationships;
3. ``persist_chunk`` (Story 1.6), which writes the chunk in ONE Neo4j transaction with MERGE;
4. the job store, which records the outcome only after step 3 committed.

What this module adds is control: sequential processing, request pacing and bounded retries
for Groq (``ThrottledLLM``), a stop signal checked between chunks, a guard that pauses the job
when many chunks in a row fail for infrastructure reasons, and one background worker thread.
"""

import logging
import threading
import time
from collections.abc import Callable

from chromadb.api.models.Collection import Collection

from graph_rag.core.llm import JSONCompletionClient, LLMError
from graph_rag.graph.build_job_store import BuildJobStore
from graph_rag.graph.corpus_models import BuildOptions
from graph_rag.graph.entity_extractor import extract_from_chunk
from graph_rag.graph.extraction_service import ChunkNotFoundError, load_chunks
from graph_rag.graph.graph_service import MissingEndpointError, persist_chunk
from graph_rag.graph.models import ChunkExtraction
from graph_rag.graph.neo4j_client import GraphStoreError, Neo4jClient

logger = logging.getLogger(__name__)

MAX_BACKOFF_SECONDS = 120.0


class BuildStopped(Exception):  # deliberately NOT an LLMError: extraction must not swallow it
    """A stop was requested while waiting; the current chunk stays pending."""


class TooManyFailuresError(Exception):
    """Several chunks in a row failed for transient/infrastructure reasons; the job pauses."""


class ThrottledLLM:
    """Wraps the Groq client: paces requests and retries transient errors with backoff.

    - At least ``delay_seconds`` pass between the start of two requests.
    - A retryable ``LLMError`` (HTTP 429, 5xx, network) is retried at most ``max_retries``
      times, waiting ``retry_delay * 2**attempt`` seconds (or the server's ``retry-after``),
      capped at ``MAX_BACKOFF_SECONDS``. Non-retryable errors are raised at once.
    - ``wait(seconds)`` returns True when a stop was requested, so waits end early.
    """

    def __init__(
        self,
        llm: JSONCompletionClient,
        *,
        delay_seconds: float,
        max_retries: int,
        retry_delay_seconds: float,
        wait: Callable[[float], bool],
        clock: Callable[[], float] = time.monotonic,
    ):
        self._llm = llm
        self.delay_seconds = delay_seconds
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self._wait = wait
        self._clock = clock
        self._last_request: float | None = None
        self.requests = 0
        self.retries = 0
        self.last_error_transient = False  # did the last failure exhaust transient retries?

    def _pause(self, seconds: float) -> None:
        if seconds > 0 and self._wait(seconds):
            raise BuildStopped()

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        self.last_error_transient = False
        attempt = 0
        while True:
            if self._last_request is not None:
                self._pause(self.delay_seconds - (self._clock() - self._last_request))
            self._last_request = self._clock()
            self.requests += 1
            try:
                return self._llm.complete_json(system_prompt, user_prompt)
            except LLMError as exc:
                if not exc.retryable:
                    raise
                if attempt >= self.max_retries:
                    self.last_error_transient = True
                    raise LLMError(
                        f"{exc} Gave up after {attempt + 1} attempt(s).", retryable=True
                    ) from exc
                backoff = exc.retry_after or self.retry_delay_seconds * 2**attempt
                attempt += 1
                self.retries += 1
                logger.warning("Transient LLM error (%s); retry %d in %.1fs", exc, attempt, backoff)
                self._pause(min(backoff, MAX_BACKOFF_SECONDS))


def process_chunk(
    store: BuildJobStore,
    job_id: str,
    *,
    document_id: str,
    chunk_id: str,
    chunk_text: str | None,
    cached_extraction: str | None,
    client: Neo4jClient,
    llm: ThrottledLLM,
) -> bool:
    """Extract (or reuse a cached extraction) and persist one chunk.

    Returns True if the chunk failed for a transient/infrastructure reason (counts towards the
    consecutive-failure guard). ``BuildStopped`` propagates and leaves the chunk pending.
    """
    store.record_attempt(job_id, chunk_id)
    if cached_extraction is not None:  # extraction succeeded earlier; only the write failed
        extraction = ChunkExtraction.model_validate_json(cached_extraction)
    elif chunk_text is None:
        store.chunk_failed(job_id, chunk_id, "load", "Chunk is no longer in the collection.")
        return False
    else:
        extraction = extract_from_chunk(
            llm, document_id=document_id, chunk_id=chunk_id, chunk_text=chunk_text
        )
        if extraction.extraction_status == "failed":
            store.chunk_failed(job_id, chunk_id, "extraction", extraction.error or "")
            return llm.last_error_transient
    try:
        written = persist_chunk(client, extraction)  # one transaction: all or nothing
    except MissingEndpointError as exc:
        store.chunk_failed(job_id, chunk_id, "graph", str(exc), extraction)
        return False
    except GraphStoreError as exc:
        store.chunk_failed(job_id, chunk_id, "graph", str(exc), extraction)
        return True
    store.chunk_completed(job_id, chunk_id, extraction, written)
    return False


def process_document(
    store: BuildJobStore,
    job_id: str,
    document_id: str,
    *,
    collection: Collection,
    client: Neo4jClient,
    llm: ThrottledLLM,
    stop: threading.Event,
    on_chunk_done: Callable[[bool], None],
) -> str:
    """Process every pending chunk of one judgment, then derive the document status."""
    store.mark_document_processing(job_id, document_id)
    pending = store.pending_chunks(job_id, document_id)
    texts: dict[str, str] = {}
    if any(cached is None for _, cached in pending):
        try:
            texts = {r["id"]: r["document"] for r in load_chunks(collection, document_id)}
        except ChunkNotFoundError as exc:
            store.fail_pending_chunks(job_id, document_id, str(exc))
            return store.finalize_document(job_id, document_id, error=str(exc))
    try:
        for chunk_id, cached in pending:
            if stop.is_set():
                break
            transient_failure = process_chunk(
                store,
                job_id,
                document_id=document_id,
                chunk_id=chunk_id,
                chunk_text=texts.get(chunk_id),
                cached_extraction=cached,
                client=client,
                llm=llm,
            )
            on_chunk_done(transient_failure)
    finally:
        status = store.finalize_document(job_id, document_id)
    logger.info("Document %s: %s", document_id, status)
    return status


def run_job(
    store: BuildJobStore,
    job_id: str,
    *,
    collection: Collection,
    client: Neo4jClient,
    llm: JSONCompletionClient,
    stop: threading.Event,
) -> str:
    """Process all pending documents of a job, batch by batch, until done or stopped.

    Never raises: any problem ends the job as ``paused``/``interrupted`` with its error stored,
    and the job can be resumed later. Returns the final job status.
    """
    options: BuildOptions = store.options(job_id)
    throttled = ThrottledLLM(
        llm,
        delay_seconds=options.delay_seconds,
        max_retries=options.max_retries,
        retry_delay_seconds=options.retry_delay_seconds,
        wait=stop.wait,
    )
    consecutive = 0

    def on_chunk_done(transient_failure: bool) -> None:
        nonlocal consecutive
        consecutive = consecutive + 1 if transient_failure else 0
        if consecutive >= options.max_consecutive_failures:
            raise TooManyFailuresError(
                f"{consecutive} chunks in a row failed with rate-limit, network or Neo4j errors; "
                "the job was paused. Check Groq/Neo4j, then resume or retry."
            )

    store.set_job_status(job_id, "running")
    logger.info("Graph build job %s started", job_id)
    try:
        while not stop.is_set():
            batch = store.pending_documents(job_id, options.batch_size)
            if not batch:
                break
            for document_id in batch:
                if stop.is_set():
                    break
                process_document(
                    store,
                    job_id,
                    document_id,
                    collection=collection,
                    client=client,
                    llm=throttled,
                    stop=stop,
                    on_chunk_done=on_chunk_done,
                )
    except BuildStopped:
        pass
    except TooManyFailuresError as exc:
        store.set_job_status(job_id, "interrupted", str(exc))
        return "interrupted"
    except Exception as exc:  # ChromaDB down, unexpected bug: keep progress, report the error
        logger.exception("Graph build job %s interrupted", job_id)
        store.set_job_status(job_id, "interrupted", f"{type(exc).__name__}: {exc}")
        return "interrupted"

    counts = store.chunk_counts(job_id)
    if counts["pending"]:
        status = "paused"
    elif counts["failed"]:
        status = "completed_with_failures"
    else:
        status = "completed"
    store.set_job_status(job_id, status)
    logger.info("Graph build job %s %s (%s)", job_id, status, counts)
    return status


class JobAlreadyRunningError(RuntimeError):
    pass


class BuildRunner:
    """Runs at most ONE job at a time on a background thread inside the API process."""

    def __init__(self):
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.job_id: str | None = None

    def active_job_id(self) -> str | None:
        with self._lock:
            alive = self._thread is not None and self._thread.is_alive()
            return self.job_id if alive else None

    def start(self, job_id: str, work: Callable[[threading.Event], object]) -> None:
        """Start ``work(stop_event)`` on a new thread; refuse if a job is already running."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise JobAlreadyRunningError(self.job_id)
            self._stop = threading.Event()
            self.job_id = job_id
            # Daemon: a thread stuck in a long Groq call cannot block interpreter exit. Its job
            # stays "running" in SQLite and is marked "interrupted" on the next startup.
            self._thread = threading.Thread(
                target=work, args=(self._stop,), name=f"graph-build-{job_id}", daemon=True
            )
            self._thread.start()

    def stop(self, job_id: str | None = None) -> bool:
        """Ask the running job to stop after its current chunk. Returns False if none runs."""
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                return False
            if job_id is not None and job_id != self.job_id:
                return False
            self._stop.set()
            return True

    def shutdown(self, timeout: float = 15.0) -> None:
        """Signal stop and wait briefly; called when FastAPI shuts down."""
        self.stop()
        thread = self._thread
        if thread is not None:
            thread.join(timeout)


runner = BuildRunner()
