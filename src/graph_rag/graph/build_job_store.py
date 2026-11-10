"""Durable job tracking for the full-corpus graph build (Story 1.6.2), in a local SQLite file.

Progress lives here, NOT in Neo4j: the knowledge graph only ever contains legal entities and
relationships. Three tables:

- ``build_jobs``: one row per job (status, options, discovery figures, timestamps).
- ``build_documents``: one row per (job, judgment) with the document status.
- ``build_chunks``: one row per (job, chunk). This is the source of truth for progress, resume
  and retry. A chunk is ``completed`` only after its graph transaction committed. A chunk whose
  extraction succeeded but whose graph write failed keeps its extraction JSON, so a retry only
  repeats the Neo4j write and does not call Groq again.

Each method opens its own short connection, so the background worker thread and the API
threads never share a connection. Every write is one SQLite transaction (``with conn``).
"""

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from graph_rag.graph.corpus_discovery import INVALID_ID_SAMPLE, CorpusDiscovery
from graph_rag.graph.corpus_models import (
    BuildOptions,
    ChunkFailure,
    DocumentFailure,
    DocumentStatus,
    FailureStage,
    JobStatus,
)
from graph_rag.graph.graph_models import ChunkWriteResult
from graph_rag.graph.models import ChunkExtraction

_SCHEMA = """
CREATE TABLE IF NOT EXISTS build_jobs (
    job_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    options TEXT NOT NULL,
    total_chunk_records INTEGER NOT NULL DEFAULT 0,
    records_missing_document_id INTEGER NOT NULL DEFAULT 0,
    invalid_record_ids TEXT NOT NULL DEFAULT '[]',
    stop_requested INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE TABLE IF NOT EXISTS build_documents (
    job_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    total_chunks INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    started_at TEXT,
    completed_at TEXT,
    PRIMARY KEY (job_id, document_id)
);
CREATE TABLE IF NOT EXISTS build_chunks (
    job_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    stage TEXT,
    error TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    extracted INTEGER NOT NULL DEFAULT 0,
    entities INTEGER NOT NULL DEFAULT 0,
    relationships INTEGER NOT NULL DEFAULT 0,
    nodes_created INTEGER NOT NULL DEFAULT 0,
    nodes_merged INTEGER NOT NULL DEFAULT 0,
    relationships_created INTEGER NOT NULL DEFAULT 0,
    relationships_merged INTEGER NOT NULL DEFAULT 0,
    extraction_json TEXT,
    updated_at TEXT,
    PRIMARY KEY (job_id, chunk_id)
);
CREATE INDEX IF NOT EXISTS build_chunks_by_document ON build_chunks (job_id, document_id);
"""

FINISHED_STATUSES = frozenset({"completed", "completed_with_failures"})


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class BuildJobStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _write(self, sql: str, params: tuple = ()) -> int:
        with closing(self._connect()) as conn, conn:
            return conn.execute(sql, params).rowcount

    def _one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        with closing(self._connect()) as conn:
            return conn.execute(sql, params).fetchone()

    def _all(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with closing(self._connect()) as conn:
            return conn.execute(sql, params).fetchall()

    # --- Jobs -------------------------------------------------------------------------

    def create_job(self, discovery: CorpusDiscovery, options: BuildOptions) -> str:
        job_id = uuid.uuid4().hex
        timestamp = now()
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO build_jobs (job_id, status, options, created_at, updated_at) "
                "VALUES (?, 'pending', ?, ?, ?)",
                (job_id, options.model_dump_json(), timestamp, timestamp),
            )
            self._register(conn, job_id, discovery, options)
        return job_id

    def sync_job(self, job_id: str, discovery: CorpusDiscovery, options: BuildOptions) -> None:
        """Prepare an existing job for resuming: add newly discovered chunks, keep all progress.

        Completed and failed chunks are untouched. Documents left ``processing`` by an
        interruption go back to ``pending``.
        """
        with closing(self._connect()) as conn, conn:
            self._register(conn, job_id, discovery, options)
            conn.execute(
                "UPDATE build_jobs SET status = 'pending', error = NULL, stop_requested = 0, "
                "completed_at = NULL WHERE job_id = ?",
                (job_id,),
            )

    def _register(
        self,
        conn: sqlite3.Connection,
        job_id: str,
        discovery: CorpusDiscovery,
        options: BuildOptions,
    ) -> None:
        conn.execute(
            "UPDATE build_jobs SET options = ?, total_chunk_records = ?, "
            "records_missing_document_id = ?, invalid_record_ids = ?, updated_at = ? "
            "WHERE job_id = ?",
            (
                options.model_dump_json(),
                discovery.total_chunk_records,
                discovery.records_missing_document_id,
                json.dumps(discovery.invalid_record_ids[:INVALID_ID_SAMPLE]),
                now(),
                job_id,
            ),
        )
        conn.executemany(
            "INSERT OR IGNORE INTO build_documents (job_id, document_id, total_chunks) "
            "VALUES (?, ?, ?)",
            [(job_id, doc, len(chunks)) for doc, chunks in discovery.documents.items()],
        )
        conn.executemany(
            "INSERT OR IGNORE INTO build_chunks (job_id, document_id, chunk_id, position) "
            "VALUES (?, ?, ?, ?)",
            [
                (job_id, doc, chunk_id, position)
                for doc, chunks in discovery.documents.items()
                for position, chunk_id in enumerate(chunks)
            ],
        )
        conn.execute(
            "UPDATE build_documents SET total_chunks = (SELECT count(*) FROM build_chunks c "
            "WHERE c.job_id = build_documents.job_id AND c.document_id = "
            "build_documents.document_id) WHERE job_id = ?",
            (job_id,),
        )
        conn.execute(
            "UPDATE build_documents SET status = 'pending', completed_at = NULL "
            "WHERE job_id = ? AND (status = 'processing' OR document_id IN "
            "(SELECT document_id FROM build_chunks WHERE job_id = ? AND status = 'pending'))",
            (job_id, job_id),
        )

    def get_job(self, job_id: str) -> sqlite3.Row | None:
        return self._one("SELECT * FROM build_jobs WHERE job_id = ?", (job_id,))

    def latest_job_id(self) -> str | None:
        row = self._one("SELECT job_id FROM build_jobs ORDER BY created_at DESC, rowid DESC")
        return row["job_id"] if row else None

    def running_job_id(self) -> str | None:
        row = self._one("SELECT job_id FROM build_jobs WHERE status = 'running'")
        return row["job_id"] if row else None

    def options(self, job_id: str) -> BuildOptions:
        return BuildOptions.model_validate_json(self.get_job(job_id)["options"])

    def set_job_status(self, job_id: str, status: JobStatus, error: str | None = None) -> None:
        timestamp = now()
        started = timestamp if status == "running" else None
        completed = timestamp if status in FINISHED_STATUSES else None
        self._write(
            "UPDATE build_jobs SET status = ?, error = ?, updated_at = ?, "
            "started_at = coalesce(started_at, ?), completed_at = ? WHERE job_id = ?",
            (status, error, timestamp, started, completed, job_id),
        )

    def request_stop(self, job_id: str) -> None:
        self._write(
            "UPDATE build_jobs SET stop_requested = 1, updated_at = ? WHERE job_id = ?",
            (now(), job_id),
        )

    def recover_interrupted_jobs(self) -> list[str]:
        """After a restart no job can still be running: mark leftovers ``interrupted``."""
        with closing(self._connect()) as conn, conn:
            ids = [
                row["job_id"]
                for row in conn.execute("SELECT job_id FROM build_jobs WHERE status = 'running'")
            ]
            for job_id in ids:
                conn.execute(
                    "UPDATE build_jobs SET status = 'interrupted', updated_at = ?, error = "
                    "'The application stopped while the job was running.' WHERE job_id = ?",
                    (now(), job_id),
                )
                conn.execute(
                    "UPDATE build_documents SET status = 'pending' "
                    "WHERE job_id = ? AND status = 'processing'",
                    (job_id,),
                )
        return ids

    def reset_failures(self, job_id: str) -> int:
        """Make failed chunks pending again (extraction JSON kept). Returns how many."""
        with closing(self._connect()) as conn, conn:
            reset = conn.execute(
                "UPDATE build_chunks SET status = 'pending' WHERE job_id = ? AND status = 'failed'",
                (job_id,),
            ).rowcount
            conn.execute(
                "UPDATE build_documents SET status = 'pending', error = NULL, completed_at = NULL "
                "WHERE job_id = ? AND document_id IN "
                "(SELECT document_id FROM build_chunks WHERE job_id = ? AND status = 'pending')",
                (job_id, job_id),
            )
            conn.execute(
                "UPDATE build_jobs SET status = 'pending', error = NULL, stop_requested = 0, "
                "completed_at = NULL, updated_at = ? WHERE job_id = ?",
                (now(), job_id),
            )
        return reset

    # --- Documents and chunks -----------------------------------------------------------

    def pending_documents(self, job_id: str, limit: int) -> list[str]:
        rows = self._all(
            "SELECT document_id FROM build_documents WHERE job_id = ? "
            "AND status IN ('pending', 'processing') ORDER BY document_id LIMIT ?",
            (job_id, limit),
        )
        return [row["document_id"] for row in rows]

    def mark_document_processing(self, job_id: str, document_id: str) -> None:
        timestamp = now()
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "UPDATE build_documents SET status = 'processing', error = NULL, "
                "started_at = coalesce(started_at, ?), completed_at = NULL "
                "WHERE job_id = ? AND document_id = ?",
                (timestamp, job_id, document_id),
            )
            conn.execute(
                "UPDATE build_jobs SET updated_at = ? WHERE job_id = ?", (timestamp, job_id)
            )

    def pending_chunks(self, job_id: str, document_id: str) -> list[tuple[str, str | None]]:
        """``(chunk_id, cached extraction JSON or None)`` in reading order."""
        rows = self._all(
            "SELECT chunk_id, extraction_json FROM build_chunks WHERE job_id = ? "
            "AND document_id = ? AND status = 'pending' ORDER BY position",
            (job_id, document_id),
        )
        return [(row["chunk_id"], row["extraction_json"]) for row in rows]

    def record_attempt(self, job_id: str, chunk_id: str) -> None:
        self._write(
            "UPDATE build_chunks SET attempts = attempts + 1, updated_at = ? "
            "WHERE job_id = ? AND chunk_id = ?",
            (now(), job_id, chunk_id),
        )

    def chunk_failed(
        self,
        job_id: str,
        chunk_id: str,
        stage: FailureStage,
        error: str,
        extraction: ChunkExtraction | None = None,
    ) -> None:
        """Record a failure; an extraction that succeeded (graph stage) is kept for retry."""
        extracted = extraction is not None and extraction.extraction_status == "completed"
        self._write(
            "UPDATE build_chunks SET status = 'failed', stage = ?, error = ?, extracted = ?, "
            "entities = ?, relationships = ?, extraction_json = ?, updated_at = ? "
            "WHERE job_id = ? AND chunk_id = ?",
            (
                stage,
                error,
                int(extracted),
                len(extraction.entities) if extracted else 0,
                len(extraction.relationships) if extracted else 0,
                extraction.model_dump_json() if extracted else None,
                now(),
                job_id,
                chunk_id,
            ),
        )

    def chunk_completed(
        self, job_id: str, chunk_id: str, extraction: ChunkExtraction, written: ChunkWriteResult
    ) -> None:
        """Called only after the chunk's Neo4j transaction committed."""
        self._write(
            "UPDATE build_chunks SET status = 'completed', stage = NULL, error = NULL, "
            "extracted = 1, entities = ?, relationships = ?, nodes_created = ?, "
            "nodes_merged = ?, relationships_created = ?, relationships_merged = ?, "
            "extraction_json = NULL, updated_at = ? WHERE job_id = ? AND chunk_id = ?",
            (
                len(extraction.entities),
                len(extraction.relationships),
                written.nodes_created,
                written.nodes_merged,
                written.relationships_created,
                written.relationships_merged,
                now(),
                job_id,
                chunk_id,
            ),
        )

    def fail_pending_chunks(self, job_id: str, document_id: str, error: str) -> None:
        """The document's chunks could not be loaded: fail every still-pending chunk."""
        self._write(
            "UPDATE build_chunks SET status = 'failed', stage = 'load', error = ?, "
            "attempts = attempts + 1, updated_at = ? "
            "WHERE job_id = ? AND document_id = ? AND status = 'pending'",
            (error, now(), job_id, document_id),
        )

    def finalize_document(
        self, job_id: str, document_id: str, error: str | None = None
    ) -> DocumentStatus:
        """Derive the document status from its chunks; never ``completed`` with failures."""
        timestamp = now()
        with closing(self._connect()) as conn, conn:
            row = conn.execute(
                "SELECT sum(status = 'completed') AS done, sum(status = 'failed') AS failed, "
                "sum(status = 'pending') AS pending FROM build_chunks "
                "WHERE job_id = ? AND document_id = ?",
                (job_id, document_id),
            ).fetchone()
            done, failed, pending = (row["done"] or 0), (row["failed"] or 0), (row["pending"] or 0)
            status: DocumentStatus
            if pending:
                status = "pending"  # stopped part-way; resume continues here
            elif not failed:
                status = "completed"
            elif not done:
                status = "failed"
            else:
                status = "partially_completed"
            conn.execute(
                "UPDATE build_documents SET status = ?, error = ?, completed_at = ? "
                "WHERE job_id = ? AND document_id = ?",
                (
                    status,
                    error,
                    None if status == "pending" else timestamp,
                    job_id,
                    document_id,
                ),
            )
            conn.execute(
                "UPDATE build_jobs SET updated_at = ? WHERE job_id = ?", (timestamp, job_id)
            )
        return status

    # --- Reporting --------------------------------------------------------------------

    def chunk_counts(self, job_id: str) -> dict[str, int]:
        row = self._one(
            "SELECT count(*) AS total, sum(status = 'completed') AS completed, "
            "sum(status = 'failed') AS failed, sum(status = 'pending') AS pending "
            "FROM build_chunks WHERE job_id = ?",
            (job_id,),
        )
        return {key: row[key] or 0 for key in ("total", "completed", "failed", "pending")}

    def status(self, job_id: str) -> dict[str, Any] | None:
        """Job row plus document/chunk aggregates, all computed from stored results."""
        job = self.get_job(job_id)
        if job is None:
            return None
        documents = self._one(
            "SELECT count(*) AS total, sum(status = 'completed') AS completed, "
            "sum(status = 'partially_completed') AS partial, sum(status = 'failed') AS failed, "
            "sum(status IN ('pending', 'processing')) AS pending "
            "FROM build_documents WHERE job_id = ?",
            (job_id,),
        )
        chunks = self._one(
            "SELECT count(*) AS total, sum(attempts > 0) AS attempted, "
            "sum(extracted) AS extracted, sum(status = 'completed') AS processed, "
            "sum(status = 'failed') AS failed, sum(status = 'pending') AS pending, "
            "sum(CASE WHEN extracted = 1 THEN entities ELSE 0 END) AS entities, "
            "sum(CASE WHEN extracted = 1 THEN relationships ELSE 0 END) AS relationships, "
            "sum(nodes_created) AS nodes_created, sum(nodes_merged) AS nodes_merged, "
            "sum(relationships_created) AS relationships_created, "
            "sum(relationships_merged) AS relationships_merged "
            "FROM build_chunks WHERE job_id = ?",
            (job_id,),
        )
        return {
            "job": dict(job),
            "documents": {key: documents[key] or 0 for key in documents.keys()},
            "chunks": {key: chunks[key] or 0 for key in chunks.keys()},
        }

    def failures(self, job_id: str, offset: int, limit: int) -> tuple[int, list[DocumentFailure]]:
        """Documents with at least one failed chunk, paginated, with chunk-level errors."""
        where = (
            "FROM build_documents d WHERE d.job_id = ? AND EXISTS (SELECT 1 FROM build_chunks c "
            "WHERE c.job_id = d.job_id AND c.document_id = d.document_id AND c.status = 'failed')"
        )
        total = self._one(f"SELECT count(*) AS n {where}", (job_id,))["n"]
        documents = self._all(
            f"SELECT d.* {where} ORDER BY d.document_id LIMIT ? OFFSET ?", (job_id, limit, offset)
        )
        result = []
        for document in documents:
            chunks = self._all(
                "SELECT chunk_id, status, stage, error, attempts FROM build_chunks "
                "WHERE job_id = ? AND document_id = ? ORDER BY position",
                (job_id, document["document_id"]),
            )
            failed = [chunk for chunk in chunks if chunk["status"] == "failed"]
            result.append(
                DocumentFailure(
                    document_id=document["document_id"],
                    status=document["status"],
                    total_chunks=document["total_chunks"],
                    chunks_processed=sum(chunk["status"] == "completed" for chunk in chunks),
                    chunks_failed=len(failed),
                    error=document["error"],
                    failed_chunks=[
                        ChunkFailure(
                            chunk_id=chunk["chunk_id"],
                            stage=chunk["stage"],
                            error=chunk["error"] or "",
                            attempts=chunk["attempts"],
                        )
                        for chunk in failed
                    ],
                )
            )
        return total, result
