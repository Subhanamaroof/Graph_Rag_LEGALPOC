"""FastAPI application entry point.

The application object, the ``/health`` endpoint and the API routers are wired here.
Configuration is in ``core/config.py``. The ChromaDB client is opened lazily on the first
retrieval request (so ``/health`` works without it) and released on shutdown. The Neo4j
driver (Story 1.6) is likewise created on the first graph request and closed on shutdown.
A full-corpus graph build (Story 1.6.2) never starts on its own: at startup, jobs left
``running`` by a previous process are only marked ``interrupted``; at shutdown, a running job
is asked to stop after its current chunk before the clients are closed.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from graph_rag.core.config import get_settings
from graph_rag.generation.generation_router import router as generation_router
from graph_rag.graph.build_job_store import BuildJobStore
from graph_rag.graph.corpus_builder import runner as graph_build_runner
from graph_rag.graph.corpus_router import router as corpus_build_router
from graph_rag.graph.graph_router import router as graph_router
from graph_rag.graph.neo4j_client import close_neo4j_client
from graph_rag.graph.router import router as extraction_router
from graph_rag.ingestion.router import router as ingestion_router
from graph_rag.retrieval.graph_router import router as graph_retrieval_router
from graph_rag.retrieval.hybrid_router import router as hybrid_retrieval_router
from graph_rag.retrieval.router import router as retrieval_router
from graph_rag.vector_store.chroma_store import close_client


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    jobs_db = get_settings().graph_build_jobs_db_path
    if jobs_db.is_file():  # never create the job database just by starting the app
        BuildJobStore(jobs_db).recover_interrupted_jobs()
    yield
    graph_build_runner.shutdown()
    close_client()
    close_neo4j_client()


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness check. Does not touch ChromaDB, Neo4j, Groq or the embedding model."""
        return {"status": "ok", "application": settings.app_name}

    app.include_router(ingestion_router)
    app.include_router(retrieval_router)
    app.include_router(extraction_router)
    app.include_router(graph_router)
    app.include_router(corpus_build_router)
    app.include_router(graph_retrieval_router)
    app.include_router(hybrid_retrieval_router)
    app.include_router(generation_router)
    return app


app = create_app()


def run() -> None:
    """Start the development server using host/port from settings."""
    import uvicorn

    settings = get_settings()
    uvicorn.run("graph_rag.main:app", host=settings.app_host, port=settings.app_port)


if __name__ == "__main__":
    run()
