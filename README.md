# Graph RAG POC

A learning-oriented **Proof of Concept** for Graph RAG (Retrieval-Augmented Generation).
The goal is to understand, step by step, how a system can combine:

- **Vector retrieval** — MongoDB Atlas Vector Search over embedded text chunks.
- **Graph retrieval** — a Neo4j knowledge graph of entities and relationships.

…and feed both kinds of context to an LLM.

> **Current status (Story 1.1):** only the project architecture, configuration and a
> `GET /health` endpoint exist. Document ingestion, embeddings, graph construction,
> retrieval and answer generation are **not implemented yet**.

See [docs/architecture.md](docs/architecture.md) for the architecture and
[docs/explanation/](docs/explanation/) for per-story technical explanations.

## Technology Stack

| Component       | Technology                                 |
| --------------- | ------------------------------------------ |
| Language        | Python 3.12                                |
| API Framework   | FastAPI (served by Uvicorn)                |
| LLM Provider    | Groq                                       |
| Embedding Model | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Database | MongoDB Atlas (Atlas Vector Search)        |
| Graph Database  | Neo4j                                      |
| Configuration   | Pydantic Settings                          |
| Package Manager | uv                                         |
| Lint / Tests    | Ruff, Pytest, HTTPX                        |

## Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

Not needed yet (required in later stories): a MongoDB Atlas cluster, a Neo4j instance
and a Groq API key.

## Setup

Install uv (if not already installed):

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install the project and its dependencies (creates `.venv/` from `uv.lock`):

```bash
uv sync
```

> Note: `sentence-transformers` pulls in PyTorch, so the first sync downloads a few
> hundred MB.

## Environment Configuration

Copy the example file and fill in values as they become needed:

```bash
cp .env.example .env        # macOS / Linux / Git Bash
copy .env.example .env      # Windows cmd / PowerShell
```

All credential fields (`MONGODB_URI`, `NEO4J_*`, `GROQ_*`) may stay empty for now —
the application starts without them and makes no database or LLM connections.
`.env` is git-ignored; never commit real credentials.

## Run the Application

```bash
uv run uvicorn graph_rag.main:app --reload
```

Or, using `APP_HOST` / `APP_PORT` from the settings:

```bash
uv run python -m graph_rag.main
```

The API is served at <http://127.0.0.1:8000> (interactive docs at `/docs`).

## Test `/health`

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status": "ok", "application": "Graph RAG POC"}
```

Run the automated test and lint checks:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
