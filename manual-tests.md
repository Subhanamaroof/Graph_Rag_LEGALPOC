# Manual Tests

Manual checks for each implemented story. Base URL: `http://127.0.0.1:8000`.

## Story 1.1 — Architecture and Project Setup

**What it adds:** a Python 3.12 / uv project skeleton with centralized Pydantic settings and a FastAPI app that exposes only `GET /health`. It makes no MongoDB, Neo4j, Groq or embedding-model connections.

**Prerequisites:** run `uv sync` once. A `.env` file is optional: the app starts with every credential (`MONGODB_URI`, `NEO4J_*`, `GROQ_API_KEY`) unset.

Start the server (in its own terminal):

```bash
uv run uvicorn graph_rag.main:app --reload
```

Expected: Uvicorn reports `Uvicorn running on http://127.0.0.1:8000` with no database or credential errors.

Alternative start, using `APP_HOST` / `APP_PORT` from settings:

```bash
uv run python -m graph_rag.main
```

Expected: the server starts on the configured host and port (default `127.0.0.1:8000`).

Health check (success):

```bash
curl -s http://127.0.0.1:8000/health
```

Expected: HTTP 200 and `{"status":"ok","application":"Graph RAG POC"}`.

Wrong HTTP method (edge case):

```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8000/health
```

Expected: `405` (`/health` only accepts GET).

Unknown route (edge case):

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/does-not-exist
```

Expected: `404`.

Interactive API docs:

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/docs
```

Expected: `200`. Opening the same URL in a browser shows the Swagger UI.

## Story 1.2 — Document Loading, Text Extraction and Chunking

**What it adds:** `POST /v1/ingestion/process` reads `.txt`, `.md` and text-based `.pdf` files from `DOCUMENTS_DIRECTORY` (default `data/documents`), extracts and cleans their text, and splits it into chunks with deterministic IDs. The chunks are size-limited (`CHUNK_SIZE=800`) and overlap (`CHUNK_OVERLAP=150`). This story implements loading, extraction, cleaning and chunking only. It generates no embeddings, and it saves no chunks to MongoDB, Neo4j or local files: they exist only in the HTTP response.

**Prerequisites:**
- `uv sync` has been run.
- The sample source documents `document_1.txt`, `document_2.md` and `document_3.pdf` are in `data/documents/`.
- The server is running: `uv run uvicorn graph_rag.main:app --reload`.

Summary only (default):

```bash
curl -s -X POST http://127.0.0.1:8000/v1/ingestion/process
```

Expected: HTTP 200 and `{"status":"completed","documents_discovered":3,"documents_processed":3,"documents_failed":0,"total_chunks":3,"errors":[]}`. The response has no `chunks` field.

Include chunks (inspect IDs and metadata, not the full text):

```bash
curl -s -X POST "http://127.0.0.1:8000/v1/ingestion/process?include_chunks=true" \
  | jq -c '.chunks[] | {chunk_id, source_file, file_type, chunk_index, len: (.text|length), metadata}'
```

Expected: HTTP 200 and one chunk per sample file, with these IDs: `doc_6e1f083243944c30_chunk_0000` (txt), `doc_d8d2f54ab394d54a_chunk_0000` (md) and `doc_9a9a512b0e1e99a7_chunk_0000` (pdf). Each chunk is at most 800 characters. TXT/MD chunks have `"page_numbers":[]` and the PDF chunk has `"page_numbers":[1,2]`. Running the command again returns identical IDs.

Invalid query value (edge case):

```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST "http://127.0.0.1:8000/v1/ingestion/process?include_chunks=maybe"
```

Expected: `422` (`include_chunks` must be a boolean).

Missing documents directory (edge case). Stop the server, then restart it pointing to a folder that does not exist:

```bash
DOCUMENTS_DIRECTORY=data/missing uv run uvicorn graph_rag.main:app          # Git Bash / macOS / Linux
$env:DOCUMENTS_DIRECTORY="data/missing"; uv run uvicorn graph_rag.main:app  # PowerShell
curl -s -X POST http://127.0.0.1:8000/v1/ingestion/process
```

Expected: the server starts normally. The request returns HTTP 200 with `"status":"completed"`, `"documents_discovered":0` and `"total_chunks":0`, plus one `errors` entry whose `source_file` is the configured path and whose `error` is `"Directory not found"`. Afterwards, unset the variable (`Remove-Item Env:DOCUMENTS_DIRECTORY` in PowerShell) and restart.

Invalid chunking configuration (edge case):

```bash
CHUNK_SIZE=800 CHUNK_OVERLAP=800 uv run uvicorn graph_rag.main:app          # Git Bash / macOS / Linux
$env:CHUNK_SIZE="800"; $env:CHUNK_OVERLAP="800"; uv run uvicorn graph_rag.main:app  # PowerShell
```

Expected: the server fails to start with a settings validation error, `CHUNK_OVERLAP must be smaller than CHUNK_SIZE`. Unset both variables afterwards.

## Story 1.3 — Embedding Generation and MongoDB Vector Storage

**What it adds:** `POST /v1/ingestion/process` now also embeds every chunk locally with `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions). It then upserts each chunk into MongoDB Atlas (`MONGODB_DATABASE`.`MONGODB_COLLECTION`, default `graph_rag_poc.chunks`), with its text, vector and metadata. Writes are keyed by `chunk_id`, so re-running does not create duplicates. Nothing searches the vectors yet.

**Prerequisites:**
- `uv sync` has been run.
- A MongoDB Atlas cluster, with your IP allowed under *Network Access*.
- A `.env` file (copy `.env.example`) with at least `MONGODB_URI=mongodb+srv://<user>:<password>@<cluster-host>/`. `MONGODB_DATABASE`, `MONGODB_COLLECTION` and `MONGODB_VECTOR_INDEX` have defaults.
- Your documents (`.txt`, `.md`, text-based `.pdf`) are in `DOCUMENTS_DIRECTORY` (default `data/documents`). The three sample files are already there.
- Internet access for the first request, which downloads the embedding model (~90 MB) into the Hugging Face cache. Later requests use the cache.

Create the Atlas Vector Search index (once). Storing chunks works without it, but a later retrieval story needs it:

```bash
uv run python -m graph_rag.vector_store.search_index
```

Expected: `Vector Search index 'chunk_vector_index' on graph_rag_poc.chunks: created` (or `exists` on later runs). In Atlas → *Atlas Search*, the index reaches status `READY` after a short build. To create it in the UI instead, use the JSON in `docs/explanation/story-1.3.md`, section 9.

Start the server (in its own terminal):

```bash
uv run uvicorn graph_rag.main:app --reload
```

Run ingestion (summary only):

```bash
curl -s -X POST http://127.0.0.1:8000/v1/ingestion/process
```

Expected (sample documents, first run): HTTP 200 with `"status":"completed"`, `"documents_discovered":3`, `"documents_processed":3`, `"documents_failed":0`, `"total_chunks":3`, `"total_embeddings":3`, `"chunks_stored":3`, `"stale_chunks_deleted":0`, `"errors":[]`. The `documents` array has one entry per file, with `"status":"succeeded"`, `"chunks":1`, `"embeddings":1`, `"chunks_inserted":1` and `"chunks_updated":0`. The response contains no embedding vectors and no `chunks` field. The first request is slower because it loads the model.

Run it again (idempotency), showing only the per-document counts:

```bash
curl -s -X POST http://127.0.0.1:8000/v1/ingestion/process \
  | jq -c '{status, chunks_stored, docs: [.documents[] | {source_file, status, chunks_inserted, chunks_updated, stale_chunks_deleted}]}'
```

Expected: `"chunks_stored":3`, and every document has `"chunks_inserted":0,"chunks_updated":1`. The collection in Atlas (*Browse Collections* → `graph_rag_poc.chunks`) still contains exactly 3 documents. Each one has `document_id`, `chunk_id`, `text`, an `embedding` array of 384 numbers, `embedding_model`, `metadata` (`source`, `file_type`, `chunk_index`, `page_numbers`, `start_char`, `end_char`), `created_at` and `updated_at`.

Include chunks (chunk IDs only; the response never contains embeddings):

```bash
curl -s -X POST "http://127.0.0.1:8000/v1/ingestion/process?include_chunks=true" \
  | jq -c '[.chunks[] | {chunk_id, has_embedding: has("embedding")}]'
```

Expected: `doc_6e1f083243944c30_chunk_0000`, `doc_d8d2f54ab394d54a_chunk_0000` and `doc_9a9a512b0e1e99a7_chunk_0000`, each with `"has_embedding":false`.

Stale chunk cleanup (edge case): restart the server with a smaller chunk size, run ingestion, then restore the default and run it again:

```bash
CHUNK_SIZE=200 CHUNK_OVERLAP=40 uv run uvicorn graph_rag.main:app                    # Git Bash / macOS / Linux
$env:CHUNK_SIZE="200"; $env:CHUNK_OVERLAP="40"; uv run uvicorn graph_rag.main:app    # PowerShell
curl -s -X POST http://127.0.0.1:8000/v1/ingestion/process | jq -c '{total_chunks, chunks_stored, stale_chunks_deleted}'
```

Expected: more chunks than before (`chunks_stored` equals `total_chunks`). Then unset the variables (`Remove-Item Env:CHUNK_SIZE, Env:CHUNK_OVERLAP` in PowerShell), restart, and run the same curl again. Expected: `total_chunks` is 3 again and `stale_chunks_deleted` is greater than 0. Atlas then holds exactly 3 documents.

MongoDB not configured (failure case): stop the server, remove or comment out `MONGODB_URI` in `.env`, and restart it. The app still starts. Then:

```bash
curl -s -w "\n%{http_code}\n" -X POST http://127.0.0.1:8000/v1/ingestion/process
```

Expected: `503` and `{"detail":"MongoDB is not configured: set MONGODB_URI in .env (see .env.example) to store embeddings."}`. No documents are processed. With a wrong host or a blocked IP, the response is also `503`, with a `Cannot connect to MongoDB: …` message, after about `MONGODB_TIMEOUT_MS` (5 s). Restore `MONGODB_URI` afterwards.

Invalid document (edge case): copy any non-PDF file into the documents folder with a `.pdf` extension, e.g. `echo "not a pdf" > data/documents/broken.pdf`, then run the summary curl. Expected: HTTP 200 with `"status":"completed_with_errors"` and `"documents_failed":1`. The `documents` entry for `broken.pdf` has `"status":"failed"` and an `error` message. The other three documents are still stored. Delete `data/documents/broken.pdf` afterwards.

## Story 1.4 — Vector Retrieval and Similarity Search

**What it adds:** `POST /v1/retrieval/vector` embeds a natural-language `query` with the same `all-MiniLM-L6-v2` model used for ingestion and runs an Atlas `$vectorSearch` on `chunk_vector_index`. It returns the `top_k` nearest chunks (default 5, allowed range 1–20), best match first. Each result has `document_id`, `chunk_id`, `text`, `metadata` and `score`, where `score` is MongoDB's `vectorSearchScore` (0–1 for cosine, not a probability). The response never contains embedding vectors. This story does no graph retrieval and no answer generation.

**Prerequisites:**
- `uv sync` has been run, and `.env` has a working `MONGODB_URI` (see Story 1.3).
- Chunks are ingested: `curl -s -X POST http://127.0.0.1:8000/v1/ingestion/process | jq -c '{status, chunks_stored}'`.
- The Vector Search index exists and is `READY`: `uv run python -m graph_rag.vector_store.search_index` (prints `created` or `exists`). In Atlas → *Atlas Search*, wait until `chunk_vector_index` shows `READY`.

Start the server (in its own terminal):

```bash
uv run uvicorn graph_rag.main:app --reload
```

Successful search (IDs and scores only; no passage text):

```bash
curl -s -X POST http://127.0.0.1:8000/v1/retrieval/vector \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the responsibilities of a hospital?", "top_k": 3}' \
  | jq -c '{query, total_results, hits: [.results[] | {chunk_id, document_id, score, source: .metadata.source, text_len: (.text|length)}]}'
```

Expected: HTTP 200 with `"total_results":3` and three hits. Each hit has a `chunk_id` such as `doc_…_chunk_0007`, a `document_id`, a `score` between 0 and 1, the stored `source` file name and a non-zero `text_len`. Scores never increase down the list, because Atlas returns hits best first. The first request is slower because it may load the model.

Default `top_k` and no embedding in the response:

```bash
curl -s -X POST http://127.0.0.1:8000/v1/retrieval/vector \
  -H "Content-Type: application/json" -d '{"query": "hospital duties"}' \
  | jq -c '{total_results, keys: (.results[0] | keys)}'
```

Expected: `"total_results":5` (if at least 5 chunks are stored) and keys `["chunk_id","document_id","metadata","score","text"]`. There is no `embedding` key.

Invalid input (edge cases):

```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8000/v1/retrieval/vector -H "Content-Type: application/json" -d '{"query": "   "}'
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8000/v1/retrieval/vector -H "Content-Type: application/json" -d '{"query": "hospital", "top_k": 0}'
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8000/v1/retrieval/vector -H "Content-Type: application/json" -d '{"query": "hospital", "top_k": 21}'
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8000/v1/retrieval/vector -H "Content-Type: application/json" -d '{}'
```

Expected: `422` for each (a blank query, `top_k` outside 1–20, a missing `query`).

Empty results: vector search always returns the nearest chunks when any are stored, even for unrelated questions, which just get lower scores. So `total_results: 0` occurs only when the indexed collection is empty. It cannot be reliably demonstrated against the ingested collection. It is covered by the automated test `test_no_matches_returns_empty_response`.

Missing Vector Search index (failure case): before the index is created, or while it is still building:

```bash
curl -s -w "\n%{http_code}\n" -X POST http://127.0.0.1:8000/v1/retrieval/vector -H "Content-Type: application/json" -d '{"query": "hospital"}'
```

Expected: `503` with `"detail":"Vector Search index 'chunk_vector_index' does not exist. Create it with …"`, or `… is not ready yet (status: …)` while it is building.

MongoDB not configured (failure case): remove `MONGODB_URI` from `.env` and restart. Run the same curl. Expected: `503` and `{"detail":"MongoDB is not configured: set MONGODB_URI in .env (see .env.example) to store embeddings."}` (this message is shared with ingestion). Restore `MONGODB_URI` afterwards.

---

## Story 1.3.2 — ChromaDB Integration and Vector Retrieval Migration

**What changes:** `POST /v1/retrieval/vector` now searches the local ChromaDB collection `judgments` in `data/chroma/` (7,325 legal judgment chunks with 384-dimensional MiniLM vectors) instead of MongoDB Atlas. The request schema is unchanged. Each result has `document_id` (the stored `case_id`), `chunk_id` (the ChromaDB record ID), `text`, `metadata` (`case_id`, `court`, `date`, `section`, `title`), `score` (cosine similarity = `1 - distance`, not a probability) and `distance` (ChromaDB's raw cosine distance). MongoDB is no longer used. The Story 1.3/1.4 MongoDB steps above are obsolete.

**Prerequisites:**
- `uv sync` has been run (installs `chromadb==1.5.9`).
- `data/chroma/` contains `chroma.sqlite3` and its segment folders. `.env` either omits `CHROMA_PERSIST_DIRECTORY`/`CHROMA_COLLECTION_NAME` (defaults: `data/chroma`, `judgments`) or sets them as in `.env.example`.
- The `sentence-transformers/all-MiniLM-L6-v2` model is cached locally or can be downloaded.
- Optional: back up the database first (`cp -r data/chroma data/chroma.bak`). Opening the database with Chroma records write-lock bookkeeping in `chroma.sqlite3`, but no records are added, changed or deleted.

Verify ChromaDB initialization (developer command; prints counts and dimensions only):

```bash
uv run python -m graph_rag.vector_store.chroma_store
```

Expected:

```text
collection=judgments count=7325 space=cosine
sample_ids=['NI138-2019-0042_chunk0', 'NI138-2019-0042_chunk1'] has_text=True
metadata_fields=['case_id', 'court', 'date', 'section', 'title']
embedding_dimensions=[384, 384] (expected 384)
```

Start the server (in its own terminal):

```bash
uv run uvicorn graph_rag.main:app --reload
```

Successful search (IDs and scores only; no passage text):

```bash
curl -s -X POST http://127.0.0.1:8000/v1/retrieval/vector \
  -H "Content-Type: application/json" \
  -d '{"query": "bail granted to the accused", "top_k": 3}' \
  | jq -c '{total_results, hits: [.results[] | {chunk_id, document_id, score, distance, court: .metadata.court, text_len: (.text|length)}]}'
```

Expected: HTTP 200 with `"total_results":3`. Each hit has a `chunk_id` such as `NI138-…_chunk16`, a `document_id` equal to the chunk's `case_id`, `score + distance = 1`, a `court` value and a non-zero `text_len`. Scores never increase down the list. The first request is slower because it loads the model and opens the collection.

Response keys (no embedding is returned):

```bash
curl -s -X POST http://127.0.0.1:8000/v1/retrieval/vector \
  -H "Content-Type: application/json" -d '{"query": "dowry death"}' \
  | jq -c '{total_results, keys: (.results[0] | keys), meta: (.results[0].metadata | keys)}'
```

Expected: `"total_results":5`, keys `["chunk_id","distance","document_id","metadata","score","text"]` and metadata keys `["case_id","court","date","section","title"]`.

Invalid input:

```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8000/v1/retrieval/vector -H "Content-Type: application/json" -d '{"query": "   "}'
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8000/v1/retrieval/vector -H "Content-Type: application/json" -d '{"query": "bail", "top_k": 21}'
```

Expected: `422` for both.

Empty results: a non-empty collection always returns its nearest chunks, so `total_results: 0` cannot be demonstrated against `judgments`. It is covered by the automated tests `test_no_matches_returns_empty_response` and `test_api_empty_result`.

Database not found (failure case): restart the server with a wrong path, e.g. `CHROMA_PERSIST_DIRECTORY=./data/missing uv run uvicorn graph_rag.main:app`, then repeat the successful curl with `-w "\n%{http_code}\n"` and no `jq`. Expected: `503` and `"detail":"ChromaDB database not found at '…missing'. Check CHROMA_PERSIST_DIRECTORY (see .env.example)."`. No `data/missing` directory is created. A wrong `CHROMA_COLLECTION_NAME` gives `503` with `… does not exist … No replacement collection is created.`

Ingestion: `POST /v1/ingestion/process` now only chunks documents (Story 1.2 behavior). It no longer embeds or stores, and the response no longer has `total_embeddings`, `chunks_stored` or `stale_chunks_deleted`.

## Story 1.3.2.2 — New Document Ingestion into ChromaDB

**What changes:** `POST /v1/ingestion/process` now also embeds every chunk (`all-MiniLM-L6-v2`, 384 dimensions) and upserts it into a second ChromaDB collection, `new_documents`, in the same `data/chroma/` database. `judgments` is never written. The request is unchanged (`include_chunks` query flag). The response adds `total_embeddings`, `records_upserted`, `obsolete_chunks_deleted` and `collection`, and each `documents[]` entry adds `embeddings`, `records_upserted` and `obsolete_chunks_deleted`. This replaces the "Ingestion … only chunks" note in the Story 1.3.2 section above.

**Prerequisites:**
- `uv sync` has been run.
- `data/chroma/chroma.sqlite3` exists (the `judgments` database). `new_documents` is created inside it on the first ingestion.
- The `sentence-transformers/all-MiniLM-L6-v2` model is cached locally or can be downloaded.
- Optional: back up the database first (`cp -r data/chroma data/chroma.bak`), because this story writes to it.

**Place documents:** copy `.txt`, `.md` or text-based `.pdf` files directly into `data/documents/` (or the folder set in `DOCUMENTS_DIRECTORY`). Sub-folders, hidden files and other extensions are ignored.

Check the starting state of both collections (developer command; prints counts and dimensions only):

```bash
uv run python -m graph_rag.vector_store.chroma_store                 # judgments: count=7325
uv run python -m graph_rag.vector_store.chroma_store new_documents   # fails with "does not exist" before the first ingestion
```

Start the server (in its own terminal):

```bash
uv run uvicorn graph_rag.main:app --reload
```

Trigger ingestion (counters only):

```bash
curl -s -X POST http://127.0.0.1:8000/v1/ingestion/process \
  | jq -c '{status, collection, documents_discovered, documents_processed, documents_failed, total_chunks, total_embeddings, records_upserted, obsolete_chunks_deleted}'
```

Expected: HTTP 200. `"collection":"new_documents"`. If every file is valid, `"status":"completed"` and `total_chunks == total_embeddings == records_upserted`. The first request is slower because it loads the model.

Per-document status:

```bash
curl -s -X POST http://127.0.0.1:8000/v1/ingestion/process | jq -c '.documents[]'
```

Expected: one entry per file with `source_file`, `document_id`, `status`, `chunks`, `embeddings`, `records_upserted` and `obsolete_chunks_deleted` (plus `error` for failures).

Idempotency: run the trigger command again. Expected: the same `records_upserted`, `obsolete_chunks_deleted: 0`, and the `new_documents` count (below) does not grow.

Changed document: shorten one of your files so it produces fewer chunks, then trigger again. Expected: that document's `obsolete_chunks_deleted` equals the number of chunks it lost. Other documents are unchanged.

Chunks without embeddings in the response:

```bash
curl -s -X POST "http://127.0.0.1:8000/v1/ingestion/process?include_chunks=true" \
  | jq -c '.chunks[] | {chunk_id, document_id, source_file, chunk_index, keys: (keys)}'
```

Expected: chunk keys `["chunk_id","chunk_index","document_id","file_type","metadata","source_file","text"]`, with no embedding.

Invalid document (edge case): put a corrupt PDF next to a valid file, then trigger again:

```bash
printf 'not a pdf' > data/documents/broken.pdf
curl -s -X POST http://127.0.0.1:8000/v1/ingestion/process | jq -c '{status, documents_failed, errors}'
rm data/documents/broken.pdf
```

Expected: HTTP 200, `"status":"completed_with_errors"`, `"documents_failed":1` and an `errors` entry for `broken.pdf`. The valid files are still stored.

Database not found (failure case): restart the server with `CHROMA_PERSIST_DIRECTORY=./data/missing uv run uvicorn graph_rag.main:app`, then:

```bash
curl -s -w "\n%{http_code}\n" -X POST http://127.0.0.1:8000/v1/ingestion/process
```

Expected: `503` with `"detail":"ChromaDB database not found at '…missing'. …"`. No `data/missing` directory is created. Similarly, setting `CHROMA_NEW_DOCUMENTS_COLLECTION=judgments` gives `503` with `… is read-only and cannot receive new documents …`.

Confirm what was stored (stop the server first, or run it from a second terminal):

```bash
uv run python -m graph_rag.vector_store.chroma_store new_documents
uv run python -m graph_rag.vector_store.chroma_store
```

Expected for `new_documents`: `count=` equals `records_upserted` from the latest run (as long as no files were renamed or removed), `space=cosine`, `has_text=True`, metadata fields `['chunk_index', 'document_id', 'end_char', 'file_type', 'ingested_at', 'source_file', 'start_char']` (plus `page_numbers` when a PDF was ingested) and `embedding_dimensions=[384, …] (expected 384)`. The second command still prints `collection=judgments count=7325`.

Note: `POST /v1/retrieval/vector` still searches only `judgments`, so newly ingested documents do not appear in its results yet.

## Story 1.5 — Legal Entity and Relationship Extraction

**What it does:** `POST /v1/extraction/process` reads chunks from the read-only `judgments` collection, sends each chunk's text to Groq and returns validated entities and relationships with `document_id`/`chunk_id` provenance. Nothing is written to ChromaDB or Neo4j.

**Prerequisites:**
- `data/chroma/chroma.sqlite3` exists (the `judgments` collection).
- `.env` sets `GROQ_API_KEY` and `GROQ_MODEL_NAME` (any Groq chat model with JSON mode, e.g. `openai/gpt-oss-120b`). Optional: `GROQ_TIMEOUT_SECONDS` (default 60), `EXTRACTION_MAX_CHUNKS_PER_DOCUMENT` (default 20).
- Each chunk is one Groq request, so document-level calls use your Groq quota and take several seconds per chunk.

Start the server (in its own terminal):

```bash
uv run uvicorn graph_rag.main:app --reload
```

**Request body:** `document_id` (required, the judgment's `case_id`), `chunk_id` (optional, a record ID such as `NI138-2019-0042_chunk0`), `max_chunks` (optional, 1–250, document-level only; default `EXTRACTION_MAX_CHUNKS_PER_DOCUMENT`).

**Response:** `document_id`, `extraction_status` (`completed` | `partial` | `failed`), `total_chunks`, `chunks_processed`, `chunks_succeeded`, `chunks_failed`, `chunks_skipped`, `total_entities`, `total_relationships`, and `results[]`. Each result has `document_id`, `chunk_id`, `extraction_status` (`completed` | `failed`), `error`, `warnings` (LLM items that were rejected), `entities[]` (`entity_id`, `name`, `normalized_name`, `entity_type`, `description`, `aliases`, `source_document_id`, `source_chunk_id`, `source_text`) and `relationships[]` (`relationship_id`, `source_entity_id`, `target_entity_id`, `relationship_type`, `description`, `evidence`, `source_document_id`, `source_chunk_id`).

Single-chunk extraction (counters and entity types only):

```bash
curl -s -X POST http://127.0.0.1:8000/v1/extraction/process \
  -H "Content-Type: application/json" \
  -d '{"document_id": "NI138-2019-0042", "chunk_id": "NI138-2019-0042_chunk0"}' \
  | jq -c '{extraction_status, total_chunks, total_entities, total_relationships, entities: [.results[0].entities[] | {entity_type, normalized_name}], warnings: .results[0].warnings}'
```

Expected: HTTP 200, `"extraction_status":"completed"`, `"total_chunks":1`, entities such as `PERSON Rajesh Kumar`, `LEGAL_SECTION Section 138`, `STATUTE Negotiable Instruments Act, 1881`. Exact output varies with the model.

Relationships and their evidence:

```bash
curl -s -X POST http://127.0.0.1:8000/v1/extraction/process \
  -H "Content-Type: application/json" \
  -d '{"document_id": "NI138-2019-0042", "chunk_id": "NI138-2019-0042_chunk0"}' \
  | jq -c '.results[0] | (reduce .entities[] as $e ({}; .[$e.entity_id] = $e.normalized_name)) as $names | .relationships[] | {relationship_type, source: $names[.source_entity_id], target: $names[.target_entity_id], evidence}'
```

Expected: `source` and `target` are never `null` (every relationship points at an entity of the same result), and every `evidence` is an exact phrase from the chunk, e.g. `RELATED_TO Section 138 → Negotiable Instruments Act, 1881`.

Document-level extraction (first 3 chunks; per-chunk status only):

```bash
curl -s -X POST http://127.0.0.1:8000/v1/extraction/process \
  -H "Content-Type: application/json" \
  -d '{"document_id": "NI138-2019-0042", "max_chunks": 3}' \
  | jq -c '{extraction_status, total_chunks, chunks_processed, chunks_succeeded, chunks_failed, chunks_skipped, per_chunk: [.results[] | {chunk_id, extraction_status, entities: (.entities|length), relationships: (.relationships|length), error}]}'
```

Expected: HTTP 200, chunks in reading order (`_chunk0`, `_chunk1`, `_chunk2`). This case has 4 chunks, so `"chunks_skipped":1` and `"extraction_status":"partial"`. With `"max_chunks": 4` and no failed chunk the status is `completed`. A chunk that failed (for example a Groq rate limit) shows `"extraction_status":"failed"` and an `error`, and the overall status is `partial` (or `failed` if no chunk succeeded).

Error scenarios:

```bash
curl -s -w "\n%{http_code}\n" -X POST http://127.0.0.1:8000/v1/extraction/process -H "Content-Type: application/json" -d '{"document_id": "DOES-NOT-EXIST"}'
curl -s -w "\n%{http_code}\n" -X POST http://127.0.0.1:8000/v1/extraction/process -H "Content-Type: application/json" -d '{"document_id": "NI138-2019-0042", "chunk_id": "NI138-2019-0042_chunk999"}'
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8000/v1/extraction/process -H "Content-Type: application/json" -d '{"document_id": "  "}'
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8000/v1/extraction/process -H "Content-Type: application/json" -d '{"document_id": "NI138-2019-0042", "max_chunks": 1000}'
```

Expected: `404` (`Document 'DOES-NOT-EXIST' has no chunks in the collection.`), `404` (`Chunk '…_chunk999' was not found.`), `422`, `422`. A `chunk_id` of another judgment gives `404` (`… does not belong to document …`).

Groq not configured: restart the server with `GROQ_API_KEY=` unset in `.env`, repeat any valid request. Expected: `503` with `Groq is not configured. Set GROQ_API_KEY and GROQ_MODEL_NAME …`. The key is never echoed.

Collection unchanged: `uv run python -m graph_rag.vector_store.chroma_store` still prints `count=7325`.

---

## Story 1.6 — Neo4j Knowledge Graph Construction

**What it does:** `POST /v1/graph/initialize` creates constraints/indexes; `POST /v1/graph/build` extracts a judgment's chunks (Story 1.5, Groq) and writes entities as nodes and relationships as edges to Neo4j with provenance.

**Prerequisites:**
- Everything from Story 1.5 (ChromaDB `judgments`, `GROQ_API_KEY`, `GROQ_MODEL_NAME`).
- A Neo4j 5.x server (local Neo4j Desktop or Aura). In `.env` (never commit real values):

```text
NEO4J_URI=neo4j+s://<your-instance>.databases.neo4j.io   # or neo4j://localhost:7687
NEO4J_USERNAME=<username>
NEO4J_PASSWORD=<password>
NEO4J_DATABASE=neo4j
```

These requests **write to the configured Neo4j database**. Use a database you are happy to fill with test data.

Start the server: `uv run uvicorn graph_rag.main:app --reload`

**Initialize the schema** (safe to repeat):

```bash
curl -s -X POST http://127.0.0.1:8000/v1/graph/initialize \
  | jq -c '{status, server_agent, constraints: (.constraints|length), indexes: (.indexes|length)}'
```

Expected: `{"status":"initialized","server_agent":"Neo4j/5.x…","constraints":11,"indexes":11}`. Calling it again returns the same result and creates nothing new. Without Neo4j settings: HTTP 503 with a message naming `NEO4J_URI`.

**Build the graph for a judgment** (request: `document_id`, optional `max_chunks` 1–250):

```bash
curl -s -X POST http://127.0.0.1:8000/v1/graph/build \
  -H "Content-Type: application/json" \
  -d '{"document_id": "NI138-2019-0042", "max_chunks": 4}' \
  | jq -c '{status, chunks_processed, chunks_failed, chunks_skipped, entities_extracted, nodes_created, nodes_merged, relationships_extracted, relationships_created, relationships_merged, failed: [.failed_chunks[] | {chunk_id, stage}]}'
```

Expected (exact counts vary with the model): HTTP 200, `"status":"completed"`, `"chunks_failed":0`, `nodes_created` > 0. With a smaller `max_chunks` the status is `partial` (chunks skipped). A Groq failure appears in `failed` with `stage: "extraction"`, a Neo4j write failure with `stage: "graph"`.

**Verify duplicate prevention:** run the same build request again. Expected: `nodes_created: 0`, `relationships_created: 0`, and `nodes_merged` / `relationships_merged` equal to the rows written (if the LLM answers identically; a different answer may add new items but never duplicates). Then in Neo4j Browser:

```cypher
MATCH (n:Entity) RETURN count(n) AS nodes, count(DISTINCT n.entity_id) AS unique_ids;
```

`nodes` must equal `unique_ids`.

**Error scenarios:**

```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8000/v1/graph/build -H "Content-Type: application/json" -d '{"document_id": "DOES-NOT-EXIST"}'   # 404
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8000/v1/graph/build -H "Content-Type: application/json" -d '{"document_id": "  "}'                # 422
```

**Inspect in Neo4j Browser** (open your instance, log in, run):

```cypher
// Node counts per type
MATCH (n:Entity) RETURN n.entity_type AS type, count(*) AS nodes ORDER BY nodes DESC;

// The graph of one judgment
MATCH (n:Entity)-[r]->(m:Entity)
WHERE 'NI138-2019-0042' IN n.source_document_ids
RETURN n, r, m LIMIT 50;

// Schema
SHOW CONSTRAINTS YIELD name, labelsOrTypes, properties;
SHOW INDEXES YIELD name, labelsOrTypes, properties;
```

**Verify provenance:**

```cypher
MATCH (n:Entity) RETURN labels(n), n.name, n.source_document_ids, n.source_chunk_ids LIMIT 10;
MATCH ()-[r]->() RETURN type(r), r.evidence, r.source_document_id, r.source_chunk_id LIMIT 10;
```

Every node has at least one chunk ID, every edge has `evidence` and one `source_chunk_id`, and each chunk ID (e.g. `NI138-2019-0042_chunk0`) exists in the ChromaDB `judgments` collection, where its full text lives.

---

## Story 1.6.2 — Full Corpus Graph Construction

Prerequisites: `.env` has `NEO4J_*`, `GROQ_API_KEY` and `GROQ_MODEL_NAME` (values never shown
here); server running with `uv run python -m graph_rag.main`. The ChromaDB `judgments`
collection is only read. Job state is written to `data/graph_build_jobs.sqlite3`.

Set a helper variable:

```bash
BASE=http://127.0.0.1:8000/v1/graph/build-all
```

**1. Discovery / dry run** (no writes, no Groq, no Neo4j):

```bash
curl -s $BASE/dry-run | jq '{discovery: .discovery | del(.invalid_record_ids), resumable_job_id, chunks_to_process}'
```

Expected (current corpus):

```json
{
  "discovery": {"total_chunk_records": 7325, "total_documents": 194, "records_missing_document_id": 0,
                "documents_eligible": 194, "chunks_eligible": 7325},
  "resumable_job_id": null,
  "chunks_to_process": 7325
}
```

**2. Start a full-corpus build** (returns at once, HTTP 202):

```bash
curl -s -X POST $BASE -H "Content-Type: application/json" \
  -d '{"batch_size": 5, "delay_seconds": 2, "max_retries": 3}' | jq '{job_id, status, resumed, chunks_to_process}'
```

```json
{"job_id": "3f2c…", "status": "running", "resumed": false, "chunks_to_process": 7325}
```

Save the ID: `JOB=3f2c…`. A second start while it runs returns `409`:

```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST $BASE -H "Content-Type: application/json" -d '{}'   # 409
```

**3–4. Job status and progress:**

```bash
curl -s $BASE/$JOB | jq '{status, documents_progress, chunks_progress, progress_percentage, chunks_attempted, chunks_extracted, chunks_failed, documents_partially_completed, nodes_created, nodes_merged, relationships_created, error}'
```

```json
{"status": "running", "documents_progress": "3/194", "chunks_progress": "88/7325",
 "progress_percentage": 1.2, "chunks_attempted": 89, "chunks_extracted": 89, "chunks_failed": 0,
 "documents_partially_completed": 0, "nodes_created": 512, "nodes_merged": 301,
 "relationships_created": 240, "error": null}
```

`chunks_processed` (and the percentage) only counts chunks that were extracted **and** committed
to Neo4j.

**5. Failed documents** (paginated):

```bash
curl -s "$BASE/$JOB/failures?offset=0&limit=5" | jq '{total_failed_documents, documents: [.documents[] | {document_id, status, chunks_failed, failed: [.failed_chunks[] | {chunk_id, stage, attempts}]}]}'
```

```json
{"total_failed_documents": 1,
 "documents": [{"document_id": "NI138-…", "status": "partially_completed", "chunks_failed": 1,
                "failed": [{"chunk_id": "NI138-…_chunk7", "stage": "extraction", "attempts": 1}]}]}
```

Show one error message: `curl -s "$BASE/$JOB/failures?limit=1" | jq -r '.documents[0].failed_chunks[0].error'`.

**6. Retry failed chunks** (only when the job is not running):

```bash
curl -s -X POST $BASE/$JOB/retry | jq '{status, chunks_to_process, message}'
```

```json
{"status": "running", "chunks_to_process": 1, "message": "Retrying 1 failed chunk(s); 1 chunk(s) pending in total."}
```

Graph-stage failures reuse their stored extraction (no Groq call); extraction-stage failures call
Groq again.

**7. Pause and resume an interrupted job:**

```bash
curl -s -X POST $BASE/$JOB/stop | jq '{message}'        # pauses after the current chunk
curl -s $BASE/$JOB | jq '{status, chunks_progress}'      # "paused"
```

Or stop the server (Ctrl+C) mid-run; after restarting, the job shows `"status": "interrupted"`.
Either way resume with the start request (`resume` defaults to true):

```bash
curl -s -X POST $BASE -H "Content-Type: application/json" -d '{"delay_seconds": 2}' | jq '{job_id, resumed, chunks_to_process}'
```

Expected: the same `job_id`, `"resumed": true`, and `chunks_to_process` equal to the previous
`chunks_pending`. `chunks_attempted` for completed chunks does not grow (they are skipped).

**8. Verify Neo4j graph data** (Neo4j Browser):

```cypher
MATCH (n:Entity) RETURN count(n) AS nodes, count(DISTINCT n.entity_id) AS unique_ids;
MATCH ()-[r]->() RETURN count(r) AS edges, count(DISTINCT r.relationship_id) AS unique_edges;
MATCH (n:Entity) UNWIND n.source_document_ids AS doc RETURN count(DISTINCT doc) AS judgments_in_graph;
```

`nodes = unique_ids`, `edges = unique_edges`, and `judgments_in_graph` approaches 194 as the job
progresses. No job/progress nodes exist: `MATCH (n) WHERE NOT n:Entity RETURN count(n)` → 0
(unless other data was already in the database).

**9. Repeated processing creates no duplicates.** Note the three counts above, then rebuild a
small amount again with a fresh job and stop it after a few chunks:

```bash
curl -s -X POST $BASE -H "Content-Type: application/json" -d '{"resume": false}' | jq -r .job_id   # → JOB2
curl -s -X POST $BASE/$JOB2/stop > /dev/null
curl -s $BASE/$JOB2 | jq '{chunks_processed, nodes_created, nodes_merged, relationships_created, relationships_merged}'
```

For chunks already in the graph, `nodes_created` and `relationships_created` stay 0 (identical
LLM answers merge everything; a differing answer may add new facts, never duplicates). Re-run the
Cypher queries: `nodes = unique_ids` and `edges = unique_edges` still hold. Note that
`resume: false` re-calls Groq for every chunk it processes.

**Error scenarios:**

```bash
curl -s -o /dev/null -w "%{http_code}\n" $BASE/does-not-exist                    # 404
curl -s -o /dev/null -w "%{http_code}\n" -X POST $BASE/does-not-exist/retry      # 404
curl -s -o /dev/null -w "%{http_code}\n" -X POST $BASE -H "Content-Type: application/json" -d '{"batch_size": 0}'   # 422
```

---

## Story 1.7 — Graph Retrieval

**What it does:** `POST /v1/retrieval/graph` identifies the entities named in a question, matches them to Neo4j nodes, walks at most `max_hops` edges and returns ranked paths with evidence (quote, `document_id`, `chunk_id` per edge). Read-only: no request changes the graph. No answer is generated.

**Prerequisites:** Neo4j settings from Story 1.6 and a graph built with Story 1.6 / 1.6.2. Groq is optional (used only when no entity pattern matches). Start the server: `uv run uvicorn graph_rag.main:app --reload`

Request fields: `question` (required), `max_hops` 1–3 (default 2), `limit` 1–100 (default 20), `entity_types` (filters the related nodes, e.g. `["CASE"]`), `relationship_types` (every edge must have one of these types).

```bash
BASE=http://127.0.0.1:8000/v1/retrieval/graph
SUMMARY='{retrieval_status, analysis_method, total_results, truncated, identified: [.identified_entities[] | {name, entity_type}], matched: [.matched_entities[] | {name, match_type, ambiguous}], unmatched: [.unmatched_entities[].name], errors}'
```

**One-hop example:**

```bash
curl -s -X POST $BASE -H "Content-Type: application/json" \
  -d '{"question": "Which courts applied Section 138 of the NI Act?", "max_hops": 1, "limit": 5}' | jq -c "$SUMMARY"
curl -s -X POST $BASE -H "Content-Type: application/json" \
  -d '{"question": "Which courts applied Section 138 of the NI Act?", "max_hops": 1, "limit": 5}' \
  | jq -c '.results[] | {rank, score, hop_distance, path, chunks: .source_chunk_ids, provenance_complete}'
```

Expected: `"retrieval_status":"success"` (or `partial` if one entity is not in the graph), `analysis_method: "deterministic"`, every result `hop_distance: 1`, `path` like `(Section 138:LEGAL_SECTION) <-[APPLIES]- (…)`, chunk IDs like `NI138-…_chunkN`.

**Two-hop example** (results from both levels; 1-hop paths rank first):

```bash
curl -s -X POST $BASE -H "Content-Type: application/json" \
  -d '{"question": "What is connected to the Negotiable Instruments Act, 1881?", "max_hops": 2, "limit": 10}' \
  | jq -c '{retrieval_status, total_results, hops: [.results[].hop_distance], top: .results[0].path}'
```

Expected: `hops` non-decreasing in most cases (ranking also weighs match type and support), 2-hop paths list both edges in `relationships`.

**Inspect evidence and provenance of one result** (quotes only, no chunk text):

```bash
curl -s -X POST $BASE -H "Content-Type: application/json" \
  -d '{"question": "Which courts applied Section 138?", "max_hops": 1, "limit": 1}' \
  | jq -c '.results[0].evidence[] | {relationship_type, evidence, document_id, chunk_id}'
```

Each `chunk_id` must exist in the ChromaDB `judgments` collection.

**Relationship and entity-type filtering:**

```bash
curl -s -X POST $BASE -H "Content-Type: application/json" \
  -d '{"question": "What is connected to Section 138?", "max_hops": 2, "relationship_types": ["APPLIES"]}' \
  | jq -c '[.results[].relationships[].relationship_type] | unique'          # ["APPLIES"]
curl -s -X POST $BASE -H "Content-Type: application/json" \
  -d '{"question": "What is connected to Section 138?", "max_hops": 2, "entity_types": ["CASE"]}' \
  | jq -c '[.results[].related_node.entity_type] | unique'                    # ["CASE"] or []
```

**Ambiguous match:** a name that exists in several judgments (people and sections are scoped per judgment) returns every candidate:

```bash
curl -s -X POST $BASE -H "Content-Type: application/json" \
  -d '{"question": "What is connected to Section 138?", "max_hops": 1}' | jq -c '{ambiguous_entities, matched: (.matched_entities | length), warnings}'
```

**No-match scenarios:**

```bash
curl -s -X POST $BASE -H "Content-Type: application/json" \
  -d '{"question": "Which cases cited the Foreign Exchange Management Act?"}' | jq -c "$SUMMARY"   # no_match, unmatched lists the Act
curl -s -X POST $BASE -H "Content-Type: application/json" \
  -d '{"question": "hello there"}' | jq -c '{retrieval_status, analysis_method, warnings}'           # no_match (may call Groq once)
```

A matched node without qualifying edges returns `"retrieval_status":"no_results"`.

**Validation errors (422):**

```bash
for body in '{"question": "  "}' '{"question": "IPC", "max_hops": 4}' '{"question": "IPC", "limit": 0}' '{"question": "IPC", "relationship_types": ["LIKES"]}'; do
  curl -s -o /dev/null -w "%{http_code}\n" -X POST $BASE -H "Content-Type: application/json" -d "$body"
done
```

**Neo4j unavailable:** stop the server, blank `NEO4J_URI` in `.env` (or point it at a closed port), restart, then:

```bash
curl -s -w "\n%{http_code}\n" -X POST $BASE -H "Content-Type: application/json" -d '{"question": "Which cases cited the IPC?"}'
```

Expected: HTTP 503 with a message naming `NEO4J_URI` (or "Cannot connect to Neo4j"); no credentials in the message. A Neo4j failure *after* connecting (e.g. a query timeout, `GRAPH_RETRIEVAL_QUERY_TIMEOUT_SECONDS`) returns 200 with `retrieval_status` `failed` or `partial` and the reason in `errors`. Restore `.env` afterwards.

---

## Story 1.8 — Hybrid Retrieval (Vector + Graph)

**What it does:** `POST /v1/retrieval/hybrid` runs vector retrieval (Story 1.4) and graph retrieval (Story 1.7) concurrently, resolves the graph evidence `(document_id, chunk_id)` pairs to original chunks in the ChromaDB `judgments` collection, merges duplicate chunks and ranks everything with Reciprocal Rank Fusion. It returns evidence only, never an answer, and writes nothing.

**Prerequisites:** the ChromaDB `judgments` collection (Story 1.3.2) and, for the graph branch, Neo4j with a built graph (Story 1.6 / 1.6.2). Groq is optional. Start the server: `uv run uvicorn graph_rag.main:app --reload`

Request fields (all limits optional; `null` uses `.env` defaults): `question`, `vector_top_k` 1–20, `graph_top_k` 1–100 (graph *paths*), `final_top_k` 1–100, `max_hops` 1–3, `entity_types`, `relationship_types` (graph branch only).

```bash
BASE=http://127.0.0.1:8000/v1/retrieval/hybrid
SUMMARY='{retrieval_status, total_results, vector_results_count, graph_results_count, merged_results_count, vector: .vector.status, graph: .graph.status, graph_status: .graph.graph_retrieval_status, unresolved: (.unresolved_references | length), errors}'
ROWS='.results[] | {rank, chunk_id, retrieval_sources, vector_rank, graph_rank, rrf_score}'
```

**Combined retrieval (and duplicate chunk merging):**

```bash
curl -s -X POST $BASE -H "Content-Type: application/json" \
  -d '{"question": "Which cases refer to Section 138 of the Negotiable Instruments Act?", "final_top_k": 5}' | jq -c "$SUMMARY"
curl -s -X POST $BASE -H "Content-Type: application/json" \
  -d '{"question": "Which cases refer to Section 138 of the Negotiable Instruments Act?", "final_top_k": 5}' | jq -c "$ROWS"
```

Expected: `"retrieval_status":"success"`, both counts > 0, `merged_results_count` ≤ `vector_results_count + graph_results_count` (equal minus the number of chunks found by both). A chunk found by both has `retrieval_sources: ["vector","graph"]`, both ranks and the highest `rrf_score` (≈ `1/(60+v) + 1/(60+g)`); it appears once. Results are sorted by `rrf_score` descending, `rank` 1..N.

**Graph context of one merged chunk** (quotes only, no chunk text):

```bash
curl -s -X POST $BASE -H "Content-Type: application/json" \
  -d '{"question": "Which cases refer to Section 138 of the Negotiable Instruments Act?"}' \
  | jq -c '[.results[] | select(.graph_rank != null)][0] | {chunk_id, paths: [.graph_context.paths[] | {graph_rank, path}], evidence: [.graph_context.evidence[] | {relationship_type, document_id, chunk_id}]}'
```

Every evidence `chunk_id` equals the result's `chunk_id`.

**Vector-only scenario** (no entity in the question, or an entity not in the graph):

```bash
curl -s -X POST $BASE -H "Content-Type: application/json" \
  -d '{"question": "dishonour of cheque due to insufficient funds"}' | jq -c "$SUMMARY"
```

Expected: `success`, `graph_status` `no_match`, `graph_results_count: 0`, every result `["vector"]`. (Without Groq the analysis stays deterministic; with Groq one call may be made.)

**Graph-only scenario:** set `vector_top_k` to 1 and look for results with `["graph"]` only, or simulate a vector outage (see below) — the graph chunks are still returned with `partial`.

**No-results scenario:** both branches empty is rare because vector search always returns its nearest chunks. It is covered by the automated tests (`test_both_branches_empty_is_no_results`).

**Partial failure / Neo4j unavailable:** stop the server, blank `NEO4J_URI` in `.env`, restart, then:

```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST $BASE -H "Content-Type: application/json" -d '{"question": "Which cases refer to Section 138?"}'
curl -s -X POST $BASE -H "Content-Type: application/json" -d '{"question": "Which cases refer to Section 138?"}' | jq -c "$SUMMARY"
```

Expected: HTTP 200 (not 503), `"retrieval_status":"partial"`, `graph: "failed"`, `errors: ["Graph retrieval failed: Neo4j is not configured."]`, vector results present. Restore `.env` afterwards.

**ChromaDB unavailable:** stop the server, set `CHROMA_PERSIST_DIRECTORY=./data/missing` in `.env`, restart, run the combined request. Expected: vector `failed` with "ChromaDB is unavailable", graph references listed in `unresolved_references` with reason `lookup_failed`, `retrieval_status` `failed` (nothing usable, no text invented). Restore `.env` afterwards.

**Validation errors (422):**

```bash
for body in '{"question": " "}' '{"question": "IPC", "vector_top_k": 21}' '{"question": "IPC", "final_top_k": 0}' '{"question": "IPC", "max_hops": 4}' '{"question": "IPC", "entity_types": ["X"]}'; do
  curl -s -o /dev/null -w "%{http_code}\n" -X POST $BASE -H "Content-Type: application/json" -d "$body"
done
```

---

## Story 1.9 — LLM-Based Answer Generation

**What it does:** `POST /v1/generation/answer` calls Story 1.8 hybrid retrieval in-process, builds `[SOURCE n]` context from the ranked chunks, asks Groq for a JSON answer with per-claim citations, removes citations that do not match the evidence and returns answer, claims, sources and diagnostics.

**Prerequisites:** everything Story 1.8 needs, plus `GROQ_API_KEY` and `GROQ_MODEL_NAME` in `.env` (never paste the key into commands or docs). Start the server: `uv run uvicorn graph_rag.main:app --reload`

Request: `question` (1–2000 chars), `top_k` 1–100 (Story 1.8 `final_top_k`), `max_hops` 1–3, `include_sources` (default `true`).

```bash
BASE=http://127.0.0.1:8000/v1/generation/answer
SUMMARY='{status, retrieval_status: .retrieval_summary.retrieval_status, evidence_used: .retrieval_summary.evidence_used, claims: (.claims | length), invalid: (.generation_metadata.invalid_citations | length), rejected: (.generation_metadata.rejected_claims | length), warnings, errors}'
```

**1. Successful answer generation:**

```bash
curl -s -X POST $BASE -H "Content-Type: application/json" \
  -d '{"question": "What are the important judgments discussing Section 138 of the Negotiable Instruments Act?", "top_k": 10}' | jq -c "$SUMMARY"
curl -s -X POST $BASE -H "Content-Type: application/json" \
  -d '{"question": "What are the important judgments discussing Section 138 of the Negotiable Instruments Act?", "top_k": 10}' | jq -r '.answer'
```

Expected: `"status":"success"`, `claims` ≥ 1, `errors: []`; the answer contains inline `[document_id, chunk_id]` citations; `generation_metadata.llm_called` and `citations_validated` are `true`.

**2. Question with multiple relevant judgments:**

```bash
curl -s -X POST $BASE -H "Content-Type: application/json" \
  -d '{"question": "How have courts treated the requirement of a legally enforceable debt in cheque dishonour cases?"}' \
  | jq -c '[.claims[].citations[].document_id] | unique'
```

Expected: more than one distinct `document_id`; some claims carry several citations.

**3. Question with no evidence:** vector search always returns nearest chunks, so true `no_results` is rare (covered by `test_no_evidence_skips_llm`). Ask an out-of-corpus question:

```bash
curl -s -X POST $BASE -H "Content-Type: application/json" \
  -d '{"question": "What is the capital gains tax rate on cryptocurrency in Japan?"}' | jq -c "$SUMMARY"
```

Expected: `"status":"insufficient_evidence"`, an answer stating the sources do not cover it, no invented facts.

**4. Partial retrieval:** stop the server, blank `NEO4J_URI` in `.env`, restart, repeat test 1. Expected: `"status":"partial"`, `retrieval_status: "partial"`, a warning that retrieval was partial, `.retrieval_summary.errors` containing "Graph retrieval failed: Neo4j is not configured.". Restore `.env`.

**5. Invalid citation handling:** cannot be forced against the real model; covered by `test_invalid_citations_are_removed_and_reported` and `test_answer_with_no_valid_claim_is_not_returned`. On any real response confirm:

```bash
curl -s -X POST $BASE -H "Content-Type: application/json" \
  -d '{"question": "Which cases refer to Section 138 of the NI Act?"}' | jq -c '.generation_metadata | {invalid_citations, rejected_claims}'
```

Expected: usually empty lists; any entry has a `reason` and the matching citation is absent from `claims` and `answer`.

**6. Groq failure:** stop the server, blank `GROQ_API_KEY` in `.env`, restart, repeat test 1. Expected: HTTP 200, `"status":"generation_failed"`, `answer: null`, `errors: ["Answer generation failed: Groq is not configured."]`, `retrieval_summary` still populated. Restore `.env`. (Timeouts and rate limits are covered by `test_groq_failures_return_no_answer`.)

**7. Retrieval failure:** stop the server, set `CHROMA_PERSIST_DIRECTORY=./data/missing` and blank `NEO4J_URI` in `.env`, restart, repeat test 1. Expected: `"status":"retrieval_failed"`, `answer: null`, `llm_called: false`. Restore `.env`.

**8. Source citation verification:** every cited pair must be a returned source with `cited: true`:

```bash
curl -s -X POST $BASE -H "Content-Type: application/json" \
  -d '{"question": "Which cases refer to Section 138 of the NI Act?"}' \
  | jq -c '([.sources[] | [.document_id, .chunk_id]]) as $s | [.claims[].citations[] | [.document_id, .chunk_id] | IN($s[])] | all'
curl -s -X POST $BASE -H "Content-Type: application/json" \
  -d '{"question": "Which cases refer to Section 138 of the NI Act?"}' \
  | jq -c '.sources[] | {source_number, document_id, chunk_id, rank, retrieval_source, cited}'
```

Expected: `true`; sources in Story 1.8 rank order. To read one cited chunk's text: `jq -r '.sources[] | select(.cited) | .content' | head -20`.

**Validation errors (422):**

```bash
for body in '{"question": " "}' '{"question": "IPC", "top_k": 0}' '{"question": "IPC", "top_k": 101}' '{"question": "IPC", "max_hops": 4}'; do
  curl -s -o /dev/null -w "%{http_code}\n" -X POST $BASE -H "Content-Type: application/json" -d "$body"
done
```

Expected: `422` four times.

---

## Story 1.9.1 — Hybrid Retrieval Explainability

**What it does:** `POST /v1/generation/answer` now also returns `retrieval_details`: vector, graph and hybrid (RRF) results, graph-only chunks, graph paths/entities/relationships, a fusion summary and claim-to-source mapping, all taken from the single retrieval run behind the answer. Request schema is unchanged.

**Prerequisites:** as Story 1.9. Optional limits in `.env`: `RETRIEVAL_DETAILS_MAX_ITEMS=50`, `RETRIEVAL_DETAILS_PREVIEW_CHARS=400`. Start the server: `uv run uvicorn graph_rag.main:app --reload`

```bash
BASE=http://127.0.0.1:8000/v1/generation/answer
Q='{"question": "Which cases refer to Section 138 of the Negotiable Instruments Act?", "top_k": 10}'
```

**1. Successful answer request + fusion summary:**

```bash
curl -s -X POST $BASE -H "Content-Type: application/json" -d "$Q" \
  | jq -c '{status, d: .retrieval_details | {vector_status, graph_status, graph_retrieval_status, fusion_summary, truncated_lists}}'
```

Expected: `status` `success`; both branch statuses `completed`; `vector_only_count + graph_only_count + both_count == hybrid_count`; `overlap_count == both_count`; `final_results_count` ≤ `top_k`. Example `fusion_summary`: `{"vector_count":10,"graph_count":14,"hybrid_count":21,"overlap_count":3,"vector_only_count":7,"graph_only_count":11,"both_count":3,"final_results_count":10,"rrf_k":60}` (numbers depend on the corpus).

**2. Inspect hybrid results (IDs, ranks, scores, origin — no text):**

```bash
curl -s -X POST $BASE -H "Content-Type: application/json" -d "$Q" \
  | jq -c '.retrieval_details.hybrid_results[] | {hybrid_rank, document_id, chunk_id, retrieval_origin, rrf_score, vector_rank, graph_rank, in_final_results}'
```

Expected: `hybrid_rank` 1, 2, 3…; `rrf_score` non-increasing; `retrieval_origin` (`vector`, `graph`, `both`) matches which ranks are set; `in_final_results` true for the first `final_results_count` rows; these ranks equal `sources[].rank`.

**3. Compare vector and graph results:**

```bash
curl -s -X POST $BASE -H "Content-Type: application/json" -d "$Q" \
  | jq -c '.retrieval_details | {vector: [.vector_results[] | [.vector_rank, .chunk_id, .vector_score]], graph: [.graph_results[] | [.graph_rank, .chunk_id]]}'
```

Expected: vector rows ordered by `vector_rank` with cosine scores; graph rows ordered by `graph_rank`; a chunk present in both lists has origin `both`.

**4. Inspect graph-only chunks and the graph behind them:**

```bash
curl -s -X POST $BASE -H "Content-Type: application/json" -d "$Q" \
  | jq -c '.retrieval_details.graph_only_chunks[] | {chunk_id, graph_rank, paths: [.graph_context.paths[].path], rels: [.graph_context.relationships[] | [.relationship_type, .source_document_id, .source_chunk_id]]}'
curl -s -X POST $BASE -H "Content-Type: application/json" -d "$Q" \
  | jq -c '.retrieval_details | {entities: [.graph_entities[].name], rels: (.graph_relationships | length), paths: (.graph_paths | length), matched: [.matched_entities[].name], unresolved: (.unresolved_references | length)}'
```

Expected: every graph-only chunk has `vector_rank: null` and at least one path; relationship provenance names real `(document_id, chunk_id)` pairs. Graph-only chunks are additional evidence candidates, not a measured answer-quality improvement.

**5. Inspect claim citations:**

```bash
curl -s -X POST $BASE -H "Content-Type: application/json" -d "$Q" \
  | jq -c '.retrieval_details.claim_citations[] | {supported, rejection_reason, cites: [.citations[] | [.source_number, .chunk_id, .retrieval_origin]]}'
```

Expected: supported claims list only citations present in top-level `claims`; each `source_number` matches a `sources[]` entry; a rejected claim shows `supported: false`, no citations and a `rejection_reason`.

**6. No-evidence scenario:**

```bash
curl -s -X POST $BASE -H "Content-Type: application/json" \
  -d '{"question": "What is the capital gains tax rate on cryptocurrency in Japan?"}' \
  | jq -c '{status, llm: .generation_metadata.llm_called, s: .retrieval_details.fusion_summary, claims: (.retrieval_details.claim_citations | length)}'
```

Expected: `insufficient_evidence`. Vector search always returns nearest chunks, so counts may be non-zero. A fully empty retrieval (all counts 0, `llm_called: false`) is covered by `test_no_evidence_skips_llm_and_reports_empty_summary`.

**7. Generation-failure scenario:** stop the server, blank `GROQ_API_KEY` in `.env`, restart, repeat test 1. Expected: `generation_failed`, `answer: null`, `retrieval_details` still populated (same kind of `fusion_summary` as test 1), `claim_citations: []`. Restore `.env`.

**8. Partial retrieval:** blank `NEO4J_URI`, restart, repeat test 1. Expected: `graph_status: "failed"`, empty `graph_results` and `graph_paths`, every hybrid origin `vector`. Restore `.env`.

**9. Hybrid endpoint unchanged:**

```bash
curl -s -X POST http://127.0.0.1:8000/v1/retrieval/hybrid -H "Content-Type: application/json" \
  -d '{"question": "Section 138 NI Act"}' | jq -c '[has("ranked_results"), has("graph_paths")]'
```

Expected: `[false,false]`.

---

## Story 1.10 — Graph RAG Research Frontend (Streamlit)

**Prerequisites:** `uv sync` (installs the `frontend` group with Streamlit), backend configured as for Story 1.9 (Groq key, ChromaDB data, Neo4j graph). Optional `.env`: `GRAPH_RAG_API_BASE_URL=http://localhost:8000`. Run all commands from the project root.

**1. Start FastAPI** (terminal 1):

```bash
uv run uvicorn graph_rag.main:app --reload
```

**2. Start Streamlit** (terminal 2):

```bash
uv run streamlit run frontend/app.py
```

Open http://localhost:8501.

**3. Backend connection check:** the header shows a green *Backend connected* pill and the sidebar shows *Connected to Graph RAG POC*. Cross-check: `curl -s http://localhost:8000/health`.

**4. Ask a legal question:** type e.g. *Which cases refer to Section 138 of the Negotiable Instruments Act?* in the chat box. Expected: the question appears, a spinner shows while the backend works, then the answer appears below it. Only one request reaches `POST /v1/generation/answer` (visible in the uvicorn log).

**5. Inspect the answer:** a status banner (*Answer generated*, *Partial answer*, …), the answer text with inline `[DOC_ID, CHUNK_ID]` citations, a *Validated citations* caption and a metadata line (retrieval status, evidence used, claims accepted, model). Backend warnings, if any, are in the *Backend warnings and errors* expander.

**6. Vector evidence:** tab *Retrieval evidence* → *A. Vector retrieval*. The count equals `retrieval_details.fusion_summary.vector_count`; the table lists vector rank, document/chunk IDs, similarity and distance; each chunk expander shows its text (full text if it was in the model context, otherwise a preview marked as such).

**7. Graph evidence:** same tab → *B. Graph retrieval*. Each row shows the graph rank, origin and the best graph path; chunk expanders show the path that reached the chunk. If the graph found nothing, an info message says so.

**8. Entities and relationships:** tab *Entities & relationships*: matched question entities (normalized name, match type, source documents), path entities grouped by type, a relationship table (source → type → target, evidence, source document/chunk) and a graph view with matched entities in blue. Compare with:

```bash
curl -s -X POST http://localhost:8000/v1/generation/answer -H "Content-Type: application/json" \
  -d '{"question": "Which cases refer to Section 138 of the Negotiable Instruments Act?"}' \
  | jq -c '.retrieval_details | {entities: (.graph_entities | length), rels: (.graph_relationships | length), matched: [.matched_entities[].name]}'
```

**9. RRF results:** *C. Hybrid results* shows merged / vector-only / graph-only / both counts, duplicates merged, the RRF formula with the backend `rrf_k`, and the hybrid table ordered by hybrid rank with non-increasing RRF scores. Tab *How Graph Retrieval Contributed* repeats the counts and lists observations (e.g. graph-only chunks and the relationships that reached them, claims that cite graph-only chunks). It must never claim an answer-quality improvement.

**10. Citations:** tab *Sources & citations*: supported claims with `[SOURCE n] DOC · CHUNK (origin)`, rejected claims struck through with the reason, and one expander per cited passage with the original chunk text; uncited evidence given to the model is listed separately.

**11. Insufficient evidence:** ask an off-corpus question, e.g. *What is the capital of Australia?* Expected: an *Insufficient evidence* (or *Generation failed*) banner with the status explanation and no answer text; the evidence tabs still show what was retrieved.

**12. Backend unavailable:** stop uvicorn (Ctrl+C) and refresh the page. Expected: red *Backend unavailable* pill and an error naming the API URL. Asking a question shows *Cannot reach the backend…* in the assistant message; nothing is retried. Restart uvicorn and refresh to reconnect.

**13. New chat:** click *New Chat* in the sidebar. Expected: the conversation is empty, `top_k` returns to its default and the session ID changes. *Clear Chat* empties the messages but keeps the `top_k` setting.
