"""Story 1.6.2 tests: full-corpus graph construction.

ChromaDB, Groq and Neo4j are all faked: ``FakeCollection`` pages through in-memory records,
``FakeLLM`` returns a canned grounded answer (or scripted errors), and Neo4j is the in-memory
``FakeDriver`` from the Story 1.6 tests behind the real ``Neo4jClient``. Job state goes to a
temporary SQLite file per test.
"""

import json
import threading
from types import SimpleNamespace

import groq
import httpx
import pytest
from fastapi.testclient import TestClient

from graph_rag.core.llm import GroqJSONClient, LLMError
from graph_rag.graph.build_job_store import BuildJobStore
from graph_rag.graph.corpus_builder import BuildRunner, BuildStopped, ThrottledLLM, run_job
from graph_rag.graph.corpus_discovery import discover_corpus, document_id_of
from graph_rag.graph.corpus_models import BuildOptions
from graph_rag.graph.corpus_router import get_build_runner, get_job_store
from graph_rag.graph.graph_router import get_graph_client
from graph_rag.graph.neo4j_client import GraphStoreError, Neo4jClient
from graph_rag.graph.router import get_extraction_llm
from graph_rag.main import app
from graph_rag.retrieval.router import get_chroma_collection
from test_graph_construction import FakeDriver

ANSWER = json.dumps(
    {
        "entities": [
            {
                "id": "e1",
                "name": "Rajesh Kumar",
                "entity_type": "PERSON",
                "evidence": "Rajesh Kumar appeared",
            },
            {
                "id": "e2",
                "name": "Delhi High Court",
                "entity_type": "COURT",
                "evidence": "Delhi High Court",
            },
        ],
        "relationships": [
            {
                "source": "e1",
                "target": "e2",
                "relationship_type": "RELATED_TO",
                "evidence": "Rajesh Kumar appeared before the Delhi High Court",
            }
        ],
    }
)


def text(doc: str, n: int) -> str:
    return f"Rajesh Kumar appeared before the Delhi High Court in {doc} part {n}."


def corpus(sizes: dict[str, int], invalid: int = 0) -> list[tuple[str, str, dict]]:
    records = [
        (f"{doc}_chunk{n}", text(doc, n), {"case_id": doc, "title": "t"})
        for doc, size in sizes.items()
        for n in range(size)
    ]
    records += [(f"orphan_{i}", "no id", {"title": "t"}) for i in range(invalid)]
    return records


class FakeCollection:
    name = "judgments"

    def __init__(self, records):
        self.records = records
        self.calls: list[dict] = []

    def get(self, ids=None, where=None, limit=None, offset=None, include=()):
        self.calls.append({"limit": limit, "offset": offset, "include": list(include)})
        rows = self.records
        if ids is not None:
            rows = [row for row in rows if row[0] in ids]
        if where:
            ((key, value),) = where.items()
            rows = [row for row in rows if row[2].get(key) == value]
        rows = rows[offset or 0 :]
        if limit is not None:
            rows = rows[:limit]
        result = {"ids": [row[0] for row in rows]}
        if "documents" in include:
            result["documents"] = [row[1] for row in rows]
        if "metadatas" in include:
            result["metadatas"] = [row[2] for row in rows]
        return result


class FakeLLM:
    """Canned answer; ``errors[marker]`` is a list of exceptions raised for matching chunks."""

    def __init__(self, errors=None, on_call=None):
        self.errors = {key: list(value) for key, value in (errors or {}).items()}
        self.on_call = on_call
        self.calls = 0

    def complete_json(self, system_prompt, user_prompt):
        self.calls += 1
        if self.on_call:
            self.on_call(self.calls)
        for marker, queue in self.errors.items():
            if marker in user_prompt and queue:
                raise queue.pop(0)
        return ANSWER


class FlakyGraphClient(Neo4jClient):
    """Real Neo4jClient over FakeDriver; writes for chunks in ``fail_chunks`` raise."""

    def __init__(self, driver, fail_chunks=()):
        super().__init__(driver, "neo4j")
        self.schema_ready = True
        self.fail_chunks = set(fail_chunks)

    def execute_write(self, work, **kwargs):
        chunk_ids = {row["chunk_id"] for rows in kwargs["nodes"].values() for row in rows}
        if chunk_ids & self.fail_chunks:
            raise GraphStoreError("Neo4j write failed (ServiceUnavailable: down).")
        return super().execute_write(work, **kwargs)


OPTIONS = BuildOptions(
    batch_size=2,
    delay_seconds=0,
    max_retries=2,
    retry_delay_seconds=0,
    max_consecutive_failures=5,
)


@pytest.fixture
def store(tmp_path):
    return BuildJobStore(tmp_path / "jobs.sqlite3")


def new_job(store, collection, options=OPTIONS):
    return store.create_job(discover_corpus(collection, page_size=3), options)


def run(store, job_id, collection, llm, client=None, stop=None):
    return run_job(
        store,
        job_id,
        collection=collection,
        client=client or FlakyGraphClient(FakeDriver()),
        llm=llm,
        stop=stop or threading.Event(),
    )


def status(store, job_id):
    return store.status(job_id)


# --- 1-4. Discovery -------------------------------------------------------------------


def test_discovery_pages_metadata_only_and_groups_by_case_id():
    records = corpus({"B-2": 11, "A-1": 2}, invalid=1)
    records.append(("blank_1", "x", {"case_id": "  "}))
    collection = FakeCollection(records)

    discovery = discover_corpus(collection, page_size=4)

    assert discovery.total_chunk_records == 15
    assert discovery.total_documents == 2  # chunk count != document count
    assert list(discovery.documents) == ["A-1", "B-2"]  # deterministic order
    assert discovery.documents["B-2"][-2:] == ["B-2_chunk9", "B-2_chunk10"]  # numeric order
    assert discovery.invalid_record_ids == ["blank_1", "orphan_0"]
    assert discovery.eligible_chunks == 13
    assert [call["offset"] for call in collection.calls] == [0, 4, 8, 12]  # paged to the end
    assert all(call["include"] == ["metadatas"] for call in collection.calls)  # no text/vectors


def test_document_id_comes_from_case_id_not_document_id():
    assert document_id_of({"case_id": "NI-1"}) == "NI-1"
    assert document_id_of({"case_id": 42}) == "42"
    assert document_id_of({"document_id": "X"}) is None  # judgments has no document_id field
    assert document_id_of(None) is None


# --- 5-8, 20. Full document pipeline --------------------------------------------------


def test_job_processes_every_chunk_without_the_four_chunk_limit(store):
    collection = FakeCollection(corpus({"A-1": 7, "B-2": 2}))
    llm = FakeLLM()
    driver = FakeDriver()
    job_id = new_job(store, collection)

    assert run(store, job_id, collection, llm, FlakyGraphClient(driver)) == "completed"

    data = status(store, job_id)
    assert llm.calls == 9  # every chunk, not max_chunks=4
    assert data["documents"]["completed"] == 2
    assert data["chunks"]["processed"] == data["chunks"]["total"] == 9
    assert data["chunks"]["entities"] == 18 and data["chunks"]["relationships"] == 9
    # PERSON is document-scoped, COURT global: 2 people + 1 court; one edge per chunk
    assert len(driver.nodes) == 3 and len(driver.edges) == 9
    assert data["chunks"]["nodes_created"] == 3
    assert data["chunks"]["nodes_merged"] == 18 - 3
    assert {c for n in driver.nodes.values() for c in n["source_chunk_ids"]} >= {"A-1_chunk6"}


# --- 9-11. Failures -------------------------------------------------------------------


def test_extraction_failure_gives_partially_completed_document(store):
    collection = FakeCollection(corpus({"A-1": 3}))
    llm = FakeLLM({"A-1 part 1": [LLMError("bad request")]})  # permanent: not retried
    job_id = new_job(store, collection)

    assert run(store, job_id, collection, llm) == "completed_with_failures"

    data = status(store, job_id)
    assert llm.calls == 3
    assert data["documents"]["partial"] == 1 and data["documents"]["completed"] == 0
    assert data["chunks"]["processed"] == 2 and data["chunks"]["failed"] == 1
    total, failures = store.failures(job_id, 0, 10)
    assert total == 1
    assert failures[0].failed_chunks[0].stage == "extraction"
    assert failures[0].failed_chunks[0].chunk_id == "A-1_chunk1"


def test_graph_failure_is_not_counted_as_processed_and_retry_reuses_extraction(store):
    collection = FakeCollection(corpus({"A-1": 2}))
    llm = FakeLLM()
    job_id = new_job(store, collection)
    run(store, job_id, collection, llm, FlakyGraphClient(FakeDriver(), {"A-1_chunk0"}))

    data = status(store, job_id)
    assert data["chunks"]["extracted"] == 2  # extraction succeeded for both ...
    assert data["chunks"]["processed"] == 1  # ... but only one was persisted
    _, failures = store.failures(job_id, 0, 10)
    assert failures[0].failed_chunks[0].stage == "graph"

    assert store.reset_failures(job_id) == 1
    assert run(store, job_id, collection, llm) == "completed"
    assert llm.calls == 2  # the graph retry used the stored extraction, no new Groq call
    assert status(store, job_id)["chunks"]["processed"] == 2


def test_document_failed_when_every_chunk_fails(store):
    collection = FakeCollection(corpus({"A-1": 2}))
    llm = FakeLLM({"A-1": [LLMError("x"), LLMError("y")]})
    job_id = new_job(store, collection)
    run(store, job_id, collection, llm)
    assert status(store, job_id)["documents"]["failed"] == 1


# --- 12-14. Persistence and resume ----------------------------------------------------


def test_resume_after_interruption_skips_completed_chunks(store, tmp_path):
    collection = FakeCollection(corpus({"A-1": 3, "B-2": 3}))
    stop = threading.Event()
    llm = FakeLLM(on_call=lambda n: stop.set() if n == 2 else None)
    job_id = new_job(store, collection)

    assert run(store, job_id, collection, llm, stop=stop) == "paused"
    assert status(store, job_id)["chunks"]["processed"] == 2

    reopened = BuildJobStore(tmp_path / "jobs.sqlite3")  # simulated restart: state on disk
    assert reopened.status(job_id)["chunks"]["processed"] == 2
    reopened.sync_job(job_id, discover_corpus(collection), OPTIONS)
    resumed_llm = FakeLLM()
    assert run(reopened, job_id, collection, resumed_llm) == "completed"
    assert resumed_llm.calls == 4  # only the 4 remaining chunks
    assert reopened.status(job_id)["chunks"]["processed"] == 6


def test_recover_marks_running_jobs_interrupted(store):
    collection = FakeCollection(corpus({"A-1": 1}))
    job_id = new_job(store, collection)
    store.set_job_status(job_id, "running")
    store.mark_document_processing(job_id, "A-1")

    assert store.recover_interrupted_jobs() == [job_id]
    assert store.get_job(job_id)["status"] == "interrupted"
    assert store.pending_documents(job_id, 10) == ["A-1"]


# --- 17. Idempotency ------------------------------------------------------------------


def test_rebuilding_the_corpus_creates_no_duplicates(store):
    collection = FakeCollection(corpus({"A-1": 2, "B-2": 2}))
    driver = FakeDriver()
    first = new_job(store, collection)
    run(store, first, collection, FakeLLM(), FlakyGraphClient(driver))
    nodes, edges = len(driver.nodes), len(driver.edges)

    second = new_job(store, collection)  # resume: false -> a fresh job over the same graph
    run(store, second, collection, FakeLLM(), FlakyGraphClient(driver))

    assert (len(driver.nodes), len(driver.edges)) == (nodes, edges)
    again = status(store, second)["chunks"]
    assert again["nodes_created"] == 0 and again["relationships_created"] == 0
    assert again["nodes_merged"] > 0 and again["relationships_merged"] == edges


# --- 18-19. Rate limits and retries ---------------------------------------------------


def throttled(llm, waits, **overrides):
    options = {"delay_seconds": 0, "max_retries": 2, "retry_delay_seconds": 5}
    options.update(overrides)
    return ThrottledLLM(llm, wait=lambda s: waits.append(s) or False, **options)


def test_rate_limit_is_retried_with_exponential_backoff():
    waits: list[float] = []
    llm = FakeLLM({"p": [LLMError("429", retryable=True), LLMError("429", retryable=True)]})
    assert throttled(llm, waits).complete_json("s", "p") == ANSWER
    assert waits == [5, 10] and llm.calls == 3


def test_retry_limit_and_permanent_errors():
    waits: list[float] = []
    llm = FakeLLM({"p": [LLMError("429", retryable=True, retry_after=1.5)] * 5})
    client = throttled(llm, waits)
    with pytest.raises(LLMError, match="Gave up after 3"):
        client.complete_json("s", "p")
    assert llm.calls == 3 and waits == [1.5, 1.5]  # server's retry-after wins
    assert client.last_error_transient

    permanent = FakeLLM({"p": [LLMError("bad json")]})
    with pytest.raises(LLMError):
        throttled(permanent, waits).complete_json("s", "p")
    assert permanent.calls == 1


def test_requests_are_paced():
    waits: list[float] = []
    clock = iter([0.0, 0.5, 0.5]).__next__
    llm = ThrottledLLM(
        FakeLLM(),
        delay_seconds=2,
        max_retries=0,
        retry_delay_seconds=0,
        wait=lambda s: waits.append(s) or False,
        clock=clock,
    )
    llm.complete_json("s", "a")
    llm.complete_json("s", "b")
    assert waits == [1.5]


def test_groq_429_is_marked_retryable():
    response = httpx.Response(
        429, headers={"retry-after": "7"}, request=httpx.Request("POST", "https://x")
    )
    error = groq.RateLimitError("rate limited", response=response, body=None)

    class Raising:
        def create(self, **_):
            raise error

    fake = SimpleNamespace(chat=SimpleNamespace(completions=Raising()))
    client = GroqJSONClient("k", "m", 1, client=fake)
    with pytest.raises(LLMError) as caught:
        client.complete_json("s", "u")
    assert caught.value.retryable and caught.value.retry_after == 7.0


def test_stop_during_backoff_leaves_chunk_pending():
    llm = FakeLLM({"p": [LLMError("429", retryable=True)]})
    client = ThrottledLLM(
        llm, delay_seconds=0, max_retries=3, retry_delay_seconds=5, wait=lambda s: True
    )
    with pytest.raises(BuildStopped):
        client.complete_json("s", "p")


def test_consecutive_transient_failures_interrupt_job(store):
    collection = FakeCollection(corpus({"A-1": 6}))
    options = OPTIONS.model_copy(update={"max_retries": 0, "max_consecutive_failures": 3})
    llm = FakeLLM({"A-1": [LLMError("503", retryable=True)] * 6})
    job_id = new_job(store, collection, options)

    assert run(store, job_id, collection, llm) == "interrupted"
    assert llm.calls == 3  # stopped instead of burning through the corpus
    assert "in a row" in store.get_job(job_id)["error"]
    assert status(store, job_id)["chunks"]["pending"] == 3


# --- 16, 21-24. API -------------------------------------------------------------------


@pytest.fixture
def api(store):
    state = {
        "collection": FakeCollection(corpus({"A-1": 2, "B-2": 3})),
        "llm": FakeLLM(),
        "client": FlakyGraphClient(FakeDriver()),
        "runner": BuildRunner(),
    }
    app.dependency_overrides.update(
        {
            get_job_store: lambda: store,
            get_build_runner: lambda: state["runner"],
            get_chroma_collection: lambda: state["collection"],
            get_graph_client: lambda: state["client"],
            get_extraction_llm: lambda: state["llm"],
        }
    )
    yield TestClient(app), state
    state["runner"].shutdown(5)
    app.dependency_overrides.clear()


def finish(state):
    state["runner"]._thread.join(5)


def test_dry_run_reports_counts_without_creating_a_job(api, store):
    http, state = api
    body = http.get("/v1/graph/build-all/dry-run").json()
    assert body["discovery"]["total_chunk_records"] == 5
    assert body["discovery"]["total_documents"] == 2
    assert body["chunks_to_process"] == 5 and body["resumable_job_id"] is None
    assert store.latest_job_id() is None and state["llm"].calls == 0


def test_start_build_and_status_api(api):
    http, state = api
    response = http.post("/v1/graph/build-all", json={"delay_seconds": 0, "batch_size": 1})
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    assert response.json()["chunks_to_process"] == 5
    finish(state)

    body = http.get(f"/v1/graph/build-all/{job_id}").json()
    assert body["status"] == "completed" and body["progress_percentage"] == 100.0
    assert body["documents_progress"] == "2/2" and body["chunks_progress"] == "5/5"
    assert body["options"]["batch_size"] == 1 and body["completed_at"]
    assert http.get("/v1/graph/build-all/nope").status_code == 404

    again = http.post("/v1/graph/build-all", json={"delay_seconds": 0})  # resume: nothing left
    assert again.json()["job_id"] == job_id and again.json()["chunks_to_process"] == 0
    finish(state)
    assert state["llm"].calls == 5


def test_duplicate_job_prevented_and_safe_stop(api, store):
    http, state = api
    release = threading.Event()
    state["llm"] = FakeLLM(on_call=lambda n: release.wait(5))
    job_id = http.post("/v1/graph/build-all", json={"delay_seconds": 0}).json()["job_id"]

    assert http.post("/v1/graph/build-all", json={}).status_code == 409
    assert http.post(f"/v1/graph/build-all/{job_id}/stop").status_code == 202
    release.set()
    finish(state)

    body = http.get(f"/v1/graph/build-all/{job_id}").json()
    assert body["status"] == "paused" and body["stop_requested"]
    assert body["chunks_processed"] == 1 and body["chunks_pending"] == 4
    assert http.post(f"/v1/graph/build-all/{job_id}/stop").status_code == 409


def test_runner_shutdown_pauses_job(store):
    collection = FakeCollection(corpus({"A-1": 3}))
    job_id = new_job(store, collection)
    started = threading.Event()
    runner = BuildRunner()

    def work(stop):
        llm = FakeLLM(on_call=lambda n: (started.set(), stop.wait(5)))
        return run(store, job_id, collection, llm, stop=stop)

    runner.start(job_id, work)
    started.wait(5)
    runner.shutdown(timeout=5)
    assert store.get_job(job_id)["status"] == "paused"
    assert store.chunk_counts(job_id)["completed"] == 1  # the in-flight chunk finished cleanly


def test_failures_api_paginates_and_retry_api_fixes_them(api):
    http, state = api
    state["client"] = FlakyGraphClient(FakeDriver(), {"A-1_chunk0", "B-2_chunk2"})
    job_id = http.post("/v1/graph/build-all", json={"delay_seconds": 0}).json()["job_id"]
    finish(state)

    page = http.get(f"/v1/graph/build-all/{job_id}/failures?limit=1").json()
    assert page["total_failed_documents"] == 2 and len(page["documents"]) == 1
    assert page["documents"][0]["document_id"] == "A-1"
    assert page["documents"][0]["status"] == "partially_completed"
    assert page["documents"][0]["failed_chunks"][0]["stage"] == "graph"
    assert http.get(f"/v1/graph/build-all/{job_id}").json()["status"] == ("completed_with_failures")

    state["client"] = FlakyGraphClient(FakeDriver())
    retry = http.post(f"/v1/graph/build-all/{job_id}/retry")
    assert retry.status_code == 202 and retry.json()["chunks_to_process"] == 2
    finish(state)
    body = http.get(f"/v1/graph/build-all/{job_id}").json()
    assert body["status"] == "completed" and body["chunks_failed"] == 0
    assert state["llm"].calls == 5  # graph-stage retries reused stored extractions
