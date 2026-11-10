"""Neo4j connection management (Story 1.6).

One ``Neo4jClient`` (one driver, with its own connection pool) is shared by the whole process.
It is created on the first graph request, so the application starts without Neo4j, and it is
closed when FastAPI shuts down. Sessions are opened per unit of work with ``with`` blocks, so
they are always closed. Credentials come from settings and are never logged or returned.
"""

import re
from collections.abc import Callable
from typing import Any, TypeVar

from neo4j import GraphDatabase, unit_of_work
from neo4j.exceptions import AuthError, ConfigurationError, DriverError, Neo4jError

from graph_rag.core.config import Settings

T = TypeVar("T")

_VERSION = re.compile(r"(\d+)\.(\d+)")


class GraphStoreError(Exception):
    """A Neo4j operation failed."""


class GraphNotConfiguredError(GraphStoreError):
    """NEO4J_URI, NEO4J_USERNAME or NEO4J_PASSWORD is missing."""


class GraphConnectionError(GraphStoreError):
    """Neo4j could not be reached or rejected the credentials."""


class GraphQueryTimeoutError(GraphStoreError):
    """A read query exceeded its transaction timeout (Story 1.7)."""


def _describe(exc: Exception) -> str:
    """Error type plus the server's message. Neo4j messages never contain the password."""
    return f"{type(exc).__name__}: {getattr(exc, 'message', None) or exc}"


class Neo4jClient:
    """Thin wrapper over one driver: connectivity check, write transactions, schema statements."""

    def __init__(self, driver: Any, database: str):
        self._driver = driver
        self.database = database
        self.schema_ready = False  # set once constraints/indexes were ensured by this process

    @classmethod
    def from_settings(
        cls, settings: Settings, driver_factory: Callable[..., Any] = GraphDatabase.driver
    ) -> "Neo4jClient":
        """Create the driver and verify connectivity. Raises ``GraphStoreError`` subclasses."""
        if not (settings.neo4j_uri and settings.neo4j_username and settings.neo4j_password):
            raise GraphNotConfiguredError(
                "Neo4j is not configured. Set NEO4J_URI, NEO4J_USERNAME and NEO4J_PASSWORD "
                "in .env (see .env.example)."
            )
        try:
            driver = driver_factory(
                settings.neo4j_uri,
                auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()),
            )
        except (ConfigurationError, ValueError) as exc:
            raise GraphConnectionError(f"Invalid Neo4j configuration ({_describe(exc)}).") from exc
        client = cls(driver, settings.neo4j_database)
        try:
            client.verify_connectivity()
        except GraphStoreError:
            client.close()
            raise
        return client

    def verify_connectivity(self) -> None:
        try:
            self._driver.verify_connectivity()
        except AuthError as exc:
            raise GraphConnectionError(
                "Neo4j rejected the credentials. Check NEO4J_USERNAME and NEO4J_PASSWORD."
            ) from exc
        except (Neo4jError, DriverError, OSError) as exc:
            raise GraphConnectionError(
                f"Cannot connect to Neo4j ({_describe(exc)}). Check NEO4J_URI."
            ) from exc

    def server_agent(self) -> str:
        """Server product string such as ``Neo4j/5.27-aura``."""
        try:
            return self._driver.get_server_info().agent
        except (Neo4jError, DriverError, OSError) as exc:
            raise GraphConnectionError(
                f"Cannot read Neo4j server info ({_describe(exc)})."
            ) from exc

    def server_version(self) -> tuple[int, int]:
        match = _VERSION.search(self.server_agent())
        return (int(match.group(1)), int(match.group(2))) if match else (0, 0)

    def execute_write(self, work: Callable[..., T], **kwargs: Any) -> T:
        """Run ``work(tx, **kwargs)`` in ONE managed write transaction.

        The driver commits when ``work`` returns and rolls back if it raises; transient errors
        are retried (safe because every write is a ``MERGE``). Errors raised by ``work`` itself
        are passed through unchanged.
        """
        try:
            with self._driver.session(database=self.database) as session:
                return session.execute_write(work, **kwargs)
        except (Neo4jError, DriverError) as exc:
            raise GraphStoreError(f"Neo4j write failed ({_describe(exc)}).") from exc

    def execute_read(
        self, work: Callable[..., T], *, timeout: float | None = None, **kwargs: Any
    ) -> T:
        """Run ``work(tx, **kwargs)`` in ONE managed *read* transaction (Story 1.7).

        Read access mode means the server rejects any write, so retrieval cannot change the
        graph. ``timeout`` (seconds) is enforced by the server; exceeding it raises
        ``GraphQueryTimeoutError``.
        """
        transaction_work = unit_of_work(timeout=timeout)(work) if timeout else work
        try:
            with self._driver.session(database=self.database) as session:
                return session.execute_read(transaction_work, **kwargs)
        except Neo4jError as exc:
            if "TransactionTimedOut" in (getattr(exc, "code", None) or ""):
                raise GraphQueryTimeoutError(
                    f"Neo4j read query timed out after {timeout} s."
                ) from exc
            raise GraphStoreError(f"Neo4j read failed ({_describe(exc)}).") from exc
        except DriverError as exc:
            raise GraphStoreError(f"Neo4j read failed ({_describe(exc)}).") from exc

    def run_schema_statements(self, statements: list[str]) -> None:
        """Run schema commands, each in its own auto-commit transaction (Neo4j requires this)."""
        try:
            with self._driver.session(database=self.database) as session:
                for statement in statements:
                    session.run(statement).consume()
        except (Neo4jError, DriverError) as exc:
            raise GraphStoreError(
                f"Neo4j schema initialization failed ({_describe(exc)})."
            ) from exc

    def close(self) -> None:
        self._driver.close()


_client: Neo4jClient | None = None


def get_neo4j_client(settings: Settings) -> Neo4jClient:
    """Return the shared client, creating and verifying it on first use.

    A failed attempt is not cached, so the next request tries again.
    """
    global _client
    if _client is None:
        _client = Neo4jClient.from_settings(settings)
    return _client


def close_neo4j_client() -> None:
    """Close the shared driver (called on application shutdown)."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
