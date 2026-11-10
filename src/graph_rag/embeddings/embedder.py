"""Local embedding generation with Sentence Transformers.

The model is loaded lazily on first use and then kept in memory, so it is loaded
once per process instead of once per chunk. ``sentence_transformers`` (and PyTorch)
are imported only when the model is loaded, which keeps application startup fast.
"""

import math
from functools import lru_cache
from typing import Any


class EmbeddingError(Exception):
    """Raised when text cannot be turned into valid embedding vectors."""


class EmbeddingModelError(EmbeddingError):
    """Raised when the embedding model itself cannot be loaded."""


def _load_sentence_transformer(model_name: str) -> Any:
    """Load a Sentence Transformers model (downloaded from Hugging Face on first use)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def validate_vector(vector: list[float], dimension: int) -> None:
    """Check that ``vector`` has exactly ``dimension`` finite numbers."""
    if len(vector) != dimension:
        raise EmbeddingError(f"Expected a {dimension}-dimensional vector, got {len(vector)}")
    if not all(isinstance(value, float) and math.isfinite(value) for value in vector):
        raise EmbeddingError("Embedding vector contains non-numeric or non-finite values")


class Embedder:
    """Turns texts into fixed-size vectors using one reusable model instance."""

    def __init__(self, model_name: str, dimension: int, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.batch_size = batch_size
        self._model: Any = None

    def load(self) -> None:
        """Load the model if it is not loaded yet. Safe to call repeatedly."""
        if self._model is not None:
            return
        try:
            self._model = _load_sentence_transformer(self.model_name)
        except Exception as exc:
            raise EmbeddingModelError(
                f"Could not load embedding model '{self.model_name}': {exc}. "
                "Check EMBEDDING_MODEL_NAME and that the model can be downloaded or is cached."
            ) from exc

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one validated vector per text, in the same order as ``texts``.

        Empty or whitespace-only texts are rejected because they carry no meaning
        to embed. Vectors are L2-normalized (length 1), which suits cosine similarity.
        """
        if not texts:
            return []
        blank_positions = [index for index, text in enumerate(texts) if not text.strip()]
        if blank_positions:
            raise EmbeddingError(f"Cannot embed empty text (positions {blank_positions})")

        self.load()
        try:
            raw_vectors = self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError(f"Embedding generation failed: {exc}") from exc

        # encode() returns a NumPy array; .tolist() turns it into plain Python floats.
        vectors = raw_vectors.tolist() if hasattr(raw_vectors, "tolist") else list(raw_vectors)
        if len(vectors) != len(texts):
            raise EmbeddingError(f"Expected {len(texts)} embeddings, got {len(vectors)}")
        try:
            vectors = [[float(value) for value in vector] for vector in vectors]
        except (TypeError, ValueError) as exc:
            raise EmbeddingError(f"Embedding vector contains non-numeric values: {exc}") from exc
        for vector in vectors:
            validate_vector(vector, self.dimension)
        return vectors

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text."""
        return self.embed_texts([text])[0]


@lru_cache
def get_embedder(model_name: str, dimension: int, batch_size: int = 32) -> Embedder:
    """Return one shared ``Embedder`` per configuration, so the model loads only once."""
    return Embedder(model_name, dimension, batch_size)
