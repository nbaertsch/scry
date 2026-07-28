"""Embedding pipeline for scry.

Implements DESIGN.md §3.4 (two-tier embedding), §6 (``embeddings:`` config
block), and §11 (tech stack: fastembed BAAI/bge-small-en-v1.5 default).

Public API
----------
EmbeddingProviderError  — raised for unimplemented Wave-2 providers
Embedder                — structural Protocol every backend implements
StubEmbedder            — deterministic hash-based embedder (unit tests)
LocalFastEmbedder       — fastembed-backed embedder (lazy-loaded)
make_embedder           — factory dispatching on EmbeddingsConfig.provider
chunk_anchor            — SECTION sub-chunking helper (§3.4)
serialize_embedding     — sqlite-vec wire serialization
deserialize_embedding   — inverse of serialize_embedding
cosine_similarity       — ad-hoc cosine scoring between serialized blobs
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import struct
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

import sqlite_vec

from scry.extract.markdown import split_section_text
from scry.models import Anchor, AnchorType, EmbeddingsConfig, SubChunk

__all__ = [
    "Embedder",
    "EmbeddingProviderError",
    "LocalFastEmbedder",
    "StubEmbedder",
    "chunk_anchor",
    "cosine_similarity",
    "deserialize_embedding",
    "make_embedder",
    "serialize_embedding",
]


class EmbeddingProviderError(Exception):
    """Raised when config requests an unimplemented embedding provider.

    Providers not yet implemented in Wave 2: ``'openai'``, ``'voyage'``,
    ``'custom'``.  Add a custom :class:`Embedder` implementation and
    dispatch it from :func:`make_embedder` to enable them.
    """


class Embedder(Protocol):
    """The interface every embedding backend implements.

    Implementations
    ---------------
    - :class:`LocalFastEmbedder` — default; lazy-loads fastembed on
      first :meth:`encode` call.
    - :class:`StubEmbedder` — deterministic hash-based vectors for unit
      tests; does not download model weights.
    - ``OpenAIEmbedder``, ``VoyageEmbedder``, etc. — Wave 5, not in scope.
    """

    @property
    def model_name(self) -> str:
        """Name of the embedding model (e.g. ``'BAAI/bge-small-en-v1.5'``)."""
        ...

    @property
    def dimensions(self) -> int:
        """Output vector dimensionality."""
        ...

    @property
    def provider(self) -> str:
        """Provider tag: ``'local' | 'openai' | 'voyage' | 'custom' | 'stub'``."""
        ...

    @property
    def tokenizer_version(self) -> str | None:
        """Tokenizer version string; ``None`` when unknown.

        Included in ``IndexMetadata`` when present so that downstream
        tooling can detect tokenizer-level incompatibilities independently
        of the model name.
        """
        ...

    def encode(self, texts: list[str]) -> list[bytes]:
        """Batch-encode *texts* and return one serialized vector per input.

        The bytes representation uses :func:`serialize_embedding`
        (``sqlite_vec.serialize_float32``) so it is directly storable in
        the W2a ``chunks`` vector table.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            List of ``bytes`` blobs, one per input text, in the same order.
        """
        ...


# ──────────────────────────────────────────────────────────────────────
# Serialization helpers
# ──────────────────────────────────────────────────────────────────────


def serialize_embedding(vec: Sequence[float]) -> bytes:
    """Serialize a float vector to sqlite-vec wire format.

    Uses ``sqlite_vec.serialize_float32(vec)`` — the same format the W2a
    vector table expects for the ``embedding`` column.

    Args:
        vec: Sequence of floats representing the embedding vector.

    Returns:
        Raw ``bytes`` in little-endian float32 layout.
    """
    result: bytes = sqlite_vec.serialize_float32(list(vec))
    return result


def deserialize_embedding(blob: bytes) -> list[float]:
    """Inverse of :func:`serialize_embedding`.

    Interprets *blob* as a sequence of little-endian ``float32`` values
    and returns them as a Python ``list[float]``.

    Args:
        blob: Bytes produced by :func:`serialize_embedding`.

    Returns:
        List of floats reconstructed from the wire format.
    """
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


def cosine_similarity(a: bytes, b: bytes) -> float:
    """Compute cosine similarity between two serialized embeddings.

    Intended for ad-hoc retrieval scoring outside sqlite-vec.  Both blobs
    must have been produced by :func:`serialize_embedding` and represent
    vectors of the same dimensionality.

    Returns:
        Float in ``[-1.0, 1.0]``.  Returns ``0.0`` when either vector
        has zero magnitude (degenerate case).
    """
    va = deserialize_embedding(a)
    vb = deserialize_embedding(b)
    dot = sum(x * y for x, y in zip(va, vb, strict=True))
    norm_a = math.sqrt(sum(x * x for x in va))
    norm_b = math.sqrt(sum(x * x for x in vb))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ──────────────────────────────────────────────────────────────────────
# StubEmbedder
# ──────────────────────────────────────────────────────────────────────


class StubEmbedder:
    """Deterministic embedder that hashes input text into a fixed-dim vector.

    Useful for unit tests that need an embedder but do not want to load
    fastembed (which downloads ~30 MB on first call).  **Not for production.**

    Each text is mapped to a vector of *dimensions* floats where element
    *i* is derived from ``sha256(text + str(i)).digest()[0]`` scaled to
    ``[-1.0, 1.0]``.  The mapping is fully deterministic: identical inputs
    always produce identical outputs, while distinct inputs almost certainly
    produce distinct outputs.
    """

    def __init__(self, dimensions: int = 384) -> None:
        """Initialise the stub embedder.

        Args:
            dimensions: Output vector length.  Must be positive.
        """
        if dimensions <= 0:
            raise ValueError(f"dimensions must be positive, got {dimensions}")
        self._dimensions = dimensions

    @property
    def model_name(self) -> str:
        """Fixed sentinel string identifying this as a stub."""
        return "stub"

    @property
    def dimensions(self) -> int:
        """Output vector dimensionality."""
        return self._dimensions

    @property
    def provider(self) -> str:
        """Provider tag for the stub backend."""
        return "stub"

    @property
    def tokenizer_version(self) -> str | None:
        """Stubs have no tokenizer version."""
        return None

    def encode(self, texts: list[str]) -> list[bytes]:
        """Deterministically encode *texts* into fixed-dim hash-derived vectors.

        Args:
            texts: Strings to embed.

        Returns:
            One serialized float32 blob per input text.
        """
        results: list[bytes] = []
        for text in texts:
            vec: list[float] = []
            for idx in range(self._dimensions):
                digest = hashlib.sha256(f"{text}{idx}".encode()).digest()
                # Map first byte (0-255) to [-1.0, 1.0].
                vec.append(float(digest[0]) / 127.5 - 1.0)
            results.append(serialize_embedding(vec))
        return results


# ──────────────────────────────────────────────────────────────────────
# LocalFastEmbedder
# ──────────────────────────────────────────────────────────────────────


def _default_model_cache_dir() -> Path:
    """Return a stable platform-appropriate cache directory for model weights.

    Uses ``%LOCALAPPDATA%/scry/models`` on Windows, ``~/.cache/scry/models``
    elsewhere.  Falls back to fastembed's default (system temp) only if the
    preferred location cannot be created.
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            p = Path(base) / "scry" / "models"
        else:
            p = Path.home() / ".cache" / "scry" / "models"
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        if xdg:
            p = Path(xdg) / "scry" / "models"
        else:
            p = Path.home() / ".cache" / "scry" / "models"
    try:
        p.mkdir(parents=True, exist_ok=True)
        return p
    except OSError:
        # Fall back to letting fastembed decide (system temp).
        return None  # type: ignore[return-value]


class LocalFastEmbedder:
    """fastembed-backed embedder.  Lazy-loads the model on first :meth:`encode`.

    Default model: ``BAAI/bge-small-en-v1.5`` (384 dimensions, ~30 MB ONNX
    weights, no API key required).  The model is downloaded to *cache_dir*
    (or ``%LOCALAPPDATA%/scry/models`` on Windows, ``~/.cache/scry/models``
    elsewhere) on first use.

    Thread-safety: the lazy-init guard is not thread-safe.  For concurrent
    use, instantiate from a single thread or guard externally.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        dimensions: int = 384,
        cache_dir: Path | None = None,
    ) -> None:
        """Initialise the fastembed embedder (does **not** load the model yet).

        Args:
            model_name: fastembed model identifier.
            dimensions: Expected output dimensionality.  The first call to
                :meth:`encode` validates this against the loaded model and
                raises :class:`EmbeddingProviderError` on mismatch — without
                this guard a misconfigured ``dimensions`` value would
                silently flow into ``IndexMetadata.embedding_dimensions``,
                breaking the §7.2.1 model-mismatch detector and the
                ``--reembed`` migration path.
            cache_dir:  Directory for model weight caching.  ``None`` uses
                ``%LOCALAPPDATA%/scry/models`` (Windows) or
                ``~/.cache/scry/models`` (elsewhere).
        """
        self._model_name = model_name
        self._dimensions = dimensions
        self._cache_dir = cache_dir or _default_model_cache_dir()
        self._model: Any = None  # fastembed.TextEmbedding; typed Any (no stubs)
        self._dimensions_validated = False

    @property
    def model_name(self) -> str:
        """fastembed model identifier."""
        return self._model_name

    @property
    def dimensions(self) -> int:
        """Configured output vector dimensionality."""
        return self._dimensions

    @property
    def provider(self) -> str:
        """Provider tag for the local fastembed backend."""
        return "local"

    @property
    def tokenizer_version(self) -> str | None:
        """Not exposed by fastembed; returns ``None``."""
        return None

    def _ensure_model(self) -> None:
        """Lazily initialise the fastembed TextEmbedding model.

        If the cached model files are corrupt or missing (e.g. system temp
        cleanup deleted the ONNX weights), clears the offending cache entry
        and retries the download once.
        """
        if self._model is not None:
            return
        try:
            import fastembed
        except ImportError as exc:
            raise EmbeddingProviderError(
                "fastembed is not installed.  "
                "Run `pip install fastembed` or add it to your project dependencies."
            ) from exc

        kwargs: dict[str, Any] = {"model_name": self._model_name}
        if self._cache_dir is not None:
            kwargs["cache_dir"] = str(self._cache_dir)

        try:
            self._model = fastembed.TextEmbedding(**kwargs)
        except Exception as exc:
            # Detect corrupt/missing model files in cache and retry once.
            exc_str = str(exc)
            if "NO_SUCHFILE" in exc_str or "Load model" in exc_str:
                logger.warning(
                    "scry: cached model files appear corrupt — clearing cache "
                    "entry and re-downloading model '%s'",
                    self._model_name,
                )
                self._clear_corrupt_cache()
                # Retry — this will trigger a fresh download.
                self._model = fastembed.TextEmbedding(**kwargs)
            else:
                raise

    def _clear_corrupt_cache(self) -> None:
        """Remove the corrupt model cache directory so fastembed re-downloads."""
        import shutil

        if self._cache_dir is None:
            return
        cache_path = Path(self._cache_dir)
        if not cache_path.exists():
            return
        # fastembed stores models as models--<org>--<model>-onnx-q
        model_slug = self._model_name.replace("/", "--")
        for entry in cache_path.iterdir():
            if model_slug in entry.name and entry.is_dir():
                logger.info("scry: removing corrupt cache entry: %s", entry)
                shutil.rmtree(entry, ignore_errors=True)

    def encode(self, texts: list[str]) -> list[bytes]:
        """Batch-encode *texts* using fastembed.

        Loads the model on the first call.  After the first call, asserts
        that the actual model output dimensionality matches the value
        passed at construction; mismatch raises
        :class:`EmbeddingProviderError`.  This guard is the W2k-side defense
        for §7.2.1 v3.1 — without it a user supplying the wrong
        ``dimensions`` for their model would silently poison
        ``IndexMetadata.embedding_dimensions``, breaking the model-mismatch
        detector and the ``--reembed`` migration path.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            One serialized float32 blob per input text.

        Raises:
            EmbeddingProviderError: If fastembed cannot be imported, or
                the loaded model emits vectors of a different
                dimensionality than the one declared at construction.
        """
        self._ensure_model()
        embeddings: list[Any] = list(self._model.embed(texts))
        results = [serialize_embedding([float(x) for x in vec]) for vec in embeddings]
        # One-shot dimensionality check (the first non-empty batch).
        if not self._dimensions_validated and results:
            actual_dims = len(deserialize_embedding(results[0]))
            if actual_dims != self._dimensions:
                raise EmbeddingProviderError(
                    f"LocalFastEmbedder dimensions mismatch: model "
                    f"{self._model_name!r} emits {actual_dims}-dim vectors "
                    f"but constructor was passed dimensions={self._dimensions}. "
                    f"Update your .scry/config.yaml `embeddings.dimensions` to "
                    f"{actual_dims} (or remove it to derive from the model)."
                )
            self._dimensions_validated = True
        return results


# ──────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────


def make_embedder(config: EmbeddingsConfig, *, force_stub: bool = False) -> Embedder:
    """Return an :class:`Embedder` appropriate for *config*.

    Wave 2 supports ``provider='local'`` (fastembed) and the
    ``force_stub=True`` escape hatch (:class:`StubEmbedder`).

    Providers ``'openai'``, ``'voyage'``, and ``'custom'`` raise
    :class:`EmbeddingProviderError` with a message directing the caller
    to implement a custom backend.

    Args:
        config:     Embeddings configuration block from ``.scry/config.yaml``.
        force_stub: When ``True``, always return a :class:`StubEmbedder`
                    regardless of ``config.provider``.  Used in tests.

    Returns:
        Concrete :class:`Embedder` instance.

    Raises:
        EmbeddingProviderError: For any provider not yet implemented.
    """
    if force_stub:
        return StubEmbedder(dimensions=config.dimensions)
    if config.provider == "local":
        return LocalFastEmbedder(model_name=config.model, dimensions=config.dimensions)
    raise EmbeddingProviderError(
        f"Provider '{config.provider}' not implemented in Wave 2; "
        "add a custom Embedder subclass and dispatch via make_embedder()."
    )


# ──────────────────────────────────────────────────────────────────────
# Sub-chunking helper
# ──────────────────────────────────────────────────────────────────────


def chunk_anchor(
    anchor: Anchor,
    *,
    max_tokens: int = 600,
    overlap_tokens: int = 50,
) -> list[SubChunk]:
    """Split a parent anchor into sub-chunks for embedding (DESIGN.md §3.4).

    SECTION anchors
    ~~~~~~~~~~~~~~~
    Calls :func:`scry.extract.markdown.split_section_text` to divide the
    anchor's ``content_text`` into overlapping sub-chunks.  Each resulting
    :class:`~scry.models.SubChunk` carries:

    - ``parent_id``           — the parent anchor's stable Layer-1 ID
    - ``chunk_index``         — 0-based position in the sub-chunk sequence
    - ``text``                — the sub-chunk body
    - ``parent_content_hash`` — copy of the parent's hash (for the §7.3
                                stale-orphan filter)
    - ``overlap_with_prev``   — ``int(overlap_tokens)`` for all chunks
                                except the first (approximation, since
                                ``split_section_text`` does not return
                                per-chunk overlap counts)

    CODE and CODE_IN_DOC anchors
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    These are not sub-chunked per §3.4 (they are typically small and
    atomic).  A single :class:`~scry.models.SubChunk` containing the
    entire ``content_text`` is returned, with ``chunk_index=0`` and
    ``overlap_with_prev=0``.

    Args:
        anchor:         Parent anchor to split.
        max_tokens:     Maximum approximate token budget per sub-chunk.
        overlap_tokens: Approximate token overlap with the preceding chunk.

    Returns:
        Non-empty list of :class:`~scry.models.SubChunk` records.
    """
    if anchor.type in (AnchorType.CODE, AnchorType.CODE_IN_DOC):
        return [
            SubChunk(
                parent_id=anchor.id,
                chunk_index=0,
                text=anchor.content_text,
                parent_content_hash=anchor.content_hash,
                overlap_with_prev=0,
            )
        ]

    # SECTION path: delegate to the paragraph/sentence splitter.
    raw_chunks = split_section_text(anchor.content_text, max_tokens, overlap_tokens)
    overlap_approx = int(overlap_tokens)

    return [
        SubChunk(
            parent_id=anchor.id,
            chunk_index=i,
            text=text,
            parent_content_hash=anchor.content_hash,
            overlap_with_prev=0 if i == 0 else overlap_approx,
        )
        for i, text in enumerate(raw_chunks)
    ]
