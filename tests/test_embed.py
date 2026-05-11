"""Tests for scry.embed — embedding pipeline (workstream W2k).

Covers:
- StubEmbedder: dimensionality, determinism, distinctness
- serialize_embedding / deserialize_embedding: round-trip fidelity
- cosine_similarity: identical, orthogonal, opposite
- chunk_anchor: SECTION (short/long), CODE, CODE_IN_DOC
- make_embedder: factory dispatch, unsupported provider error
- Integration: real fastembed encoding (skipped by default)
"""

from __future__ import annotations

import struct

import pytest

from scry.embed import (
    EmbeddingProviderError,
    LocalFastEmbedder,
    StubEmbedder,
    chunk_anchor,
    cosine_similarity,
    deserialize_embedding,
    make_embedder,
    serialize_embedding,
)
from scry.models import Anchor, AnchorType, EmbeddingsConfig, SubChunk

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

_HASH = "sha256:" + "a" * 64
_HASH2 = "sha256:" + "b" * 64

# ~650-word paragraph repeated to generate text exceeding the 600-token
# default max_tokens threshold (approximation: word_count * 1.4).
_LONG_PARA = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! "
    "The five boxing wizards jump quickly. "
    "Sphinx of black quartz judge my vow. "
    "Bright vixens jump dozing fowl quack. "
) * 40  # ~1680 words → ~2352 approx-tokens, well above 600


def _make_anchor(
    text: str,
    anchor_type: AnchorType = AnchorType.SECTION,
    anchor_id: str = "docs/spec.md::intro",
) -> Anchor:
    """Build a minimal Anchor for testing."""
    return Anchor(
        id=anchor_id,
        type=anchor_type,
        path="docs/spec.md",
        content_text=text,
        content_hash=_HASH,
        fingerprint_simhash=12345,
    )


# ──────────────────────────────────────────────────────────────────────
# StubEmbedder
# ──────────────────────────────────────────────────────────────────────


class TestStubEmbedder:
    def test_dimensions_default(self) -> None:
        emb = StubEmbedder()
        assert emb.dimensions == 384

    def test_dimensions_custom(self) -> None:
        emb = StubEmbedder(dimensions=128)
        assert emb.dimensions == 128

    def test_dimensions_reflected_in_output(self) -> None:
        """Encoded blob has exactly dimensions * 4 bytes (float32)."""
        for dims in (64, 128, 384):
            emb = StubEmbedder(dimensions=dims)
            blob = emb.encode(["hello"])[0]
            assert len(blob) == dims * 4, f"dims={dims}"

    def test_provider(self) -> None:
        assert StubEmbedder().provider == "stub"

    def test_model_name(self) -> None:
        assert StubEmbedder().model_name == "stub"

    def test_tokenizer_version_is_none(self) -> None:
        assert StubEmbedder().tokenizer_version is None

    def test_deterministic_same_input(self) -> None:
        emb = StubEmbedder(dimensions=16)
        b1 = emb.encode(["hello world"])[0]
        b2 = emb.encode(["hello world"])[0]
        assert b1 == b2, "StubEmbedder must be deterministic"

    def test_distinct_inputs_produce_distinct_outputs(self) -> None:
        emb = StubEmbedder(dimensions=16)
        b1 = emb.encode(["apple"])[0]
        b2 = emb.encode(["orange"])[0]
        assert b1 != b2, "Distinct inputs should produce distinct embeddings"

    def test_batch_output_length_matches_input(self) -> None:
        emb = StubEmbedder(dimensions=16)
        texts = ["alpha", "beta", "gamma"]
        result = emb.encode(texts)
        assert len(result) == len(texts)

    def test_invalid_dimensions_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            StubEmbedder(dimensions=0)
        with pytest.raises(ValueError, match="positive"):
            StubEmbedder(dimensions=-1)

    def test_output_values_in_range(self) -> None:
        """Each float in the decoded vector must be in [-1, 1]."""
        emb = StubEmbedder(dimensions=32)
        blob = emb.encode(["test"])[0]
        values = deserialize_embedding(blob)
        assert all(-1.0 <= v <= 1.0 for v in values)


# ──────────────────────────────────────────────────────────────────────
# serialize_embedding / deserialize_embedding
# ──────────────────────────────────────────────────────────────────────


class TestSerializeDeserialize:
    def test_round_trip_simple(self) -> None:
        vec = [0.1, 0.2, 0.3, -0.5, 1.0]
        blob = serialize_embedding(vec)
        recovered = deserialize_embedding(blob)
        assert len(recovered) == len(vec)
        for orig, got in zip(vec, recovered, strict=True):
            assert abs(orig - got) < 1e-5, f"{orig} != {got}"

    def test_round_trip_zero_vector(self) -> None:
        vec = [0.0] * 16
        assert deserialize_embedding(serialize_embedding(vec)) == pytest.approx(vec)

    def test_round_trip_large_dim(self) -> None:
        import math

        vec = [math.sin(i / 10.0) for i in range(384)]
        recovered = deserialize_embedding(serialize_embedding(vec))
        for orig, got in zip(vec, recovered, strict=True):
            assert abs(orig - got) < 1e-5

    def test_blob_is_bytes(self) -> None:
        blob = serialize_embedding([1.0, 2.0])
        assert isinstance(blob, bytes)

    def test_blob_length_is_4x_dims(self) -> None:
        dims = 8
        blob = serialize_embedding([0.0] * dims)
        assert len(blob) == dims * 4

    def test_blob_is_little_endian_float32(self) -> None:
        """Verify the binary layout is packed little-endian float32."""
        blob = serialize_embedding([1.0])
        expected = struct.pack("<f", 1.0)
        assert blob == expected


# ──────────────────────────────────────────────────────────────────────
# cosine_similarity
# ──────────────────────────────────────────────────────────────────────


class TestCosineSimilarity:
    def _blob(self, *values: float) -> bytes:
        return serialize_embedding(list(values))

    def test_identical_vectors_give_1(self) -> None:
        b = self._blob(0.5, -0.3, 0.8)
        assert abs(cosine_similarity(b, b) - 1.0) < 1e-5

    def test_opposite_vectors_give_minus_1(self) -> None:
        b1 = self._blob(1.0, 0.0, 0.0)
        b2 = self._blob(-1.0, 0.0, 0.0)
        assert abs(cosine_similarity(b1, b2) - (-1.0)) < 1e-5

    def test_orthogonal_vectors_give_zero(self) -> None:
        b1 = self._blob(1.0, 0.0)
        b2 = self._blob(0.0, 1.0)
        assert abs(cosine_similarity(b1, b2)) < 1e-5

    def test_zero_vector_returns_zero(self) -> None:
        b_zero = self._blob(0.0, 0.0, 0.0)
        b_unit = self._blob(1.0, 0.0, 0.0)
        assert cosine_similarity(b_zero, b_unit) == 0.0

    def test_range_is_within_minus1_to_1(self) -> None:
        """Stub embedder vectors should yield similarity in [-1, 1]."""
        emb = StubEmbedder(dimensions=32)
        b1 = emb.encode(["hello"])[0]
        b2 = emb.encode(["world"])[0]
        sim = cosine_similarity(b1, b2)
        assert -1.0 <= sim <= 1.0


# ──────────────────────────────────────────────────────────────────────
# chunk_anchor
# ──────────────────────────────────────────────────────────────────────


class TestChunkAnchor:
    def test_section_short_text_single_chunk(self) -> None:
        """Text below max_tokens threshold → one SubChunk."""
        short = "This is a short section with only a few words."
        anchor = _make_anchor(short, AnchorType.SECTION)
        chunks = chunk_anchor(anchor, max_tokens=600)
        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0
        assert chunks[0].overlap_with_prev == 0
        assert chunks[0].parent_id == anchor.id
        assert chunks[0].parent_content_hash == _HASH

    def test_section_short_text_preserves_full_text(self) -> None:
        short = "Just a few words."
        anchor = _make_anchor(short)
        chunks = chunk_anchor(anchor, max_tokens=600)
        assert chunks[0].text == short

    def test_section_long_text_multiple_chunks(self) -> None:
        """Long text exceeding max_tokens → multiple SubChunks."""
        anchor = _make_anchor(_LONG_PARA, AnchorType.SECTION)
        chunks = chunk_anchor(anchor, max_tokens=100, overlap_tokens=20)
        assert len(chunks) > 1, "Long text should produce multiple sub-chunks"

    def test_section_chunks_carry_parent_content_hash(self) -> None:
        anchor = _make_anchor(_LONG_PARA)
        chunks = chunk_anchor(anchor, max_tokens=100, overlap_tokens=20)
        for chunk in chunks:
            assert chunk.parent_content_hash == _HASH

    def test_section_chunk_indices_sequential(self) -> None:
        anchor = _make_anchor(_LONG_PARA)
        chunks = chunk_anchor(anchor, max_tokens=100, overlap_tokens=20)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_section_first_chunk_no_overlap(self) -> None:
        anchor = _make_anchor(_LONG_PARA)
        chunks = chunk_anchor(anchor, max_tokens=100, overlap_tokens=20)
        assert chunks[0].overlap_with_prev == 0

    def test_section_subsequent_chunks_have_overlap(self) -> None:
        anchor = _make_anchor(_LONG_PARA)
        overlap = 30
        chunks = chunk_anchor(anchor, max_tokens=100, overlap_tokens=overlap)
        assert len(chunks) > 1
        for chunk in chunks[1:]:
            assert chunk.overlap_with_prev == int(overlap)

    def test_section_parent_id_propagated(self) -> None:
        anchor = _make_anchor(_LONG_PARA, anchor_id="docs/design.md::arch")
        chunks = chunk_anchor(anchor, max_tokens=100, overlap_tokens=20)
        for chunk in chunks:
            assert chunk.parent_id == "docs/design.md::arch"

    def test_code_anchor_always_single_chunk(self) -> None:
        """CODE anchors must not be sub-chunked, regardless of length."""
        anchor = _make_anchor(
            _LONG_PARA,
            AnchorType.CODE,
            anchor_id="src/policy.py:PolicyRule",
        )
        # Patch the path validator: CODE anchors don't have heading_path
        anchor = Anchor(
            id="src/policy.py:PolicyRule",
            type=AnchorType.CODE,
            path="src/policy.py",
            symbol_name="PolicyRule",
            content_text=_LONG_PARA,
            content_hash=_HASH,
            fingerprint_simhash=99,
            transitive_hash_status=None,
        )
        chunks = chunk_anchor(anchor, max_tokens=50)  # very small limit
        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0
        assert chunks[0].text == _LONG_PARA
        assert chunks[0].overlap_with_prev == 0

    def test_code_in_doc_anchor_always_single_chunk(self) -> None:
        """CODE_IN_DOC anchors must not be sub-chunked, regardless of length."""
        anchor = Anchor(
            id="docs/spec.md::intro::my-func",
            type=AnchorType.CODE_IN_DOC,
            path="docs/spec.md",
            content_text=_LONG_PARA,
            content_hash=_HASH,
            fingerprint_simhash=77,
        )
        chunks = chunk_anchor(anchor, max_tokens=50)
        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0
        assert chunks[0].overlap_with_prev == 0

    def test_chunk_anchor_returns_subchunk_instances(self) -> None:
        anchor = _make_anchor("Short text.")
        chunks = chunk_anchor(anchor)
        assert all(isinstance(c, SubChunk) for c in chunks)

    def test_chunk_anchor_never_returns_empty(self) -> None:
        """chunk_anchor must always return at least one SubChunk."""
        anchor = _make_anchor("")  # Empty content_text edge-case
        chunks = chunk_anchor(anchor)
        assert len(chunks) >= 1


# ──────────────────────────────────────────────────────────────────────
# make_embedder factory
# ──────────────────────────────────────────────────────────────────────


class TestMakeEmbedder:
    def test_local_provider_returns_local_fast_embedder(self) -> None:
        config = EmbeddingsConfig(provider="local")
        emb = make_embedder(config)
        assert isinstance(emb, LocalFastEmbedder)

    def test_local_provider_propagates_model_name(self) -> None:
        config = EmbeddingsConfig(provider="local", model="BAAI/bge-small-en-v1.5")
        emb = make_embedder(config)
        assert isinstance(emb, LocalFastEmbedder)
        assert emb.model_name == "BAAI/bge-small-en-v1.5"

    def test_local_provider_propagates_dimensions(self) -> None:
        config = EmbeddingsConfig(provider="local", dimensions=256)
        emb = make_embedder(config)
        assert emb.dimensions == 256

    def test_local_embedder_is_lazy(self) -> None:
        """LocalFastEmbedder.__init__ must NOT load the model (no download)."""
        config = EmbeddingsConfig(provider="local")
        emb = make_embedder(config)
        assert isinstance(emb, LocalFastEmbedder)
        # The internal model slot is None until encode() is called.
        assert emb._model is None  # white-box test: lazy-init guard

    def test_force_stub_overrides_local(self) -> None:
        config = EmbeddingsConfig(provider="local")
        emb = make_embedder(config, force_stub=True)
        assert isinstance(emb, StubEmbedder)

    def test_force_stub_dimensions_match_config(self) -> None:
        config = EmbeddingsConfig(provider="local", dimensions=128)
        emb = make_embedder(config, force_stub=True)
        assert emb.dimensions == 128

    def test_openai_provider_raises(self) -> None:
        config = EmbeddingsConfig(
            provider="openai", model="text-embedding-3-small", dimensions=1536
        )
        with pytest.raises(EmbeddingProviderError, match="not implemented in Wave 2"):
            make_embedder(config)

    def test_voyage_provider_raises(self) -> None:
        config = EmbeddingsConfig(provider="voyage", model="voyage-2", dimensions=1024)
        with pytest.raises(EmbeddingProviderError, match="not implemented in Wave 2"):
            make_embedder(config)

    def test_custom_provider_raises(self) -> None:
        config = EmbeddingsConfig(provider="custom", model="my-model", dimensions=512)
        with pytest.raises(EmbeddingProviderError, match="not implemented in Wave 2"):
            make_embedder(config)

    def test_error_message_mentions_provider_name(self) -> None:
        config = EmbeddingsConfig(
            provider="openai", model="text-embedding-3-small", dimensions=1536
        )
        with pytest.raises(EmbeddingProviderError, match="openai"):
            make_embedder(config)


# ──────────────────────────────────────────────────────────────────────
# LocalFastEmbedder (unit — no model download)
# ──────────────────────────────────────────────────────────────────────


class TestLocalFastEmbedder:
    def test_provider(self) -> None:
        assert LocalFastEmbedder().provider == "local"

    def test_model_name_default(self) -> None:
        assert LocalFastEmbedder().model_name == "BAAI/bge-small-en-v1.5"

    def test_dimensions_default(self) -> None:
        assert LocalFastEmbedder().dimensions == 384

    def test_tokenizer_version_is_none(self) -> None:
        assert LocalFastEmbedder().tokenizer_version is None

    def test_custom_model_name(self) -> None:
        emb = LocalFastEmbedder(model_name="BAAI/bge-base-en-v1.5", dimensions=768)
        assert emb.model_name == "BAAI/bge-base-en-v1.5"
        assert emb.dimensions == 768

    def test_cache_dir_stored(self, tmp_path: object) -> None:
        from pathlib import Path

        assert isinstance(tmp_path, Path)
        emb = LocalFastEmbedder(cache_dir=tmp_path)
        assert emb._cache_dir == tmp_path


# ──────────────────────────────────────────────────────────────────────
# Integration test (skipped unless -m integration)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestIntegration:
    """Real fastembed integration tests.

    These tests download ~30 MB of model weights on first run.
    Run with: ``pytest -m integration``
    """

    def test_local_embedder_encode_single(self) -> None:
        emb = LocalFastEmbedder()
        blobs = emb.encode(["Hello, world!"])
        assert len(blobs) == 1
        vec = deserialize_embedding(blobs[0])
        assert len(vec) == 384

    def test_local_embedder_encode_batch(self) -> None:
        emb = LocalFastEmbedder()
        texts = ["first", "second", "third"]
        blobs = emb.encode(texts)
        assert len(blobs) == 3
        for blob in blobs:
            vec = deserialize_embedding(blob)
            assert len(vec) == 384

    def test_local_embedder_deterministic(self) -> None:
        emb = LocalFastEmbedder()
        b1 = emb.encode(["determinism test"])[0]
        b2 = emb.encode(["determinism test"])[0]
        assert b1 == b2

    def test_local_embedder_distinct_inputs(self) -> None:
        emb = LocalFastEmbedder()
        b1 = emb.encode(["cat"])[0]
        b2 = emb.encode(["dog"])[0]
        assert b1 != b2

    def test_cosine_similarity_with_real_embeddings(self) -> None:
        emb = LocalFastEmbedder()
        b_same1 = emb.encode(["the quick brown fox"])[0]
        b_same2 = emb.encode(["the quick brown fox"])[0]
        b_diff = emb.encode(["completely unrelated zebra quantum"])[0]
        assert abs(cosine_similarity(b_same1, b_same2) - 1.0) < 1e-3
        assert cosine_similarity(b_same1, b_diff) < cosine_similarity(b_same1, b_same2)

    def test_make_embedder_local_encodes_correctly(self) -> None:
        config = EmbeddingsConfig(provider="local")
        emb = make_embedder(config)
        blobs = emb.encode(["integration test"])
        assert len(blobs) == 1
        vec = deserialize_embedding(blobs[0])
        assert len(vec) == config.dimensions

    def test_local_embedder_dimensions_mismatch_raises(self) -> None:
        """Regression (review-w2k MEDIUM): wrong `dimensions` arg must raise.

        Without this guard, a misconfigured `dimensions` value silently
        flows into IndexMetadata.embedding_dimensions, breaking the
        §7.2.1 model-mismatch detector and the --reembed migration.
        """
        # bge-small-en-v1.5 emits 384-dim vectors; pass 768 to trigger.
        emb = LocalFastEmbedder(model_name="BAAI/bge-small-en-v1.5", dimensions=768)
        with pytest.raises(EmbeddingProviderError, match="dimensions mismatch"):
            emb.encode(["any text"])

    def test_local_embedder_dimensions_validated_only_once(self) -> None:
        """Validation runs on first encode; subsequent encodes don't re-check."""
        emb = LocalFastEmbedder()
        emb.encode(["first"])
        # If the second encode tried to re-validate, swapping the model
        # mid-flight would break it. Mutate _dimensions and confirm a
        # second encode does NOT raise.
        emb._dimensions = 99999
        emb.encode(["second"])  # must not raise

# uat-r5-5 pr-d noise
