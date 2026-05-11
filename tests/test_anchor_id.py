"""Tests for `scry.anchor_id` — W1c: anchor IDs, slugification, canonicalization.

Covers every §15.3 table row and every §5.4 canonicalization clause.
"""

from __future__ import annotations

import re

import pytest

from scry.anchor_id import (
    canonicalize_content,
    content_hash,
    derive_anonymous_code_in_doc_id,
    derive_code_id,
    derive_code_in_doc_id,
    derive_section_id,
    disambiguate_siblings,
    fingerprint_simhash,
    parse_html_comment_id,
    short_content_hash,
    slugify,
)

# ──────────────────────────────────────────────────────────────────────
# slugify (§15.3)
# ──────────────────────────────────────────────────────────────────────


def test_slugify_basic_ascii() -> None:
    assert slugify("Rule Structure") == "rule-structure"


def test_slugify_mixed_case() -> None:
    assert slugify("Hello World") == "hello-world"


def test_slugify_already_lowercase() -> None:
    assert slugify("examples") == "examples"


def test_slugify_numbers_preserved() -> None:
    assert slugify("Section 42") == "section-42"


def test_slugify_non_ascii_cafe() -> None:
    # Diacritic stripped via NFKD decomposition.
    assert slugify("Café") == "cafe"


def test_slugify_non_ascii_multiple() -> None:
    # Multiple diacritics.
    assert slugify("Naïve résumé") == "naive-resume"


def test_slugify_empty_string_fallback() -> None:
    result = slugify("")
    assert result.startswith("section-")
    # Fallback hash is exactly 8 hex characters.
    suffix = result[len("section-") :]
    assert re.fullmatch(r"[0-9a-f]{8}", suffix), f"unexpected fallback: {result!r}"


def test_slugify_punctuation_only_fallback() -> None:
    result = slugify("?!@#")
    assert result.startswith("section-")


def test_slugify_emoji_only_fallback() -> None:
    # Emoji → non-ASCII → discarded → empty → fallback.
    result = slugify("🎉")
    assert result.startswith("section-")


def test_slugify_custom_fallback_prefix() -> None:
    result = slugify("", fallback_prefix="heading")
    assert result.startswith("heading-")


def test_slugify_consecutive_non_alnum_collapsed() -> None:
    # Multiple spaces / mixed separators collapse to one dash.
    assert slugify("a  b") == "a-b"
    assert slugify("a---b") == "a-b"
    assert slugify("a !? b") == "a-b"


def test_slugify_leading_trailing_dash_stripped() -> None:
    # Dashes at the boundaries are stripped.
    assert slugify("-hello-") == "hello"
    assert slugify("  spaces  ") == "spaces"


def test_slugify_emoji_with_ascii_letters() -> None:
    # 🎉 vanishes but the surrounding ASCII text survives.
    assert slugify("intro 🎉 guide") == "intro-guide"


def test_slugify_fallback_is_deterministic() -> None:
    # Same input always produces the same fallback.
    assert slugify("?!") == slugify("?!")


def test_slugify_fallback_differs_for_different_inputs() -> None:
    # Different empty-slug inputs produce different fallback hashes.
    # (They hash the *original* text, not the empty slug.)
    result_a = slugify("🎉")
    result_b = slugify("🎊")
    assert result_a != result_b


# ──────────────────────────────────────────────────────────────────────
# derive_section_id (§3.2)
# ──────────────────────────────────────────────────────────────────────


def test_derive_section_id_basic() -> None:
    sid = derive_section_id(
        "docs/POLICY_ENGINE.md",
        ["Policy Engine", "Rule Structure"],
    )
    assert sid == "docs/POLICY_ENGINE.md::policy-engine::rule-structure"


def test_derive_section_id_no_headings() -> None:
    # Empty heading_path → just the path (file-level anchor).
    sid = derive_section_id("README.md", [])
    assert sid == "README.md"


def test_derive_section_id_single_heading() -> None:
    sid = derive_section_id("spec.md", ["Introduction"])
    assert sid == "spec.md::introduction"


def test_derive_section_id_deep_nesting() -> None:
    sid = derive_section_id("doc.md", ["A", "B", "C"])
    assert sid == "doc.md::a::b::c"


# ──────────────────────────────────────────────────────────────────────
# derive_code_in_doc_id (§3.2)
# ──────────────────────────────────────────────────────────────────────


def test_derive_code_in_doc_id_basic() -> None:
    section = "docs/POLICY_ENGINE.md::policy-engine::rule-structure"
    cid = derive_code_in_doc_id(section, "PolicyRule")
    assert cid == f"{section}::policyrule"


def test_derive_code_in_doc_id_slugifies_declaration() -> None:
    section = "spec.md::intro"
    cid = derive_code_in_doc_id(section, "My Config Class")
    assert cid == f"{section}::my-config-class"


# ──────────────────────────────────────────────────────────────────────
# derive_code_id (§3.2)
# ──────────────────────────────────────────────────────────────────────


def test_derive_code_id_simple() -> None:
    cid = derive_code_id("python/policy.py", ["OuterClass", "method"])
    assert cid == "python/policy.py:OuterClass.method"


def test_derive_code_id_single_symbol() -> None:
    cid = derive_code_id("src/foo.py", ["MyClass"])
    assert cid == "src/foo.py:MyClass"


def test_derive_code_id_with_signature_hash() -> None:
    # @N overload suffix uses the first 6 hex chars of sig hash.
    cid = derive_code_id("api.ts", ["f"], signature_hash="abc123def456")
    assert cid == "api.ts:f@abc123"


def test_derive_code_id_truncates_sig_hash_to_6() -> None:
    cid = derive_code_id("src/a.py", ["g"], signature_hash="123456789abcdef")
    assert cid == "src/a.py:g@123456"


def test_derive_code_id_no_sig_hash() -> None:
    cid = derive_code_id("src/a.py", ["Outer", "Inner", "leaf"])
    assert "@" not in cid
    assert cid == "src/a.py:Outer.Inner.leaf"


# ──────────────────────────────────────────────────────────────────────
# disambiguate_siblings (§15.3)
# ──────────────────────────────────────────────────────────────────────


def test_disambiguate_siblings_no_duplicates() -> None:
    ids = ["intro", "overview", "examples"]
    assert disambiguate_siblings(ids) == ["intro", "overview", "examples"]


def test_disambiguate_siblings_dash_two_dupes() -> None:
    # §15.3: two H3 "Examples" → examples / examples-2
    result = disambiguate_siblings(["examples", "examples"])
    assert result == ["examples", "examples-2"]


def test_disambiguate_siblings_dash_three_dupes() -> None:
    result = disambiguate_siblings(["examples", "intro", "examples", "examples"])
    assert result == ["examples", "intro", "examples-2", "examples-3"]


def test_disambiguate_siblings_at_two_dupes() -> None:
    # §15.3: two Config code blocks → Config / Config@2
    result = disambiguate_siblings(["Config", "Config"], suffix="@")
    assert result == ["Config", "Config@2"]


def test_disambiguate_siblings_at_three_dupes() -> None:
    result = disambiguate_siblings(["Config", "Config", "Config"], suffix="@")
    assert result == ["Config", "Config@2", "Config@3"]


def test_disambiguate_siblings_preserves_non_dupes() -> None:
    # Non-duplicate IDs between duplicates are untouched.
    result = disambiguate_siblings(["a", "b", "a", "c", "a"], suffix="-")
    assert result == ["a", "b", "a-2", "c", "a-3"]


def test_disambiguate_siblings_empty_list() -> None:
    assert disambiguate_siblings([]) == []


def test_disambiguate_siblings_single_item() -> None:
    assert disambiguate_siblings(["only"]) == ["only"]


def test_disambiguate_siblings_skips_existing_dash_suffix() -> None:
    """Regression: input contains ``examples-2`` already → counter must skip past it.

    Per §15.3 the function MUST guarantee unique output; silently emitting
    ``examples-2`` twice (once from the input, once from the second
    ``examples`` collision) would defeat its only purpose.
    """
    result = disambiguate_siblings(["examples", "examples-2", "examples"])
    assert result == ["examples", "examples-2", "examples-3"]
    assert len(set(result)) == len(result)


def test_disambiguate_siblings_skips_existing_at_suffix() -> None:
    """Same regression for the ``@N`` (code) suffix style."""
    result = disambiguate_siblings(["Config", "Config@2", "Config"], suffix="@")
    assert result == ["Config", "Config@2", "Config@3"]
    assert len(set(result)) == len(result)


def test_disambiguate_siblings_chain_of_existing_suffixes() -> None:
    """Multiple pre-existing suffixed forms → counter walks past all of them."""
    result = disambiguate_siblings(["foo", "foo-2", "foo-3", "foo-4", "foo"])
    assert result == ["foo", "foo-2", "foo-3", "foo-4", "foo-5"]
    assert len(set(result)) == len(result)


def test_disambiguate_siblings_three_dupes_with_existing_suffix() -> None:
    """Two siblings collide AND the suffixed form already exists → both skip past."""
    result = disambiguate_siblings(["x", "x-2", "x", "x"])
    assert result == ["x", "x-2", "x-3", "x-4"]
    assert len(set(result)) == len(result)


# ──────────────────────────────────────────────────────────────────────
# derive_anonymous_code_in_doc_id (§15.3 row 4)
# ──────────────────────────────────────────────────────────────────────


def test_anonymous_code_in_doc_id_format() -> None:
    """§15.3: anonymous block → ``<section-id>::block-<8hex>``."""
    aid = derive_anonymous_code_in_doc_id("docs/spec.md::examples", "print('hi')")
    assert aid.startswith("docs/spec.md::examples::block-")
    suffix = aid.rsplit("block-", 1)[-1]
    assert len(suffix) == 8
    assert all(c in "0123456789abcdef" for c in suffix)


def test_anonymous_code_in_doc_id_deterministic_after_canonicalization() -> None:
    """The hash is computed over canonicalized content — equal under whitespace canonicalization."""
    a = derive_anonymous_code_in_doc_id("s", "print(1)\r\n")
    b = derive_anonymous_code_in_doc_id("s", "print(1)\n")
    c = derive_anonymous_code_in_doc_id("s", "print(1)  \n")  # trailing spaces stripped
    assert a == b == c


def test_anonymous_code_in_doc_id_different_bodies_differ() -> None:
    a = derive_anonymous_code_in_doc_id("s", "print(1)")
    b = derive_anonymous_code_in_doc_id("s", "print(2)")
    assert a != b


# ──────────────────────────────────────────────────────────────────────
# derive_code_id defensive guard
# ──────────────────────────────────────────────────────────────────────


def test_derive_code_id_empty_symbol_path_raises() -> None:
    """Defensive: empty symbol_path would produce malformed `<path>:` ID."""
    with pytest.raises(ValueError, match="non-empty symbol_path"):
        derive_code_id("src/foo.py", [])


# ──────────────────────────────────────────────────────────────────────
# canonicalize_content (§5.4)
# ──────────────────────────────────────────────────────────────────────


def test_canonicalize_bom_stripped() -> None:
    """§5.4 step 1: UTF-8 BOM (U+FEFF) is removed."""
    text = "\ufeffhello\n"
    assert canonicalize_content(text) == "hello\n"


def test_canonicalize_bom_absent_unchanged() -> None:
    text = "no bom here\n"
    assert canonicalize_content(text) == text


def test_canonicalize_crlf_to_lf() -> None:
    """§5.4 step 2a: CRLF → LF."""
    assert canonicalize_content("line1\r\nline2\r\n") == "line1\nline2\n"


def test_canonicalize_lf_only_unchanged() -> None:
    """§5.4: LF-only files are not modified by the line-ending step."""
    text = "line1\nline2\n"
    assert canonicalize_content(text) == text


def test_canonicalize_mixed_crlf_and_lf() -> None:
    """§15.4: mixed CRLF + LF → all LF."""
    mixed = "line1\r\nline2\nline3\r\n"
    assert canonicalize_content(mixed) == "line1\nline2\nline3\n"


def test_canonicalize_cr_only() -> None:
    """§15.4: CR-only (legacy Mac OS 9) → LF."""
    assert canonicalize_content("line1\rline2\r") == "line1\nline2\n"


def test_canonicalize_trailing_whitespace_per_line() -> None:
    """§5.4 step 3: trailing spaces/tabs on each line stripped."""
    text = "foo   \n  bar\t\nbaz\n"
    assert canonicalize_content(text) == "foo\n  bar\nbaz\n"


def test_canonicalize_trailing_whitespace_on_last_line() -> None:
    text = "hello   "
    assert canonicalize_content(text) == "hello"


def test_canonicalize_multiple_trailing_newlines_collapsed() -> None:
    """§5.4 step 4: multiple trailing newlines → exactly one."""
    assert canonicalize_content("content\n\n\n") == "content\n"
    assert canonicalize_content("content\n\n") == "content\n"


def test_canonicalize_no_trailing_newline_preserved() -> None:
    """§5.4 step 4: if original has no trailing newline, none is added."""
    text = "no newline at end"
    result = canonicalize_content(text)
    assert result == "no newline at end"
    assert not result.endswith("\n")


def test_canonicalize_one_trailing_newline_preserved() -> None:
    """§5.4 step 4: exactly one trailing newline is preserved as-is."""
    text = "exactly one\n"
    assert canonicalize_content(text) == "exactly one\n"


def test_canonicalize_internal_whitespace_not_collapsed() -> None:
    """§5.4: internal whitespace runs are NOT touched.
    A paragraph with multiple spaces survives round-trip unchanged."""
    text = "word1  word2   word3\n"  # double and triple spaces
    result = canonicalize_content(text)
    assert result == text, "internal whitespace must not be collapsed"


def test_canonicalize_idempotent_lf_text() -> None:
    """canonicalize_content(canonicalize_content(x)) == canonicalize_content(x)."""
    inputs = [
        "simple text\n",
        "foo  \r\n bar\r\n\r\n",
        "no newline",
        "\n\n\n",
        "",
        "\ufeffbom text\r\n",
        "   trailing spaces   \n",
        "a\tb\tc\n",
    ]
    for original in inputs:
        first = canonicalize_content(original)
        second = canonicalize_content(first)
        assert first == second, f"not idempotent for {original!r}: {first!r} != {second!r}"


def test_canonicalize_empty_string() -> None:
    assert canonicalize_content("") == ""


def test_canonicalize_only_newlines() -> None:
    # A file consisting entirely of blank lines collapses to a single "\n".
    assert canonicalize_content("\n\n\n") == "\n"


def test_canonicalize_bom_plus_crlf_plus_trailing() -> None:
    """All three main steps applied together."""
    text = "\ufeffhello  \r\n  world\r\n\r\n"
    expected = "hello\n  world\n"
    assert canonicalize_content(text) == expected


# ──────────────────────────────────────────────────────────────────────
# content_hash (§5.4)
# ──────────────────────────────────────────────────────────────────────


def test_content_hash_format() -> None:
    """Hash must match the ContentHash regex: sha256:[0-9a-f]{64}."""
    h = content_hash("hello world\n")
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", h), f"bad format: {h!r}"


def test_content_hash_deterministic() -> None:
    assert content_hash("foo\n") == content_hash("foo\n")


def test_content_hash_canonicalizes_before_hashing() -> None:
    # CRLF and LF variants of the same content must hash identically.
    assert content_hash("foo\r\nbar\r\n") == content_hash("foo\nbar\n")


def test_content_hash_different_inputs_differ() -> None:
    assert content_hash("aaa\n") != content_hash("bbb\n")


# ──────────────────────────────────────────────────────────────────────
# short_content_hash (§15.3)
# ──────────────────────────────────────────────────────────────────────


def test_short_content_hash_default_length() -> None:
    h = short_content_hash("hello\n")
    assert len(h) == 8
    assert re.fullmatch(r"[0-9a-f]{8}", h), f"bad format: {h!r}"


def test_short_content_hash_custom_length() -> None:
    h = short_content_hash("hello\n", length=16)
    assert len(h) == 16


def test_short_content_hash_length_4() -> None:
    h = short_content_hash("x", length=4)
    assert len(h) == 4


def test_short_content_hash_is_prefix_of_full_hash() -> None:
    text = "test content\n"
    full = content_hash(text)
    short = short_content_hash(text)
    # short is the first 8 hex chars of the full sha256 digest
    assert full.startswith(f"sha256:{short}")


# ──────────────────────────────────────────────────────────────────────
# fingerprint_simhash (§3.3)
# ──────────────────────────────────────────────────────────────────────


def test_fingerprint_simhash_returns_int() -> None:
    result = fingerprint_simhash("hello world\n")
    assert isinstance(result, int)


def test_fingerprint_simhash_deterministic() -> None:
    text = "some repeated content\n"
    assert fingerprint_simhash(text) == fingerprint_simhash(text)


def test_fingerprint_simhash_different_texts_differ() -> None:
    # Two very different texts should (almost certainly) have different hashes.
    h1 = fingerprint_simhash("The quick brown fox\n")
    h2 = fingerprint_simhash("SELECT * FROM users WHERE admin = 1\n")
    assert h1 != h2


def test_fingerprint_simhash_canonicalizes() -> None:
    # CRLF and LF variants must hash identically (canonicalization applied).
    assert fingerprint_simhash("foo\r\nbar\r\n") == fingerprint_simhash("foo\nbar\n")


def test_fingerprint_simhash_empty_string() -> None:
    # Should not raise; returns some integer.
    result = fingerprint_simhash("")
    assert isinstance(result, int)


def test_fingerprint_simhash_does_not_overflow_on_repeated_tokens() -> None:
    """Regression (B5 — dogfood bug): the upstream simhash library multiplies
    a numpy uint8 bitarray by an int weight, which raises ``OverflowError``
    on numpy 2.x for any weight > 255.  ``fingerprint_simhash`` must catch
    that case and fall back to a deterministic SHA-256-derived fingerprint
    so the indexer never aborts on long documents.
    """
    # Construct a string with a single token repeated thousands of times —
    # this is what triggers the overflow in real-world markdown corpora.
    text = ("scry " * 5000).strip()
    result = fingerprint_simhash(text)
    assert isinstance(result, int)
    # Determinism: identical input -> identical output (true for both the
    # SimHash fast path and the SHA-256 fallback).
    assert fingerprint_simhash(text) == result


# ──────────────────────────────────────────────────────────────────────
# parse_html_comment_id (§3.2 escape hatch)
# ──────────────────────────────────────────────────────────────────────


def test_parse_html_comment_id_positive_basic() -> None:
    assert parse_html_comment_id("<!-- scry-id: rule-structure -->") == "rule-structure"


def test_parse_html_comment_id_positive_no_space() -> None:
    assert parse_html_comment_id("<!--scry-id:my-slug-->") == "my-slug"


def test_parse_html_comment_id_positive_extra_spaces() -> None:
    assert parse_html_comment_id("<!--  scry-id:   foo-bar  -->") == "foo-bar"


def test_parse_html_comment_id_preserves_case() -> None:
    """Slug is returned EXACT — no lowercasing (§3.2: round-trip)."""
    assert parse_html_comment_id("<!-- scry-id: My-Custom-ID -->") == "My-Custom-ID"


def test_parse_html_comment_id_in_line_with_surrounding_text() -> None:
    line = "## Heading\n<!-- scry-id: override-slug -->\n"
    assert parse_html_comment_id(line) == "override-slug"


def test_parse_html_comment_id_negative_no_comment() -> None:
    assert parse_html_comment_id("## Rule Structure") is None


def test_parse_html_comment_id_negative_regular_comment() -> None:
    assert parse_html_comment_id("<!-- just a regular comment -->") is None


def test_parse_html_comment_id_negative_empty_string() -> None:
    assert parse_html_comment_id("") is None


def test_parse_html_comment_id_negative_empty_slug() -> None:
    # An empty scry-id value should return None (no slug).
    assert parse_html_comment_id("<!-- scry-id:  -->") is None


def test_parse_html_comment_id_returns_none_not_empty_string() -> None:
    result = parse_html_comment_id("no comment here")
    assert result is None

# uat-r5-5 pr-d noise
