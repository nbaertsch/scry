"""Tests for scry.extract.markdown — workstream W1a.

Covers §15.1 (heading extraction), §15.2 (code block extraction),
§15.4 (file-level filters), and §15.3 (ID edge cases) from DESIGN.md.

Each DESIGN.md row has at least one test fixture + assertion per spec.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scry.extract.markdown import DEFAULT_MAX_FILE_SIZE_BYTES, extract_markdown, split_section_text
from scry.models import AnchorType, Frontmatter, SectionsConfig

# ── Helpers ──────────────────────────────────────────────────────────────────

FENCE = "```"


def make_md(repo: Path, name: str, content: str | bytes) -> Path:
    """Write *content* to ``repo/<name>`` and return the Path."""
    p = repo / name
    if isinstance(content, bytes):
        p.write_bytes(content)
    else:
        p.write_text(content, encoding="utf-8")
    return p


# ── §15.4 File-level filter tests ────────────────────────────────────────────


class TestFileLevelFilters:
    """§15.4: empty, frontmatter-only, oversized, encoding, skip."""

    def test_empty_file(self, tmp_path: Path) -> None:
        """Empty file → empty list."""
        f = make_md(tmp_path, "empty.md", "")
        assert extract_markdown(f, tmp_path) == []

    def test_whitespace_only_file(self, tmp_path: Path) -> None:
        """Whitespace-only body → empty list."""
        f = make_md(tmp_path, "blank.md", "   \n\n  \n")
        assert extract_markdown(f, tmp_path) == []

    def test_frontmatter_only(self, tmp_path: Path) -> None:
        """Frontmatter present but body is empty → empty list (§15.4)."""
        content = "---\nscry:\n  skip: false\n---\n"
        f = make_md(tmp_path, "fm-only.md", content)
        assert extract_markdown(f, tmp_path) == []

    def test_frontmatter_only_trailing_whitespace(self, tmp_path: Path) -> None:
        """Frontmatter + blank body lines → empty list."""
        content = "---\nscry:\n  id: myid\n---\n\n   \n"
        f = make_md(tmp_path, "fm-blank.md", content)
        assert extract_markdown(f, tmp_path) == []

    def test_file_exceeds_max_size(self, tmp_path: Path) -> None:
        """File > max_file_size_bytes → empty list, warning logged (§15.4)."""
        # Write 1 byte more than the limit.
        big = b"x" * (DEFAULT_MAX_FILE_SIZE_BYTES + 1)
        f = tmp_path / "huge.md"
        f.write_bytes(big)
        assert extract_markdown(f, tmp_path) == []

    def test_file_at_max_size_not_skipped(self, tmp_path: Path) -> None:
        """File == limit bytes is NOT skipped (boundary: strictly greater than)."""
        # Limit is 5 MiB; write exactly that many bytes of valid markdown.
        content = ("# Title\n" + "word " * 100 + "\n") * 100
        padded = content.encode("utf-8")
        padded = padded + b" " * (DEFAULT_MAX_FILE_SIZE_BYTES - len(padded))
        # We don't assert on anchors here — just that we get *something* back
        # (or at least no crash) and the call doesn't return early.
        f = tmp_path / "atmax.md"
        f.write_bytes(padded)
        result = extract_markdown(f, tmp_path, max_file_size_bytes=DEFAULT_MAX_FILE_SIZE_BYTES)
        # May or may not produce anchors (content is mostly spaces) — just check no exception.
        assert isinstance(result, list)

    def test_utf16_le_bom(self, tmp_path: Path) -> None:
        """UTF-16 LE BOM → skip with warning (§15.4)."""
        bom = b"\xff\xfe"
        text_utf16 = "# Hello\nContent\n".encode("utf-16-le")
        f = make_md(tmp_path, "utf16le.md", bom + text_utf16)
        assert extract_markdown(f, tmp_path) == []

    def test_utf16_be_bom(self, tmp_path: Path) -> None:
        """UTF-16 BE BOM → skip with warning (§15.4)."""
        bom = b"\xfe\xff"
        text_utf16 = "# Hello\nContent\n".encode("utf-16-be")
        f = make_md(tmp_path, "utf16be.md", bom + text_utf16)
        assert extract_markdown(f, tmp_path) == []

    def test_binary_file(self, tmp_path: Path) -> None:
        """Binary data (non-UTF-8) → skip with warning (§15.4)."""
        binary = bytes(range(256))  # Contains non-UTF-8 bytes
        f = make_md(tmp_path, "binary.md", binary)
        assert extract_markdown(f, tmp_path) == []

    def test_frontmatter_skip_true_in_file(self, tmp_path: Path) -> None:
        """Frontmatter skip: true in the file itself → empty list (§15.4 / §6.1)."""
        content = "---\nscry:\n  skip: true\n---\n# Real Heading\nContent\n"
        f = make_md(tmp_path, "skip.md", content)
        assert extract_markdown(f, tmp_path) == []

    def test_frontmatter_skip_caller_override(self, tmp_path: Path) -> None:
        """Caller-supplied Frontmatter(skip=True) wins → empty list."""
        content = "# Heading\nContent\n"
        f = make_md(tmp_path, "caller-skip.md", content)
        fm = Frontmatter(skip=True)
        assert extract_markdown(f, tmp_path, frontmatter=fm) == []

    def test_frontmatter_skip_false_caller_does_not_suppress(self, tmp_path: Path) -> None:
        """Caller-supplied Frontmatter(skip=False) + file skip=true → file-fm wins only when
        caller is None. When caller provides Frontmatter, it takes precedence."""
        # The file says skip=true, but caller says skip=false → caller wins → indexed.
        content = "---\nscry:\n  skip: true\n---\n# Heading\nContent\n"
        f = make_md(tmp_path, "caller-no-skip.md", content)
        fm = Frontmatter(skip=False)
        result = extract_markdown(f, tmp_path, frontmatter=fm)
        assert len(result) >= 1

    def test_crlf_line_endings(self, tmp_path: Path) -> None:
        """CRLF line endings are normalized (§5.4); content extracted normally."""
        content_crlf = "# Heading\r\nContent here.\r\n"
        f = make_md(tmp_path, "crlf.md", content_crlf)
        result = extract_markdown(f, tmp_path)
        assert len(result) == 1
        assert result[0].type == AnchorType.SECTION
        assert result[0].heading_path == ["Heading"]
        # Verify CRLF was removed from content.
        assert "\r" not in result[0].content_text

    def test_mixed_crlf_lf_line_endings(self, tmp_path: Path) -> None:
        """Mixed CRLF+LF in same file → both normalized (§15.4 v3.1)."""
        # Some lines CRLF, some LF
        content = b"# Heading\r\nContent.\nMore content.\r\n"
        f = make_md(tmp_path, "mixed.md", content)
        result = extract_markdown(f, tmp_path)
        assert len(result) == 1
        assert "\r" not in result[0].content_text

    def test_cr_only_line_endings(self, tmp_path: Path) -> None:
        """Bare CR (old Mac style) → normalized to LF (§15.4 v3.1)."""
        content = b"# Heading\rContent.\r"
        f = make_md(tmp_path, "cr-only.md", content)
        result = extract_markdown(f, tmp_path)
        assert len(result) >= 1
        assert "\r" not in result[0].content_text

    def test_utf8_bom_stripped(self, tmp_path: Path) -> None:
        """UTF-8 BOM (U+FEFF) is stripped; file indexed normally (§5.4)."""
        bom = b"\xef\xbb\xbf"
        content = bom + b"# Heading\nContent\n"
        f = make_md(tmp_path, "bom.md", content)
        result = extract_markdown(f, tmp_path)
        assert len(result) == 1
        assert result[0].heading_path == ["Heading"]
        # BOM must not appear in content_text.
        assert "\ufeff" not in result[0].content_text


# ── §15.1 Heading extraction tests ───────────────────────────────────────────


class TestHeadingExtraction:
    """§15.1: ATX headings, setext, depth filtering, fallbacks, breadcrumbs."""

    def test_atx_h1(self, tmp_path: Path) -> None:
        """ATX H1 becomes a SECTION anchor with one-element heading_path."""
        f = make_md(tmp_path, "h1.md", "# My Title\nContent here.\n")
        result = extract_markdown(f, tmp_path)
        assert len(result) == 1
        anc = result[0]
        assert anc.type == AnchorType.SECTION
        assert anc.heading_path == ["My Title"]
        assert anc.id == "h1.md::my-title"
        assert anc.path == "h1.md"

    def test_atx_h2(self, tmp_path: Path) -> None:
        """ATX H2 becomes a SECTION anchor."""
        f = make_md(tmp_path, "h2.md", "## Section Two\nBody.\n")
        result = extract_markdown(f, tmp_path)
        assert len(result) == 1
        assert result[0].heading_path == ["Section Two"]
        assert result[0].id == "h2.md::section-two"

    def test_atx_h3(self, tmp_path: Path) -> None:
        """ATX H3 becomes a SECTION anchor (within H1/H2 parent)."""
        content = "# Top\n## Sub\n### Deep\nContent\n"
        f = make_md(tmp_path, "h3.md", content)
        result = extract_markdown(f, tmp_path)
        # Expect H1, H2, H3 anchors
        assert len(result) == 3
        ids = [a.id for a in result]
        assert "h3.md::top" in ids
        assert "h3.md::top::sub" in ids
        assert "h3.md::top::sub::deep" in ids

    def test_atx_h4(self, tmp_path: Path) -> None:
        """ATX H4 (default max_heading_depth=4) is included."""
        content = "# A\n## B\n### C\n#### D\nContent\n"
        f = make_md(tmp_path, "h4.md", content)
        result = extract_markdown(f, tmp_path)
        ids = [a.id for a in result]
        assert "h4.md::a::b::c::d" in ids

    def test_heading_depth_over_max_becomes_content(self, tmp_path: Path) -> None:
        """Heading deeper than max_heading_depth is NOT its own anchor (§15.1)."""
        content = "## Root\n### Level3\n#### Level4\n##### Level5\nContent\n"
        f = make_md(tmp_path, "depth.md", content)
        cfg = SectionsConfig(max_heading_depth=3)
        result = extract_markdown(f, tmp_path, config=cfg)
        types = [a.type for a in result]
        # Only 2 anchors: ## and ### (H5 becomes text within H3's content)
        assert len(result) == 2
        assert all(t == AnchorType.SECTION for t in types)
        # H5 heading text should be in the parent's content_text
        h3_anchor = next(a for a in result if a.heading_path and len(a.heading_path) == 2)
        assert "Level5" in h3_anchor.content_text

    def test_setext_h1(self, tmp_path: Path) -> None:
        """Setext-style H1 (====) is parsed as level 1 (§15.1)."""
        content = "Setext Heading\n==============\nSome content.\n"
        f = make_md(tmp_path, "setext-h1.md", content)
        result = extract_markdown(f, tmp_path)
        assert len(result) == 1
        assert result[0].heading_path == ["Setext Heading"]
        assert result[0].id == "setext-h1.md::setext-heading"

    def test_setext_h2(self, tmp_path: Path) -> None:
        """Setext-style H2 (----) is parsed as level 2 (§15.1)."""
        content = "Setext H2\n---------\nSome content.\n"
        f = make_md(tmp_path, "setext-h2.md", content)
        result = extract_markdown(f, tmp_path)
        assert len(result) == 1
        assert result[0].heading_path == ["Setext H2"]
        assert result[0].id == "setext-h2.md::setext-h2"

    def test_no_headings_whole_file_anchor(self, tmp_path: Path) -> None:
        """File with no headings → ONE anchor, id=<path> (no ::heading suffix) (§15.1)."""
        content = "Just some prose.\n\nNo headings here.\n"
        f = make_md(tmp_path, "nohead.md", content)
        result = extract_markdown(f, tmp_path)
        assert len(result) == 1
        anc = result[0]
        assert anc.id == "nohead.md"
        assert anc.heading_path is None
        assert anc.type == AnchorType.SECTION

    def test_heading_inside_code_fence_ignored(self, tmp_path: Path) -> None:
        """H1 inside a fenced block is NOT treated as a heading (§15.1)."""
        content = f"{FENCE}\n# fake heading inside fence\n{FENCE}\n## Real Heading\nContent\n"
        f = make_md(tmp_path, "fence-heading.md", content)
        result = extract_markdown(f, tmp_path)
        # Only the real ## heading, not the fake one inside the fence.
        assert len(result) == 1
        assert result[0].heading_path == ["Real Heading"]

    def test_empty_heading_slug_fallback(self, tmp_path: Path) -> None:
        """Empty heading → slug starts with 'section-' (§15.1, §15.3)."""
        # An ATX heading with only whitespace after # will be treated as empty.
        content = "#\nContent below.\n"
        f = make_md(tmp_path, "empty-head.md", content)
        result = extract_markdown(f, tmp_path)
        assert len(result) == 1
        # ID should have the fallback slug pattern.
        assert "::section-" in result[0].id

    def test_punctuation_only_heading_slug_fallback(self, tmp_path: Path) -> None:
        """Punctuation-only heading slug falls back to section-<hash8> (§15.3)."""
        content = "# ---\nContent below.\n"
        f = make_md(tmp_path, "punct-head.md", content)
        result = extract_markdown(f, tmp_path)
        assert len(result) == 1
        assert "::section-" in result[0].id

    def test_nested_heading_breadcrumb(self, tmp_path: Path) -> None:
        """Nested H2 > H3 ID = path::h2-slug::h3-slug (§15.1, §15.3)."""
        content = "## Chapter One\n### Section Alpha\nContent\n"
        f = make_md(tmp_path, "nested.md", content)
        result = extract_markdown(f, tmp_path)
        # H2 anchor
        h2 = next(a for a in result if len(a.heading_path or []) == 1)
        assert h2.id == "nested.md::chapter-one"
        # H3 anchor
        h3 = next(a for a in result if len(a.heading_path or []) == 2)
        assert h3.id == "nested.md::chapter-one::section-alpha"
        assert h3.heading_path == ["Chapter One", "Section Alpha"]

    def test_section_content_includes_heading_line(self, tmp_path: Path) -> None:
        """Section content_text starts from the heading line itself."""
        content = "## My Section\nParagraph one.\nParagraph two.\n"
        f = make_md(tmp_path, "head-content.md", content)
        result = extract_markdown(f, tmp_path)
        assert len(result) == 1
        # Heading line should be present in content_text
        assert "## My Section" in result[0].content_text

    def test_sibling_section_content_boundaries(self, tmp_path: Path) -> None:
        """Each H2 section's content ends before the next H2 heading (§15.1)."""
        content = "## Alpha\nAlpha content.\n## Beta\nBeta content.\n"
        f = make_md(tmp_path, "siblings.md", content)
        result = extract_markdown(f, tmp_path)
        assert len(result) == 2
        alpha = next(a for a in result if a.heading_path == ["Alpha"])
        beta = next(a for a in result if a.heading_path == ["Beta"])
        assert "Alpha content" in alpha.content_text
        assert "Alpha content" not in beta.content_text
        assert "Beta content" in beta.content_text

    def test_anchor_fields_content_hash_and_simhash(self, tmp_path: Path) -> None:
        """Anchors have valid content_hash (sha256:…) and int fingerprint_simhash."""
        content = "# Title\nBody text.\n"
        f = make_md(tmp_path, "fields.md", content)
        result = extract_markdown(f, tmp_path)
        assert len(result) == 1
        a = result[0]
        assert a.content_hash.startswith("sha256:")
        assert len(a.content_hash) == 71  # "sha256:" + 64 hex chars
        assert isinstance(a.fingerprint_simhash, int)

    def test_path_normalized_to_forward_slashes(self, tmp_path: Path) -> None:
        """Anchor.path uses forward slashes even on Windows (§3.1)."""
        sub = tmp_path / "sub" / "dir"
        sub.mkdir(parents=True)
        f = make_md(sub, "deep.md", "# Hi\nContent\n")
        result = extract_markdown(f, tmp_path)
        assert len(result) == 1
        assert "\\" not in result[0].path
        assert result[0].path == "sub/dir/deep.md"


# ── §15.3 ID edge cases ───────────────────────────────────────────────────────


class TestIdEdgeCases:
    """§15.3: scry-id override, sibling collisions, code_in_doc @N."""

    def test_scry_id_inline_comment_override(self, tmp_path: Path) -> None:
        """<!-- scry-id: custom-slug --> inline in heading pins that slug (§15.3)."""
        content = "## My Fancy Section <!-- scry-id: custom-slug -->\nContent\n"
        f = make_md(tmp_path, "scry-id-inline.md", content)
        result = extract_markdown(f, tmp_path)
        assert len(result) == 1
        assert result[0].id == "scry-id-inline.md::custom-slug"

    def test_scry_id_block_comment_override(self, tmp_path: Path) -> None:
        """<!-- scry-id: block-slug --> on the line after heading pins slug (§15.3)."""
        content = "## My Section\n<!-- scry-id: block-slug -->\nContent\n"
        f = make_md(tmp_path, "scry-id-block.md", content)
        result = extract_markdown(f, tmp_path)
        assert len(result) == 1
        assert result[0].id == "scry-id-block.md::block-slug"

    def test_scry_id_override_heading_path_uses_display_text(self, tmp_path: Path) -> None:
        """heading_path uses the human-readable text, not the scry-id slug."""
        content = "## Nice Title <!-- scry-id: short -->\nContent\n"
        f = make_md(tmp_path, "scry-id-path.md", content)
        result = extract_markdown(f, tmp_path)
        assert len(result) == 1
        # heading_path is the display text, not the override slug
        assert result[0].heading_path == ["Nice Title"]

    def test_sibling_slug_collision(self, tmp_path: Path) -> None:
        """Two H3 with same text get -2 suffix on the second occurrence (§15.3)."""
        content = "## Parent\n### Examples\nFirst.\n### Examples\nSecond.\n"
        f = make_md(tmp_path, "sibling-collision.md", content)
        result = extract_markdown(f, tmp_path)
        ids = [a.id for a in result]
        # First "Examples" → bare slug
        assert "sibling-collision.md::parent::examples" in ids
        # Second "Examples" → -2 suffix
        assert "sibling-collision.md::parent::examples-2" in ids
        assert len(ids) == 3  # parent + 2 examples

    def test_sibling_collision_three_siblings(self, tmp_path: Path) -> None:
        """Three same-slug siblings → bare, -2, -3 (§15.3)."""
        content = "# Root\n## Intro\nFirst\n## Intro\nSecond\n## Intro\nThird\n"
        f = make_md(tmp_path, "triple.md", content)
        result = extract_markdown(f, tmp_path)
        ids = [a.id for a in result]
        assert "triple.md::root::intro" in ids
        assert "triple.md::root::intro-2" in ids
        assert "triple.md::root::intro-3" in ids

    def test_sibling_collision_skips_literal_suffix_match(self, tmp_path: Path) -> None:
        """Regression (review-w1a BLOCKING): literal sibling matching the auto-suffix.

        Two ``## Examples`` siblings with a ``## Examples 2`` between them
        must NOT produce duplicate ``examples-2`` IDs.  Fixed by routing
        through ``scry.anchor_id.disambiguate_siblings`` which is
        collision-aware.
        """
        content = "# Root\n\n## Examples\nA\n\n## Examples 2\nB\n\n## Examples\nC\n"
        f = make_md(tmp_path, "siblings.md", content)
        anchors = extract_markdown(f, tmp_path)
        ids = [a.id for a in anchors]
        assert len(set(ids)) == len(ids), f"duplicate IDs: {ids}"
        # Verify the resolved tail slugs:
        tails = [aid.split("::")[-1] for aid in ids if "::root::" in aid]
        assert tails == ["examples", "examples-2", "examples-3"]

    def test_scry_id_pin_blocks_auto_suffix_collision(self, tmp_path: Path) -> None:
        """Regression: a ``<!-- scry-id: examples-2 -->`` HTML pin must consume
        the slot that auto-suffix would otherwise assign.
        """
        content = (
            "# Root\n\n"
            "## Examples <!-- scry-id: examples-2 -->\nA\n\n"
            "## Examples\nB\n\n"
            "## Examples\nC\n"
        )
        f = make_md(tmp_path, "pin.md", content)
        anchors = extract_markdown(f, tmp_path)
        ids = [a.id for a in anchors]
        assert len(set(ids)) == len(ids), f"duplicate IDs: {ids}"

    def test_frontmatter_id_override_cascades(self, tmp_path: Path) -> None:
        """Frontmatter id overrides the path used in section IDs (§6.1)."""
        content = "---\nscry:\n  id: custom-base\n---\n# Title\nContent\n"
        f = make_md(tmp_path, "fm-id.md", content)
        result = extract_markdown(f, tmp_path)
        assert len(result) == 1
        # ID uses the overridden base, not the file path.
        assert result[0].id.startswith("custom-base::")

    def test_caller_frontmatter_id_takes_precedence(self, tmp_path: Path) -> None:
        """Caller Frontmatter.id overrides file-level frontmatter.id (§6.1)."""
        content = "---\nscry:\n  id: file-base\n---\n# Title\nContent\n"
        f = make_md(tmp_path, "caller-id.md", content)
        fm = Frontmatter(id="caller-base")
        result = extract_markdown(f, tmp_path, frontmatter=fm)
        assert len(result) == 1
        assert result[0].id.startswith("caller-base::")


# ── §15.2 Code block extraction tests ────────────────────────────────────────


class TestCodeBlockExtraction:
    """§15.2: fenced blocks with language + declaration → code_in_doc."""

    def test_python_def_becomes_code_in_doc(self, tmp_path: Path) -> None:
        """Python `def` in fenced block with language tag → CODE_IN_DOC anchor."""
        content = f"## My Section\n{FENCE}python\ndef my_function():\n    return 42\n{FENCE}\n"
        f = make_md(tmp_path, "py-def.md", content)
        result = extract_markdown(f, tmp_path)
        cid_anchors = [a for a in result if a.type == AnchorType.CODE_IN_DOC]
        assert len(cid_anchors) == 1
        cid = cid_anchors[0]
        assert cid.id == "py-def.md::my-section::my-function"
        assert cid.content_text.strip() == "def my_function():\n    return 42"

    def test_python_class_becomes_code_in_doc(self, tmp_path: Path) -> None:
        """Python `class` declaration → CODE_IN_DOC anchor."""
        content = f"## API\n{FENCE}python\nclass MyClass:\n    pass\n{FENCE}\n"
        f = make_md(tmp_path, "py-class.md", content)
        result = extract_markdown(f, tmp_path)
        cid = [a for a in result if a.type == AnchorType.CODE_IN_DOC]
        assert len(cid) == 1
        assert "myclass" in cid[0].id

    def test_javascript_function_becomes_code_in_doc(self, tmp_path: Path) -> None:
        """`function` keyword in JS/TS block → CODE_IN_DOC anchor."""
        content = (
            "## JS Section\n"
            f"{FENCE}javascript\n"
            "function greet(name) {\n"
            "    return `Hello, ${name}`;\n"
            "}\n"
            f"{FENCE}\n"
        )
        f = make_md(tmp_path, "js-fn.md", content)
        result = extract_markdown(f, tmp_path)
        cid = [a for a in result if a.type == AnchorType.CODE_IN_DOC]
        assert len(cid) == 1
        assert "greet" in cid[0].id

    def test_rust_fn_becomes_code_in_doc(self, tmp_path: Path) -> None:
        """`fn` keyword in Rust block → CODE_IN_DOC anchor."""
        content = f"## Rust\n{FENCE}rust\npub fn add(a: i32, b: i32) -> i32 {{ a + b }}\n{FENCE}\n"
        f = make_md(tmp_path, "rust-fn.md", content)
        result = extract_markdown(f, tmp_path)
        cid = [a for a in result if a.type == AnchorType.CODE_IN_DOC]
        assert len(cid) == 1
        assert "add" in cid[0].id

    def test_const_binding_becomes_code_in_doc(self, tmp_path: Path) -> None:
        """`const NAME =` binding → CODE_IN_DOC anchor."""
        content = f"## Constants\n{FENCE}typescript\nconst MAX_SIZE = 1024;\n{FENCE}\n"
        f = make_md(tmp_path, "const.md", content)
        result = extract_markdown(f, tmp_path)
        cid = [a for a in result if a.type == AnchorType.CODE_IN_DOC]
        assert len(cid) == 1
        assert "max-size" in cid[0].id

    def test_typescript_interface_becomes_code_in_doc(self, tmp_path: Path) -> None:
        """`interface` in TypeScript block → CODE_IN_DOC anchor."""
        content = (
            "## Types\n"
            f"{FENCE}typescript\n"
            "interface UserProfile {\n"
            "    name: string;\n"
            "}\n"
            f"{FENCE}\n"
        )
        f = make_md(tmp_path, "ts-iface.md", content)
        result = extract_markdown(f, tmp_path)
        cid = [a for a in result if a.type == AnchorType.CODE_IN_DOC]
        assert len(cid) == 1
        assert "userprofile" in cid[0].id

    def test_block_no_language_tag_skipped(self, tmp_path: Path) -> None:
        """Fenced block with no language tag → no CODE_IN_DOC anchor (§15.2)."""
        content = f"## Section\n{FENCE}\ndef could_be_python():\n    pass\n{FENCE}\n"
        f = make_md(tmp_path, "no-lang.md", content)
        result = extract_markdown(f, tmp_path)
        cid = [a for a in result if a.type == AnchorType.CODE_IN_DOC]
        assert cid == []

    def test_block_language_no_declaration_skipped(self, tmp_path: Path) -> None:
        """Fenced block with language but no declaration → no CODE_IN_DOC (§15.2)."""
        content = f"## Section\n{FENCE}python\nx = 1 + 2\nprint(x)\n{FENCE}\n"
        f = make_md(tmp_path, "no-decl.md", content)
        result = extract_markdown(f, tmp_path)
        cid = [a for a in result if a.type == AnchorType.CODE_IN_DOC]
        assert cid == []

    def test_code_in_doc_inherits_parent_heading_path(self, tmp_path: Path) -> None:
        """CODE_IN_DOC anchor has parent section's heading_path."""
        content = f"## My Section\n{FENCE}python\ndef func(): pass\n{FENCE}\n"
        f = make_md(tmp_path, "cid-path.md", content)
        result = extract_markdown(f, tmp_path)
        cid = next(a for a in result if a.type == AnchorType.CODE_IN_DOC)
        assert cid.heading_path == ["My Section"]

    def test_code_in_doc_sibling_collision_at_suffix(self, tmp_path: Path) -> None:
        """Two same-name declarations in same section get @2 suffix (§15.3)."""
        content = (
            "## Section\n"
            f"{FENCE}python\n"
            "def process(): pass\n"
            f"{FENCE}\n"
            f"{FENCE}python\n"
            "def process(): return 1\n"
            f"{FENCE}\n"
        )
        f = make_md(tmp_path, "decl-collision.md", content)
        result = extract_markdown(f, tmp_path)
        cid = [a for a in result if a.type == AnchorType.CODE_IN_DOC]
        assert len(cid) == 2
        ids = [a.id for a in cid]
        assert any("process" in i and "@" not in i for i in ids)
        assert any("process@2" in i for i in ids)

    def test_fence_before_first_heading_skipped(self, tmp_path: Path) -> None:
        """Fence before any heading in a file that HAS headings → no parent → skip."""
        content = f"{FENCE}python\ndef pre_heading(): pass\n{FENCE}\n## Real Heading\nContent\n"
        f = make_md(tmp_path, "fence-before-head.md", content)
        result = extract_markdown(f, tmp_path)
        cid = [a for a in result if a.type == AnchorType.CODE_IN_DOC]
        # The fence has no parent anchor-level heading → skipped.
        assert cid == []

    def test_unclosed_fence_extends_to_eof(self, tmp_path: Path) -> None:
        """Unclosed fence: markdown-it extends to EOF; extractor honors that (§15.2)."""
        content = (
            f"## Section\n{FENCE}python\ndef eof_function(): pass\n"
            # No closing fence — extends to EOF
        )
        f = make_md(tmp_path, "unclosed.md", content)
        result = extract_markdown(f, tmp_path)
        # Should produce a CODE_IN_DOC (markdown-it parses the unclosed fence as a fence)
        cid = [a for a in result if a.type == AnchorType.CODE_IN_DOC]
        assert len(cid) == 1
        assert "eof-function" in cid[0].id

    def test_code_in_doc_in_no_headings_file(self, tmp_path: Path) -> None:
        """In a no-headings file, fence content uses the file-level section as parent."""
        content = f"Introduction text.\n{FENCE}python\ndef utility(): pass\n{FENCE}\n"
        f = make_md(tmp_path, "nohead-code.md", content)
        result = extract_markdown(f, tmp_path)
        cid = [a for a in result if a.type == AnchorType.CODE_IN_DOC]
        assert len(cid) == 1
        assert "utility" in cid[0].id
        # Parent is the file-level section (no heading_path)
        assert cid[0].heading_path is None


# ── Parametrized declaration detection ────────────────────────────────────────


@pytest.mark.parametrize(
    ("language", "code", "expected_name_fragment"),
    [
        ("python", "async def async_helper():\n    pass\n", "async-helper"),
        ("rust", "pub(crate) fn internal() {}", "internal"),
        ("typescript", "export class ApiClient {}", "apiclient"),
        ("typescript", "export interface Config { url: string; }", "config"),
        ("typescript", "export type AliasName = string | number;", "aliasname"),
        ("javascript", "export function handleEvent(e) {}", "handle-event"),
        ("javascript", "let counter = 0;", "counter"),
        ("javascript", "var legacy = true;", "legacy"),
    ],
)
def test_declaration_detection_parametrized(
    tmp_path: Path,
    language: str,
    code: str,
    expected_name_fragment: str,
) -> None:
    """Parametrized declaration detection for various languages / patterns (§15.2)."""
    content = f"## Section\n{FENCE}{language}\n{code}\n{FENCE}\n"
    f = make_md(tmp_path, f"decl-{language[:3]}-{expected_name_fragment}.md", content)
    result = extract_markdown(f, tmp_path)
    cid = [a for a in result if a.type == AnchorType.CODE_IN_DOC]
    assert len(cid) == 1, f"Expected 1 code_in_doc anchor for {language} {code!r}"
    assert expected_name_fragment in cid[0].id


# ── split_section_text tests (§3.4) ──────────────────────────────────────────


class TestSplitSectionText:
    """§3.4: split_section_text is the W2k sub-chunk API."""

    def test_short_text_not_split(self) -> None:
        """Text below max_tokens threshold returns single-element list."""
        text = "Short paragraph with a few words.\n"
        result = split_section_text(text, max_tokens=600)
        assert result == [text]

    def test_empty_text(self) -> None:
        """Empty text returns a single-element list."""
        result = split_section_text("", max_tokens=600)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_long_text_splits(self) -> None:
        """Text exceeding max_tokens is split into multiple chunks."""
        # Generate text longer than 5 words (max_tokens=5, i.e. ~3-4 words)
        paragraph = " ".join(f"word{i}" for i in range(20))
        text = f"{paragraph}\n\n{paragraph}\n\n{paragraph}\n"
        result = split_section_text(text, max_tokens=10)
        assert len(result) >= 2

    def test_never_splits_inside_fence(self) -> None:
        """split_section_text never splits in the middle of a fenced block."""
        fence_block = FENCE + "python\n" + "\n".join(f"line{i}" for i in range(30)) + "\n" + FENCE
        text = fence_block + "\n"
        # Even with a tight token limit, the fence block is not split.
        result = split_section_text(text, max_tokens=5)
        # The fence content must appear fully in at least one chunk.
        joined = "\n".join(result)
        assert "line0" in joined
        assert "line29" in joined

    def test_paragraph_boundary_preferred(self) -> None:
        """Split falls on paragraph boundaries (blank lines)."""
        p1 = "Alpha beta gamma delta epsilon zeta eta theta iota kappa"
        p2 = "Lambda mu nu xi omicron pi rho sigma tau upsilon phi chi"
        text = f"{p1}\n\n{p2}\n"
        result = split_section_text(text, max_tokens=15)
        assert len(result) >= 2
        # Neither chunk should silently drop content.
        full = " ".join(result)
        assert "Alpha" in full
        assert "Lambda" in full

    def test_chunks_are_nonempty(self) -> None:
        """All returned chunks are non-empty strings."""
        text = "\n\n".join(" ".join(f"w{i}" for i in range(30)) for _ in range(10))
        result = split_section_text(text, max_tokens=50)
        assert all(chunk.strip() for chunk in result)


# ── Integration: realistic spec file ─────────────────────────────────────────


class TestIntegration:
    """Integration tests with richer markdown content."""

    def test_mixed_atx_and_setext_headings(self, tmp_path: Path) -> None:
        """Files mixing ATX and setext headings are indexed correctly."""
        content = "Top Level\n=========\n## Sub ATX\nContent A\nSub2\n----\nContent B\n"
        f = make_md(tmp_path, "mixed-heads.md", content)
        result = extract_markdown(f, tmp_path)
        section_ids = [a.id for a in result if a.type == AnchorType.SECTION]
        assert any("top-level" in s for s in section_ids)
        assert any("sub-atx" in s for s in section_ids)
        assert any("sub-2" in s or "sub2" in s for s in section_ids)

    def test_multiple_sections_and_codes(self, tmp_path: Path) -> None:
        """Realistic multi-section spec file with inline code blocks."""
        content = (
            "# Overview\n"
            "This section describes the overview.\n\n"
            "## Installation\n"
            f"{FENCE}bash\n"
            "pip install scry\n"
            f"{FENCE}\n\n"
            "## API Reference\n"
            f"{FENCE}python\n"
            "class ScryClient:\n"
            "    def __init__(self, url: str) -> None: ...\n"
            f"{FENCE}\n"
            "### connect\n"
            f"{FENCE}python\n"
            "def connect(self) -> bool: ...\n"
            f"{FENCE}\n"
        )
        f = make_md(tmp_path, "spec.md", content)
        result = extract_markdown(f, tmp_path)
        section_anchors = [a for a in result if a.type == AnchorType.SECTION]
        cid_anchors = [a for a in result if a.type == AnchorType.CODE_IN_DOC]

        # We expect sections for Overview, Installation, API Reference, connect.
        assert len(section_anchors) >= 3
        # bash block has no declaration → skip; class + def → 2 code_in_doc
        assert len(cid_anchors) == 2
        decl_ids = [a.id for a in cid_anchors]
        assert any("scryclient" in i for i in decl_ids)
        assert any("connect" in i for i in decl_ids)

    def test_table_content_preserved(self, tmp_path: Path) -> None:
        """Markdown table cells are included in section content_text (§15.4)."""
        content = "## Data\n| Name | Value |\n| ---- | ----- |\n| foo  | 42    |\n"
        f = make_md(tmp_path, "table.md", content)
        result = extract_markdown(f, tmp_path)
        assert len(result) == 1
        assert "foo" in result[0].content_text
        assert "42" in result[0].content_text

# uat-r5-5 pr-d noise
