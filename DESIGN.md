# scry — Design

> *"Scry into your codebase."*
> A local-first, hybrid-retrieval MCP server that links spec/doc text
> to AST symbols via a typed graph and detects drift between them.

---

## 1. What it is

Scry indexes a repository at three granularities and links them as a graph:

1. **Spec / doc sections** — heading-bounded markdown blocks
2. **AST symbols** — language-server-resolved functions, classes, types, etc.
3. **Code blocks inside specs** — fenced blocks within a spec section,
   treated as a hybrid first-class anchor type

These three anchor types share one embedding space and one link graph.
Hybrid retrieval combines BM25 keyword search with vector cosine
similarity via Reciprocal Rank Fusion. An MCP server exposes the
result to coding agents (Claude Code, Copilot CLI, Cursor, etc.) so
the agent can reason over both *what the spec says* and *what the
code does* in one query.

Drift between linked anchors is detected deterministically and
surfaced — including (where the language server supports it)
transitive code drift via `callHierarchy/outgoingCalls` — so engineers
and agents see when a spec and its implementing code have diverged.

---

## 2. Non-goals

- **Not** a documentation generator (Sphinx / mkdocs / Docusaurus do that).
- **Not** a project-memory scaffold (mex does that for context-window management).
- **Not** a code editor or refactoring tool — read-only retrieval and
  drift surface; the agent makes the edits.
- **Not** a search engine for arbitrary text — files participate by
  config, not by default.
- **Not** spec-driven development tooling (Spec Kit does that). Scry
  works on existing codebases that already have docs/specs in
  whatever shape their authors prefer.

---

## 3. Core primitives

### 3.1 The Anchor

Everything in scry is an **Anchor** — the smallest addressable,
embeddable, linkable unit.

| Anchor type | Source | Example primary ID |
|---|---|---|
| `section` | A heading-bounded markdown block | `docs/POLICY_ENGINE.md::policy-engine::rule-structure` |
| `code_in_doc` | A fenced code block inside a `section` | `docs/POLICY_ENGINE.md::policy-engine::rule-structure::policy-rule-class` |
| `code` | An LSP-resolved symbol from a source file | `python/hailstone/policy/engine.py:PolicyRule` |

### 3.2 Anchor identity (Layer 1 + Layer 2)

Anchor IDs come in two layers:

**Layer 1 — Primary human-readable IDs.** Path + structural location.
Used in `links.jsonl`, in MCP responses, in CLI output. Stable across
non-renaming edits.

| Anchor type | Format |
|---|---|
| `section` | `<path>::<heading-path-slug>` |
| `code_in_doc` | `<section-id>::<declaration-name>` for blocks containing a named declaration; `<section-id>::<declaration-name>@N` for same-name collisions in the same section (deterministic by file order); `<section-id>::block-<short-content-hash>` for anonymous blocks |
| `code` | `<file-path>:<qualified-symbol-path>` (e.g. `python/policy.py:OuterClass.InnerClass.method`); `:f@<sig-hash>` suffix for overloads |

Sibling-heading slug collisions get a deterministic positional
suffix (`::examples` for the first, `::examples-2` for the second).

**Layer 2 — Secondary content fingerprints.** Stored on every anchor
record in `vectors.db`. Never appear in user-facing IDs. Used at
re-index time for **inline rebase** (see §3.3).

| Anchor type | Fingerprint |
|---|---|
| `section` | `(normalized_heading_path, content_hash)` where `content_hash` is SHA-256 over canonicalized body (see §5.4) |
| `code` | `(qualified_scope, signature_hash, ast_subtree_fingerprint)` |
| `code_in_doc` | `(language, declaration_name_or_signature, code_hash)` |

Fingerprints are **independent of the embedding model** — pure SHA-256
over canonicalized text/AST.

**Escape hatch.** Authors can pin a slug independently of structural
location with an HTML comment immediately under a heading or code
block:

```markdown
## Rule Structure
<!-- scry-id: rule-structure -->
```

Duplicate `scry-id` values within a single document are an index-time
validation error.

### 3.3 Inline rebase on re-index

When `scry index` runs, it walks the repo and builds the new anchor
set with new primary IDs. For each anchor whose primary ID is missing
from the prior index but whose Layer 2 fingerprint matches a missing
old anchor, scry **rebases links forward**:

- Existing links pointing at the old ID get appended `upsert` records
  (see §3.5) with the new ID.
- The original `link_id` is preserved.
- The link's `drift_status` reflects the actual content change.

**Where the rebase records go: the overlay, never the baseline.** See
§3.5 for the baseline-vs-overlay split. `scry index` writes rebases
to the per-branch overlay file. Users explicitly run `scry
commit-links` to promote overlay records to the committed baseline.

**Fuzzy fingerprint match algorithm:**

1. **Embedding cosine similarity** — find candidate matches in the
   missing-from-new set using the same embedding model used for
   retrieval. Threshold default `0.85`, configurable.
2. **SimHash / Jaccard confirmation** — for each candidate, compute
   SimHash over the canonicalized content and confirm Jaccard
   similarity ≥ `0.7` (configurable).
3. **Same-file constraint by default**. Cross-file rebase opt-in via
   `index.cross_file_rebase: true`.
4. **No match → broken-source / broken-target**.

**Rebase under model migration** (§7.2.2) skips step 1 and uses
SimHash-only with a higher threshold, since old and new embeddings
are not comparable across models. (In practice this rarely matters
because `scry index --reembed` preserves anchors and their
fingerprints — see §7.2.)

**Hash refresh on rebase preserves the prior baseline hash.** When a
rebase upsert is written, the new record carries an additional
`prior_content_hash` field so subsequent drift checks can still see
the rename-with-edit signal as drift, not as `fresh`.

### 3.4 Two-tier embedding

Sections vary from 30 words to 8000+ words. Naive sub-chunking breaks
links. Scry separates **link granularity** (always the heading-bounded
parent anchor) from **embedding/retrieval granularity** (parent
overview + sub-chunks under the hood):

```
Parent anchor (link target):
  docs/POLICY_ENGINE.md::policy-engine::rule-structure
  ─ stable Layer 1 ID
  ─ overview embedding (heading + first ~200 tokens)
  ─ Layer 2 content fingerprint
  ─ this is what links point at and what users/agents see

Sub-chunks (internal cache, retrieval optimization):
  ::rule-structure#chunk-0, #chunk-1, ...
  ─ each sub-chunk row stores its parent's content_hash
  ─ used INTERNALLY for retrieval scoring; NEVER returned as a link target
  ─ NOT a separate drift unit
```

Sub-chunking is triggered when section text exceeds
`sections.max_tokens` (default 600). Split priority:
1. Sub-headings inside the section (deeper level than the section's own)
2. Fenced-code-block boundaries (never split inside a code block)
3. Paragraph boundaries (blank lines)
4. Sentence boundaries (final fallback)

Each sub-chunk inherits a configurable overlap (default 50 tokens) with
its predecessor.

### 3.5 The Link

A typed, directed edge between two anchors. Stored as **append-only
event records** in two layers:

| Layer | File(s) | Purpose | Git status |
|---|---|---|---|
| **Baseline** | `.scry/links.jsonl` | Committed source-of-truth link graph | Tracked, committed |
| **Overlay** | `.scry/overlays/<branch>.jsonl` | Per-branch session state (rebases, proposals, pending changes) | Gitignored |

**Active link table = `replay(baseline) ⊕ replay(overlay-for-current-branch)`.**

The overlay layer means scry can produce session-state link mutations
(auto-rebases, proposals, pending accepts) without ever mutating the
committed baseline file. Branch switching is cheap: the baseline file
swaps automatically on `git checkout` (it's a tracked file), and the
overlay file for the new branch is loaded from `.scry/overlays/`.

#### 3.5.1 Event record schema

```jsonc
// Create or update (idempotent by link_id)
{
  "op": "upsert",
  "link_id": "lnk_abc",
  "from": "python/hailstone/policy/engine.py:PolicyRule",
  "from_type": "code",
  "to": "docs/POLICY_ENGINE.md::policy-engine::rule-structure",
  "to_type": "section",
  "type": "implements",
  "from_content_hash": "sha256:...",
  "to_content_hash": "sha256:...",
  "prior_content_hash": "sha256:...",   // optional, set on rebase upserts
  "commit_sha": "abc123def456...",       // git HEAD at upsert time
  "worktree_dirty": false,                // working tree dirty at upsert?
  "supersedes": "lnk_abc",                // REQUIRED if link_id already exists in this file
  "evidence": "Optional pull-quote",
  "ts": "2026-05-08T12:34:56Z"
}

// Delete (tombstone)
{
  "op": "delete",
  "link_id": "lnk_abc",
  "ts": "2026-05-08T13:00:00Z",
  "reason": "manual"
}
```

#### 3.5.2 Replay rules

1. **Within a single file**, file order = ordering. Last record for a
   given `link_id` wins.
2. **Replay order**: baseline first, then overlay. Overlay records
   can supersede baseline records.
3. **Tombstones are absorbing within the same file** — once a `delete`
   for `link_id L` appears in a file, a subsequent `upsert` for L in
   that same file is a validation error at write time.
4. **Different files (baseline vs overlay) can revive**: an overlay
   `upsert` after a baseline `delete` is allowed and treated as "user
   re-authored after deletion."
5. **`supersedes` is required** on every `upsert` whose `link_id`
   already exists in the file being read. Validated at write time.
   Missing `supersedes` on a duplicate `link_id` = error.
6. **Post-union-merge of baseline**: the `supersedes` chain provides
   logical ordering even when file order is non-semantic. If the chain
   is well-formed, the active state is deterministic. If broken (two
   upserts both claim to supersede the same prior), surface as a
   `merge-conflict` event in `scry status`.

#### 3.5.3 Merge driver

Ship a `.gitattributes` line installing the `union` merge driver for
`.scry/links.jsonl`. Concurrent appends from different branches union
cleanly; conflicts only arise on the same line, which is impossible
by construction. The `supersedes` chain provides post-merge semantic
ordering even when union concatenates non-chronologically.

`scry init` writes the `.gitattributes` line and `scry validate`
warns if it's missing.

#### 3.5.4 Promotion: `scry commit-links`

Overlay records are session-only by default. Users explicitly promote
selected overlay records to the baseline via `scry commit-links`,
which appends the promoted records to `links.jsonl` and removes them
from the overlay.

This is the link-graph equivalent of `git add` + `git commit`:
working state is in the overlay; committed state is in the baseline.

`scry status` shows pending overlay records and lets the user inspect
each before promoting.

### 3.6 Link types (canonical vocabulary)

Storage holds links in **canonical direction**. User-facing
inverses are accepted at write time but normalized to canonical form
(from/to swapped).

| Canonical (stored) | Inverse (accepted, rendered at query time) | Direction | Meaning |
|---|---|---|---|
| `implements` | `specifies` | code → spec | This code implements this spec section |
| `tests` | `tested-by` | test code → impl/spec | This test exercises this implementation/spec |
| `examples` | `exemplified-by` | code_in_doc → code | This code-block-in-doc is an example of this symbol |
| `mirrors` | `mirrored-by` | code_in_doc → code | This code-block-in-doc is a contract for this symbol — drift between them is a hard signal |
| `derives-from` | `derives-into` | downstream-spec → upstream-spec | This spec section is downstream of a higher-level spec |
| `references` | `referenced-by` | section → section | Cross-document reference (auto-derived from markdown links) |

`get_links(X, direction=incoming)` returns stored links with their
inverse name applied — purely a query-time view.

The vocabulary lives in `.scry/config.yaml` and can be extended
per-repo. Drift checks distinguish only `mirrors` (strongest, with
embedding-distance semantic-drift) from everything else (text-hash
only).

---

## 4. Retrieval

### 4.1 Hybrid retrieval algorithm

```
search(query, anchor_types?, top_k=10):
  1. Embed query (lazy-load model on first call; see §10)
  2. Run two retrievals in parallel:
       - Vector ANN over (parent overview embeddings + sub-chunk embeddings + code anchor embeddings)
       - BM25 (FTS5) over the same set of items
  3. PROMOTE: for each parent anchor, take its best-ranked chunk in
     each list (vector and BM25). The parent's "rank" in each list
     is the rank of its best chunk in that list. (For parent anchors
     whose own overview embedding outranks all their sub-chunks,
     the overview's rank wins.) Each parent now appears EXACTLY ONCE
     in each list.
  4. RRF-fuse the two parent-ranked lists:
       parent_score = Σ_{list ∈ {vec, bm25}} 1 / (k + best_chunk_rank_in_list)
       (k from retrieval.fusion_rrf_k, default 60)
  5. Sort parents by RRF score; take top_k
  6. For each result, populate the anchor packet (§4.2):
       - Pull the best-matching sub-chunk excerpt for the evidence field
       - Pull all 1-hop neighbors from the link graph (filtered by config caps)
       - Compute drift_status for each link
```

**Why "best chunk per parent" instead of summing**: aggregating with a
sum over sub-chunks (the v1 and v2 approaches) systematically biased
ranking toward long sections with many low-scoring chunks. Promoting
the best chunk per parent eliminates the bias entirely while keeping
RRF as the unified score-fusion primitive.

**No graph-traversal influence on ranking.** Graph context comes via
the `links` field on each result (§4.2). This dissolves the `α`
tuning problem entirely (no parameter exists), keeps the
score-fusion primitive consistent (RRF everywhere), and gives agents
the graph context they need without a tuning knob.

### 4.2 The Anchor Packet

Every search result is an **anchor packet**:

```jsonc
{
  "anchor": {
    "id": "docs/POLICY_ENGINE.md::policy-engine::rule-structure",
    "type": "section",
    "path": "docs/POLICY_ENGINE.md",
    "heading_path": ["Policy Engine", "Rule Structure"],
    "content": "...up to retrieval.content_preview_tokens of section markdown...",
    "content_truncated": false,
    "content_hash": "sha256:..."
  },
  "score": 0.83,
  "evidence_excerpt": "The matched sub-chunk text",
  "links": [
    {"to": "python/hailstone/policy/engine.py:PolicyRule",
     "to_type": "code", "type": "implements",
     "drift_status": "fresh",
     "transitive_hash_status": "complete"},
    {"to": "docs/EXECUTION_PIPELINE.md::policy-middleware",
     "to_type": "section", "type": "references",
     "drift_status": "n/a"}
  ],
  "index_state": "fresh"
}
```

**`content` is bounded.** Default `retrieval.content_preview_tokens:
500` caps inline content to prevent agent context-window overflow.
When truncated, `content_truncated: true` and full content is
available via `get_anchor(id)`.

**`transitive_hash_status`** on each `code`-typed link describes the
quality of the underlying drift signal — see §5.3.

Caps on `links` per result are configurable (default 5 outgoing + 5
incoming); use `get_links(anchor_id)` for the full neighborhood.

---

## 5. Drift detection

Drift is checked **deterministically** at the **parent anchor level**.
Sub-chunks don't have independent drift state.

### 5.1 Drift signals per link

Each link's status is computed by comparing stored vs current content
hashes plus (for `code` anchors with LSP resolution) transitive
closure hashes:

| status | When |
|---|---|
| `fresh` | Both endpoints' content hashes match the stored values |
| `spec-changed` | Spec/doc endpoint content hash changed |
| `code-changed` | Code endpoint content hash changed (own AST + transitive closure where supported — see §5.3) |
| `both-changed` | Both endpoints' content hashes changed |
| `broken-source` | Source anchor no longer exists (and no rebase candidate matched) |
| `broken-target` | Target anchor no longer exists (and no rebase candidate matched) |
| `semantic-drift` | For `mirrors` links only — both endpoints' hashes refreshed but cosine distance between their embeddings exceeds `drift.semantic_drift_threshold` (default 0.25). Caught even when text-hash drift was reconciled. |
| `merge-conflict` | The link's `supersedes` chain has multiple latest upserts post-merge. User resolves via `scry status`. |

**Status precedence** (highest first):
`broken-source` / `broken-target` > `merge-conflict` >
`both-changed` > `spec-changed` / `code-changed` >
`semantic-drift` > `fresh`

### 5.2 Drift score and coverage score

Two scores computed by `scry check`:

**Drift score** (normalized, 0–100, or `null` for empty repos):
```
drift_score = 100 × (1 - Σ(weight_c × count_c) / max(1, total_links))
            (returns `null` when total_links == 0)
```

Default weights (configurable in `drift.scoring`):
| status | weight |
|---|---|
| `broken-*` | 1.0 |
| `merge-conflict` | 1.0 |
| `both-changed` | 0.5 |
| `spec-changed` | 0.3 |
| `code-changed` | 0.3 |
| `semantic-drift` | 0.2 |

Score is clamped to `[0, 100]`. Independent of repo size.

**Coverage score** (0–100, or `null` for repos with no code anchors):
```
coverage_score = 100 × (linked_code_anchors / max(1, total_code_anchors))
               (returns `null` when total_code_anchors == 0)
```

`null` distinguishes "no code indexed" from "0% of code linked." CI
policy should treat `null` as not-applicable.

**Always emitted alongside the scores: raw counts** so CI policy can
gate on counts directly:

```jsonc
{
  "drift_score": 92.3,
  "coverage_score": 67.5,
  "counts": {
    "broken_source": 1, "broken_target": 1, "merge_conflict": 0,
    "both_changed": 0, "spec_changed": 8,
    "code_changed": 1, "semantic_drift": 0,
    "fresh": 145, "total": 156
  },
  "by_anchor_type": {...}
}
```

**Recommended CI gate**: `scry check --ci --drift-min 90 --coverage-min 50`
exits non-zero if either score falls below its threshold (treating
`null` as pass).

### 5.3 Code transitive drift via LSP

`code` anchor `content_hash` includes both the anchor's own
canonicalized AST text **and** sorted hashes of definitions reachable
via `callHierarchy/outgoingCalls` from the language's LSP server (see
§6).

This catches **directly-called functions and methods within the same
repo** that have changed since the link was created.

**This is a narrow guarantee.** `callHierarchy/outgoingCalls` returns
explicit call sites in the function body. It does NOT capture:

- Base class methods inherited without explicit `super()` calls
- Imported constants used as attribute access
- Decorator-supplied behavior
- Dynamic dispatch / monkeypatching
- Framework-registered callbacks

Each `code` anchor records a **`transitive_hash_status`** in its
metadata, surfaced on every link in the anchor packet:

| value | Meaning |
|---|---|
| `complete` | LSP returned a complete `callHierarchy` result; closure includes all reachable in-repo callees |
| `partial` | LSP returned partial results (e.g., capability quirks; some symbols unresolved) |
| `unsupported` | LSP doesn't implement `callHierarchy` for this symbol type; only the anchor's own AST text is hashed |
| `lsp_error` | LSP query failed; falls back to AST-only hash |

Agents seeing `drift_status: "fresh"` with `transitive_hash_status:
"unsupported"` know the signal is weaker than `complete`.

**Closure boundary**: walks `callHierarchy/outgoingCalls` until a
called symbol is matched by the config `exclude:` globs OR is not
matched by any `include:` glob. This ties the closure boundary to
the same include/exclude system as anchor extraction — no separate
hardcoded list of `node_modules` etc.

**Opt-in full resolution**: setting
`code_anchors.transitive_resolution: full` enables an additional pass
using `textDocument/references` + symbol lookup to capture inheritance
chains and imported constants. Default is `call_only` (the narrow
guarantee above). Implementation of `full` mode is per-language and
deferred to follow-on releases.

**LSP capability is verified at startup**: scry probes each indexed
language's LSP for `callHierarchy` capability; languages without it
are flagged in `scry doctor` output and produce
`transitive_hash_status: "unsupported"` on all their code anchors.

### 5.4 Content-hash canonicalization

`content_hash` is SHA-256 over canonicalized content. Canonicalization
steps applied before hashing:

1. Strip UTF-8 BOM if present
2. Normalize all line endings to LF
3. Trim trailing whitespace per line
4. Collapse trailing newlines at EOF to a single `\n`

What is **deliberately not** done:
- No paragraph reflow
- No collapse of internal whitespace runs
- No Unicode NFC normalization

Same canonicalization applies to `section`, `code_in_doc`, and `code`
anchor content hashes. For `code`, canonicalization runs over the
canonicalized AST subtree text plus the sorted transitive-closure
hash suffix (§5.3).

### 5.5 Reconciliation (LLM-assisted, opt-in)

`scry reconcile <link_id>` runs a structured AI loop:

1. **Reconstruct the baseline** from stored `commit_sha`. If reachable
   in git history, diff `git show <commit_sha>:path` vs current file
   for both endpoints. If not reachable (rebased away), fall back to
   "cannot reconstruct original; showing current state vs
   `content_hash` mismatch only." If `worktree_dirty` was true at
   upsert, caveat the diff.
2. **Use an LLM** to propose: "spec is correct, code needs change X" OR
   "code is correct, spec needs change Y" OR "both need an update"
3. **Output the proposed patch** as a unified diff for human review.
4. **On accept**, write a new `upsert` record with refreshed hashes
   and the new `commit_sha`. The new record goes to the **overlay**;
   user runs `scry commit-links` to promote.

The deterministic `check` path never calls an LLM. AI is opt-in,
contained, and never blocks correctness signals.

`scry check` under embedding-model mismatch evaluates all
text-hash-based statuses normally and reports `semantic-drift` as
`n/a` with an explicit warning, rather than hard-failing the entire
check. Use `--require-fresh-embedder` for stricter pipelines.

---

## 6. Configuration

A repo participates by adding **`.scry/config.yaml`** at its root:

```yaml
# Minimal config — index everything markdown, plus all source code
include:
  - "**/*.md"
  - "**/*.py"
  - "**/*.ts"
  - "**/*.zig"
exclude:
  - node_modules/**
  - .scry/**
  - dist/**
  - build/**

# Per-glob classification of markdown files. ORDERED LIST.
# First-match-wins. Validated at load time (non-list = error).
classify:
  - { glob: "docs/**.md", type: spec }
  - { glob: "README.md", type: doc }
  - { glob: "**/*.md", type: doc }     # default for unclassified

# Anchor extraction tuning
sections:
  max_heading_depth: 4         # include H1-H4 as anchors; deeper = sub-chunks of nearest ancestor
  max_tokens: 600
  overlap_tokens: 50
  min_section_tokens: 0        # 0 = embed even tiny sections; raise to 30 to skip stubs

code_anchors:
  granularity: symbol          # symbol | file
  symbol_kinds:
    python: [function_definition, class_definition, decorated_definition]
    typescript: [function_declaration, class_declaration, interface_declaration, type_alias_declaration]
    zig: [FnProto, ContainerDecl]
  languages:
    python: lsp                # lsp | skip
    typescript: lsp
    zig: lsp
  transitive_resolution: call_only   # call_only | full
  # The `lsp` block is OPTIONAL and only allows arg overrides.
  # Binary names are looked up from a hardcoded allowlist (§6.2).
  # To use a non-allowlisted binary, run with --allow-untrusted-lsp-config.
  lsp:
    python: { args: [--stdio] }
    typescript: { args: [--stdio] }
    zig: { args: [] }

# Embedding provider
embeddings:
  provider: local              # local | openai | voyage | custom (OpenAI-compatible)
  model: BAAI/bge-small-en-v1.5
  dimensions: 384

# Retrieval
retrieval:
  fusion_rrf_k: 60             # RRF k for BM25 + vector list fusion
  aggregation_rrf_k: 60        # RRF k for sub-chunk-into-parent aggregation (currently unused since promotion replaces aggregation; reserved for future)
  bm25:
    enabled: true
  links_per_result:
    outgoing: 5
    incoming: 5
  content_preview_tokens: 500  # cap on the anchor packet `content` field

# Drift detection
drift:
  semantic_drift_threshold: 0.25
  scoring:
    broken: 1.0
    merge_conflict: 1.0
    both_changed: 0.5
    spec_changed: 0.3
    code_changed: 0.3
    semantic_drift: 0.2

# Indexing
index:
  cross_file_rebase: false     # opt-in: detect rename+move combos via fingerprint match
  fuzzy_match:
    embedding_threshold: 0.85
    simhash_jaccard_threshold: 0.7
    simhash_jaccard_threshold_migration: 0.85   # raised during --reembed migrations
  max_file_size_bytes: 5242880  # 5 MB
```

### 6.1 Per-file frontmatter overrides

Frontmatter is **optional**. When present, frontmatter overrides
config:

```markdown
---
scry:
  type: spec                 # overrides classify result
  id: AUTH-LOGIN              # overrides path-derived ID; uniqueness validated at index time
  skip: false                 # opt out per-file (renamed from `exclude` for clarity)
---
```

Frontmatter `skip: true` opts a file out from indexing even when
matched by an `include:` glob. Frontmatter cannot override hard
safety excludes (paths matched by the global `exclude:` list always
win, to prevent indexing of `secrets/**` etc.).

### 6.2 LSP binary allowlist (security)

To prevent arbitrary code execution from a hostile repo's
`.scry/config.yaml`, scry resolves LSP binaries by **name only**,
looked up via PATH, against a hardcoded allowlist:

| Language | Allowed binary names |
|---|---|
| `python` | `pyright-langserver`, `pylsp`, `basedpyright-langserver` |
| `typescript` / `tsx` / `javascript` / `jsx` | `typescript-language-server` |
| `zig` | `zls` |
| `go` | `gopls` |
| `rust` | `rust-analyzer` |

The `lsp` block in `.scry/config.yaml` may set `args:` for any
allowlisted language but **may NOT set `command:`**. To run a
non-allowlisted binary (development of a custom LSP, monorepo
wrapper script), pass `--allow-untrusted-lsp-config` to the scry CLI
or set `SCRY_ALLOW_UNTRUSTED_LSP_CONFIG=1`. Without the flag,
`command:` overrides are ignored with a warning.

Adding a new LSP to the allowlist requires a scry release.

---

## 7. Persistence layout and provenance

```
<repo>/
├── .scry/
│   ├── config.yaml          ← user-authored; required
│   ├── links.jsonl          ← committed baseline link graph
│   ├── overlays/            ← gitignored; per-branch session state
│   │   ├── main.jsonl
│   │   ├── feature-x.jsonl
│   │   └── ...
│   ├── vectors.db           ← embeddings + content hashes; gitignored
│   ├── leader.lock          ← gitignored; PID + IPC endpoint of the active leader (§10)
│   ├── cache/               ← extracted anchors, file hashes; gitignored
│   └── stats.json           ← drift score history; gitignored
└── (repo files)
```

`.gitignore` (auto-installed by `scry init`):
```
.scry/overlays/
.scry/vectors.db
.scry/leader.lock
.scry/cache/
.scry/stats.json
```

`.gitattributes` (auto-installed by `scry init`):
```
.scry/links.jsonl merge=union
```

### 7.1 Index provenance metadata

`vectors.db` stores a single `index_metadata` row tracking what
produced the current state:

```jsonc
{
  "indexed_git_head": "abc123def456...",
  "indexed_git_tree_hash": "tree_hash...",
  "indexed_branch": "main",
  "indexed_file_manifest": {"path/file.md": "sha256:..."},
  "config_hash": "sha256:...",
  "embedding_provider": "local",
  "embedding_model": "BAAI/bge-small-en-v1.5",
  "embedding_dimensions": 384,
  "tokenizer_version": "..."
}
```

The `indexed_branch` field is part of provenance because the active
overlay is per-branch.

### 7.2 Auto-reconcile via polling

Scry detects git context changes by **polling at every tool call**:

1. Before serving any tool call (CLI or MCP), run `git rev-parse HEAD`
   (~3-5ms warm) and `git rev-parse --abbrev-ref HEAD`.
2. Compare to `index_metadata.indexed_git_head` and `indexed_branch`.
3. **If branch changed**: swap the active overlay file (load
   `.scry/overlays/<new-branch>.jsonl`).
4. **If HEAD changed within the same branch**: detect changed files
   via `git diff --name-only <indexed_head> <current_head>` and
   incrementally reindex only those files.
5. **If config_hash changed**: full reindex (config affects extraction
   semantics).
6. **MCP responses include `"index_state"`**: one of `"fresh"`,
   `"stale-reconciling"`, `"stale-no-write-lock"`, `"stale-warned"`.

Per-process HEAD cache with configurable refresh interval (default
30s) for high-throughput operations. Polling rationale: **multi-user
consistency** — every collaborator gets the same behavior without
per-clone hook installation. Polling is also defensive against IDE
git operations that may bypass hooks.

#### 7.2.1 Embedding-model mismatch (HARD ERROR + `--reembed` recovery)

On any mismatch in `embedding_*` fields (provider, model, dimensions,
tokenizer): **HARD ERROR.** Refuse to serve. Tell user:

> *"Embedding configuration changed. Run `scry index --reembed` to
> re-embed existing anchors with the new model (preserves anchors,
> fingerprints, and links). Use `scry index --force` only for
> suspected data corruption."*

`scry index --reembed` is the surgical migration path:
- Reads all anchor records from `vectors.db` (keeps the rows; just
  blanks the `embedding` column)
- Updates `index_metadata.embedding_*` fields
- Re-embeds every anchor + sub-chunk with the new model
- Does NOT drop anchors, fingerprints, FTS5 index, links.jsonl, or
  overlays

Cost: only the embedding work; skips parsing, AST extraction, LSP
queries, hashing, link processing. Roughly 2-3× faster than `--force`
on a hailstorm-scale repo. Rebase capability is fully preserved
because anchors and fingerprints never disappear.

`scry index --force` (full nuclear rebuild) is reserved for the
genuine data-corruption / schema-migration case.

### 7.3 Two-tier embedding consistency invariants

To prevent the "stale sub-chunk attached to fresh parent hash" race:

1. Every chunk row stores its parent's `content_hash`.
2. Re-index of a parent = single SQLite transaction:
   delete ALL existing chunks for that parent, insert ALL new chunks
   tagged with the new parent hash, update parent record.
3. `index_metadata` row updates are included in the same transaction
   as the chunk inserts they describe (no torn provenance after a
   crash).
4. Search returns only chunks whose stored `parent_content_hash`
   equals the current parent's `content_hash`. Stale orphans are
   invisible.
5. SQLite opened in WAL mode explicitly. Single writer (the leader,
   §10) + multiple concurrent readers (followers, CLI, MCP queries).
6. The leader holds an advisory write lock at `.scry/vectors.db.lock`
   (`fcntl.flock` on Unix; `msvcrt.locking` on Windows) for the
   duration of any write transaction.

---

## 8. MCP tool surface

| Tool | Purpose |
|---|---|
| `search(query, types?, top_k?)` | Hybrid retrieval; returns ranked anchor packets |
| `get_anchor(id)` | Full content of an anchor by ID |
| `get_links(anchor_id, link_types?, direction?)` | Bidirectional link enumeration; inverse names rendered for `direction=incoming` |
| `find_drift(scope?, status_filter?)` | List anchors/links with drift status > `fresh` |
| `propose_link(from_id, to_id, link_type, evidence?)` | Stages a link in the overlay (§3.5.4) |
| `accept_link(proposed_id)` | Marks an overlay-staged proposal as accepted (still overlay; promote with `commit_links`) |
| `commit_links(scope?)` | Promote accepted overlay records to the baseline `links.jsonl` |
| `status()` | Return pending overlay records, merge conflicts, index state |
| `repo_summary()` | One-shot orientation: file tree, classified docs, top symbols, drift + coverage scores |
| `reindex(scope?)` | Force re-extraction (default is incremental on file change) |

CLI surface mirrors the MCP tools 1:1.

The MCP server lazy-loads the embedder model on first call to `search`
or any tool that requires query embeddings. `get_anchor`, `get_links`,
`find_drift`, `repo_summary`, `status` do not pay the model-load tax.

---

## 9. CLI surface

| Command | What it does |
|---|---|
| `scry init` | Wizard: choose embedding provider, write `.scry/config.yaml`, install `.gitignore` and `.gitattributes` (union driver for `links.jsonl`). With `--register-global` flag, also registers MCP entry in `~/.claude.json` / `~/.cursor/mcp.json` (default: prints the JSON snippet for manual paste). |
| `scry index [--force \| --reembed]` | Build/refresh the vector store and AST cache. `--reembed` re-embeds existing anchors with the current model (preserves anchors, fingerprints, links). `--force` is the nuclear rebuild — reserved for corruption / schema migration. |
| `scry watch` | Sit on a file watcher; reindex on change; coordinates with the leader process via the IPC endpoint (§10) |
| `scry check [--format json\|md] [--ci] [--drift-min N] [--coverage-min N] [--require-fresh-embedder]` | Drift + coverage scores; `--ci` exits non-zero when thresholds violated (treats `null` as pass) |
| `scry status` | Show pending overlay records (rebases, proposals), merge conflicts, index state |
| `scry commit-links [<link_id>...]` | Promote overlay records to baseline `links.jsonl` |
| `scry search "<query>" [--top-k N] [--type section\|code]` | Same as MCP `search`, prints to stdout |
| `scry link <from> <to> --type <link_type> [--evidence "..."]` | Author a link from the CLI (writes to overlay) |
| `scry suggest-links [--scope <path>] [--accept-all]` | AI-augmented batch link suggestions (opt-in; requires LLM provider) |
| `scry reconcile <link_id>` | AI-assisted patch proposal for drifted links (opt-in) |
| `scry doctor` | Diagnostics: LSP binary discovery + capability check (per-language `callHierarchy` support), embedding provider validation, vectors.db health, lock status |
| `scry validate` | Validates `.scry/config.yaml`, frontmatter `id` uniqueness, `.gitattributes` union driver presence |
| `scry mcp` | Run the stdio MCP server (no daemon required); auto-detects whether to be leader or follower (§10) |
| `--allow-untrusted-lsp-config` | Global flag enabling `command:` overrides in `.scry/config.yaml` (§6.2) |

---

## 10. Process model

**Leader-follower architecture for multi-process coordination.**
Multiple `scry mcp` (and CLI) processes can operate against the same
repo simultaneously without races, stale state, or crashes.

### 10.1 Leader / follower roles

The first `scry mcp` instance to start in a given `.scry/` becomes the
**leader**. Subsequent instances are **followers**.

| Role | Responsibilities |
|---|---|
| **Leader** | Holds `.scry/leader.lock` (PID + IPC endpoint URI). Owns all writes to `vectors.db` and overlay files. Exposes IPC endpoint (Unix socket on macOS/Linux; Windows named pipe). Performs auto-reconcile. Performs scry index operations. Serves MCP tools normally to its agent. |
| **Follower** | Detects leader alive via `.scry/leader.lock`. Opens `vectors.db` read-only. Serves all read tools (`search`, `get_anchor`, `get_links`, `find_drift`, `repo_summary`, `status`) directly from the read-only DB. Forwards write operations (`propose_link`, `accept_link`, `commit_links`, `reindex`) to the leader via IPC; reads back results. Polls git context like the leader (so search results reflect current branch correctly). |

Followers see leader writes immediately because `vectors.db` is shared
in WAL mode and concurrent readers see committed writes.

### 10.2 Leader election and failover

On startup, every `scry mcp` instance:
1. Tries to acquire an exclusive lock on `.scry/leader.lock`.
2. **If lock acquired**: instance is the leader. Writes its PID + IPC
   endpoint URI to the lock file. Starts the IPC listener.
3. **If lock held by another**: instance is a follower. Reads the lock
   file to find the leader's IPC endpoint. Connects.
4. **If lock held but the holding PID is dead** (stale lock): force
   the lock acquisition (steal). Document this in the leader's
   startup log.

On leader exit:
- Followers detect the IPC endpoint is gone (connection failure).
- Each follower attempts to acquire the leader lock. The first one
  wins; it becomes the new leader.
- The transition is transparent to agent harnesses (their `scry mcp`
  process keeps serving; only the internal role changed).

### 10.3 IPC protocol

Lightweight JSON-over-stream over Unix socket / Windows named pipe.
Request/response shapes mirror the MCP tool surface for write tools:

```jsonc
// Follower → Leader
{"id": 42, "op": "propose_link", "args": {...}}

// Leader → Follower
{"id": 42, "ok": true, "result": {...}}
{"id": 42, "ok": false, "error": "...", "error_type": "..."}
```

Timeouts default to 5 seconds per call, configurable. On timeout or
connection error, the follower reports the operation as failed to its
MCP client and retries leader detection on the next call.

### 10.4 Cold start (no other process running)

When the only process running is the leader, behavior matches the
straight-forward "no daemon" case from earlier designs:
1. Load `.scry/config.yaml`
2. Verify `.scry/vectors.db` exists; auto-reconcile (§7.2) or
   hard-error on embedding-model mismatch
3. Acquire leader lock
4. Open vector store read-write
5. Start IPC listener
6. Serve MCP tools over stdio
7. Lazy-load embedder model on first `search` call

When the leader exits, its lock is released. The next `scry mcp` to
start is a fresh leader.

---

## 11. Tech stack

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Tree-sitter Python bindings polyglot; fastembed Python-first; brain-vault familiarity |
| Distribution | `pip install scry-cli` (`scry` binary) | Single `uv tool install` works on Win/Mac/Linux |
| MCP framework | FastMCP | Battle-tested |
| Markdown parser | `markdown-it-py` + custom frontmatter handler | CommonMark-compliant, AST gives heading boundaries cleanly |
| Code parsing | `tree-sitter-language-pack` | One wheel, ~30 langs cross-platform |
| Code resolution | Existing LSP servers as subprocesses (pyright-langserver / pylsp / basedpyright-langserver, typescript-language-server, zls, gopls, rust-analyzer) | Production-quality semantic resolution per language; transitive drift via `callHierarchy` |
| Embeddings (default) | `fastembed` with `BAAI/bge-small-en-v1.5` | Local ONNX, ~30 MB, no API key |
| Embedding providers (opt-in) | LiteLLM-style abstraction (OpenAI/Voyage/Cohere/local) | Same pattern brain-vault uses |
| Vector store | `sqlite-vec` extension | Single file, no server, mmap-fast at our scale |
| Keyword search | SQLite FTS5 | In stdlib; combined with vectors via RRF |
| IPC (leader-follower) | Unix domain sockets (Linux/macOS); Windows named pipes; JSON-over-stream | No third-party dep; cross-platform |
| Hashing | SHA-256 over canonicalized content (§5.4); SimHash for fuzzy match | Deterministic, fast |
| Tests | `pytest` | Standard |

**Total LOC target:** ~6,500 lines core for initial release (raised
from earlier ~5,000 estimate to account for IPC infrastructure and
overlay layer).

---

## 12. Implementation Order

This is the sequence we'll build in — not a release schedule. The
first published release of scry contains the full surface defined in
this document. The order below sequences risk: each wave builds on
the previous and unblocks the next.

### Wave 1 — Anchor foundation
- Markdown + tree-sitter extraction (section, code_in_doc, code stub-only)
- Two-tier embedding model with sub-chunking
- Layer 1 + Layer 2 anchor IDs with HTML-comment escape hatch
- `.scry/config.yaml` loader with classify list validation
- §15 edge-case extraction rules + test fixtures

### Wave 2 — Storage, retrieval, and section-level drift
- `vectors.db` with sqlite-vec + FTS5 + WAL mode + `index_metadata` row
- Two-tier consistency invariants (parent_content_hash on chunks,
  transactional reindex with metadata in same transaction,
  read-side hash-equality filter, advisory write lock)
- `links.jsonl` baseline + `.scry/overlays/<branch>.jsonl` per-branch
  overlay; event-record reader/writer with `.gitattributes union`
  driver; replay rules from §3.5.2
- Hybrid retrieval algorithm (BM25 + vector + RRF with best-chunk-per-parent
  promotion)
- MCP server skeleton: leader election, IPC listener, follower mode
- Tools (Wave 2): `search`, `get_anchor`, `get_links`,
  `repo_summary`, `reindex`, `status`, `propose_link`, `accept_link`,
  `commit_links`, `scry link`
- **Section-level drift** (`spec-changed`, `broken-source`,
  `broken-target`, `merge-conflict`): no LSP needed; ships in Wave 2
- Polling-based git-context detection at every tool call
- `index_state` field on every MCP response (Wave 2 hardcodes
  `"fresh"`; Wave 3 activates the other values)
- Lazy embedder model loading
- `scry doctor`, `scry validate`

### Wave 3 — LSP-resolved code anchors
- LSP subprocess infrastructure (JSON-RPC over stdio, lifecycle, caching)
- LSP binary allowlist (§6.2); `--allow-untrusted-lsp-config` flag
- Per-language LSP adapters (pyright-langserver, typescript-language-server, zls)
- `callHierarchy/outgoingCalls` for transitive closure hashing
- `transitive_hash_status` enum on every code anchor
- LSP capability probing in `scry doctor`
- Hard error on missing LSP for an indexed language

### Wave 4 — Code-level drift + auto-reconcile
- `code-changed` drift status (uses Wave 3 transitive closure)
- `semantic-drift` for `mirrors` links (embedding-distance check)
- `find_drift` MCP tool
- `scry check` CLI with `--ci` exit codes; coverage_score; raw counts;
  `--require-fresh-embedder` flag
- Auto-reconcile-on-startup activates the non-`"fresh"` `index_state`
  values (`"stale-reconciling"`, `"stale-no-write-lock"`)
- Inline rebase on rename via embedding similarity + SimHash
  confirmation; rebase records written to overlay
- `scry index --reembed` migration path
- `prior_content_hash` field on rebased upserts so drift reflects
  rename-with-edit signal

### Wave 5 — AI-assisted curation and reconciliation
- `scry suggest-links` AI batch suggestions (opt-in)
- `scry reconcile <link_id>` AI patch proposals using stored
  `commit_sha` to reconstruct baselines via `git show`
- LiteLLM-style provider abstraction for the LLM side

### Wave 6 — Polyglot expansion and ergonomics
- LSP adapters for additional languages (gopls, rust-analyzer,
  pylsp/basedpyright as secondary Python options, etc.)
- `code_anchors.transitive_resolution: full` opt-in mode
  (`textDocument/references` + symbol lookup for inheritance and
  imported constants)
- `scry watch` hot-reindex with leader-follower coordination
- Reverse-link queries leveraging the LSP-built call graph
  (`get_callers`, `get_subclasses`)
- Per-section frontmatter directives (if user feedback requests)

---

## 13. Open implementation questions (to revisit during build)

1. **LSP performance budget** — `callHierarchy` queries can be slow on
   large repos. Cache aggressively; budget per index pass; consider
   keeping a single LSP alive for the whole indexing run vs. spawn
   per call (huge perf difference).
2. **LSP multi-root workspaces** — LSPs need an init root. Monorepos
   spanning multiple package roots may need per-package LSP instances.
3. **LSP version pinning** — pyright-langserver, typescript-language-server, etc.
   APIs evolve. Test matrix and document compatibility. Store LSP
   version in `index_metadata`?
4. **Code-in-doc detection thresholds** — when is a fenced code block
   substantial enough to be its own anchor? Default: ≥ 5 lines OR an
   explicit language tag with ≥ 1 named declaration. Performance impact
   when running tree-sitter on every code block during indexing.
5. **IPC perf characteristics** — JSON-over-stream Unix socket / Windows
   pipe round-trip latency for write forwarding; ensure follower
   write operations don't degrade noticeably vs leader-direct.
6. **Stale leader lock detection** — what's the right policy for
   "the holder's PID is dead, force lock acquisition"? Simple
   PID-alive check should work but cross-platform (Windows process
   model differs).
7. **`scry init --register-global`** — write strategy for
   `~/.claude.json` / `~/.cursor/mcp.json` (existing keys, JSON merge
   semantics, backup). Default behavior is print-and-let-user-paste.
8. **Anonymous code-block hash collisions** — 8-char prefix has 2^32
   space; collisions become realistic at ~65k anchors (birthday).
   Detect at index time and extend prefix on collision, or just use
   16-char prefix from the start.
9. **JSONL replay performance at scale** — for repos with thousands
   of links and many overlay records, replay cost grows. May need
   `scry vacuum` to compact baseline + drop superseded chains.
10. **FTS5 tokenizer for non-ASCII content** — default `unicode61`
    handles diacritics but doesn't segment CJK. Document the
    limitation; consider language-aware tokenizer in future.

---

## 14. What scry deliberately is not (revisited)

| Feature | Why not |
|---|---|
| Web UI / docs portal | Agents are the consumer; humans use the CLI for hygiene |
| Chat / LLM responses | The agent harness owns that |
| Multi-project registry | One repo, one `.scry/` directory |
| Branch-aware indexing as a separate model | Per-checkout `.scry/` already isolates; per-branch overlays + auto-reconcile handle switches |
| Auto-fixing drift | Always human-in-the-loop; agents propose, humans accept |
| Spec authoring tools | Devs write markdown however they want; scry just indexes |
| Custom LSP work per language | Re-use existing language servers; we are not in the language-server business |
| Daemon mode | Not needed — the leader-follower model gives us coordinated multi-process support without long-lived processes |

---

## 15. Edge cases & extraction rules

Behavior rules for inputs the design must handle correctly. Each
rule is paired with a test fixture committed alongside the
implementation.

### 15.1 Heading extraction

| Input | Behavior |
|---|---|
| **ATX headings** (`## Heading`) | Standard. Become anchors at H1–H4 (`sections.max_heading_depth`); deeper headings split sub-chunks of nearest ancestor anchor. |
| **Setext headings** (`Heading\n====` / `Heading\n----`) | Treat as H1 (`====`) or H2 (`----`); become anchors |
| **File with no headings** | Whole file becomes one anchor with ID `<path>` (no `::heading` suffix); sub-chunked normally |
| **Single H1, no further structure** | Single H1 anchor; sub-chunks if exceeds `max_tokens` |
| **H1 used as section divider every 100 lines** | Each H1 is its own anchor; matches user's structural intent |
| **H1 inside a fenced code block** (e.g., a tutorial about Markdown) | Ignored — fence content is opaque; only "real" headings outside code blocks become anchors |
| **Heading depth deeper than `max_heading_depth`** | Treated as content of its nearest ancestor anchor; visible in the section text but not its own anchor |
| **Empty heading or pure-punctuation heading** (`## ` or `## ?!`) | Slug fallback: `section-<short-content-hash>`; same fallback applies to non-ASCII / emoji headings whose slugify result is empty |

### 15.2 Code block extraction

| Input | Behavior |
|---|---|
| **Block with language tag containing a named declaration** | Becomes a `code_in_doc` anchor with `<section-id>::<declaration-name>` suffix |
| **Block with language tag, no declaration** | Skipped as `code_in_doc`; remains part of the parent section's content |
| **Block without a language tag** | Skipped as `code_in_doc`; remains part of the parent section's content |
| **Malformed / unclosed code fence** | markdown-it-py default: code block extends to EOF; we honor that |
| **Nested fenced blocks** (` ```` `, `~~~`) | Parse per markdown-it-py; treat outer fence as the boundary |

### 15.3 ID derivation

| Input | Behavior |
|---|---|
| **HTML-comment ID present** (`<!-- scry-id: foo -->`) | Pinned slug overrides path-derived slug |
| **Sibling-heading slug collision** (two H3 "Examples" siblings) | First gets bare slug, subsequent get `-2`, `-3`, ... (deterministic by file order) |
| **Same code declaration name in same section** (two `Config` blocks) | First gets bare name, subsequent get `@2`, `@3`, ... |
| **Anonymous code block** (block without extractable declaration) | `::block-<short-content-hash>` (8-char prefix; extended if collision detected at index time) |
| **Symbol overload** (TypeScript `function f(x:string); function f(x:number);`) | `:f@<sig-hash>` suffix on each |
| **Duplicate `scry-id` HTML comments in same document** | Index-time validation error |
| **Duplicate frontmatter `id` across different files** | Index-time validation error (surfaced via `scry validate`) |

### 15.4 File-level filters

| Input | Behavior |
|---|---|
| **Empty markdown file** | Skipped (no extractable content); not surfaced in retrieval |
| **Frontmatter-only file** (frontmatter present, body empty) | Skipped |
| **File exceeds `index.max_file_size_bytes`** (default 5 MB) | Skipped with warning logged; no anchors created |
| **Binary file with `.md` extension** (UTF-8 decode fails) | Skipped with warning |
| **UTF-16 LE/BE encoded `.md` file** | Detected via BOM (`\xFE\xFF` BE / `\xFF\xFE` LE); skipped with a "transcode to UTF-8 to index" warning |
| **File matched by `exclude` glob** | Skipped silently. Frontmatter `skip: false` cannot override hard safety excludes (e.g., `secrets/**`). |
| **File matched by frontmatter `skip: true`** | Skipped, even if matched by `include:` |
| **File matched by no `classify` rule** | Excluded from indexing (must classify to participate); warning if matched by `include:` |
| **Symlink to file inside repo** | Followed; if the inode resolves to a file already indexed under another path, deduplicated; the canonical (non-symlink) path wins |
| **Symlink to file outside repo root** | Skipped with warning (security boundary) |
| **Multiple `index_metadata` rows in vectors.db** (corruption case) | Hard error at startup; suggest `scry index --force` |

### 15.5 Reference-style markdown links

| Input | Behavior |
|---|---|
| **Inline link** (`[text](path/file.md#heading)`) | Resolved at extraction; if target is a known anchor, contributes a `references` link |
| **Reference-style link** (`[text][ref]` + `[ref]: path/file.md`) | Resolve `[ref]: ...` definitions per CommonMark, then process as inline |
| **Link to anchor that doesn't exist at index time** | Recorded as a candidate `references` link with `drift_status: broken-target` |
| **Link to absolute URL** (`http://`, `https://`, `mailto:`) | Skipped; no `references` link is created (would flood `find_drift` output otherwise) |

### 15.6 Link graph integrity

| Input | Behavior |
|---|---|
| **Circular `derives-from` chain** (A→B→C→A) | Detected at link-write time; `scry link`, `accept_link`, `propose_link` all validate that the new edge does not create a cycle in the `derives-from` subgraph; error surfaced as validation failure |
| **`upsert` with a `link_id` that already exists in the file but no `supersedes` field** | Validation error at write time |
| **`upsert` after a `delete` for the same `link_id` in the same file** | Validation error at write time |
| **Two competing latest upserts for one `link_id` post-merge** | `merge-conflict` drift status; surfaced via `scry status`; user resolves by writing a new `upsert` superseding both |

---

*Last updated: 2026-05-08*
*Status: design v3 — incorporates all 8 v2 swarm-review decisions*
