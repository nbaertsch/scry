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
surfaced — including transitive code drift via LSP-resolved call
hierarchies — so engineers and agents see when a spec and its
implementing code have diverged.

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
| `section` | `<path>::<heading-path-slug>` (e.g. `docs/POLICY_ENGINE.md::policy-engine::rule-structure`) |
| `code_in_doc` | `<section-id>::<declaration-name>` for blocks containing a named declaration; `<section-id>::<declaration-name>@N` for same-name collisions in the same section (deterministic by file order); `<section-id>::block-<short-content-hash>` for anonymous blocks (no extractable declaration) |
| `code` | `<file-path>:<qualified-symbol-path>` (e.g. `python/policy.py:OuterClass.InnerClass.method`); `:f@<sig-hash>` suffix for overloads |

Sibling-heading slug collisions get a deterministic positional
suffix (`::examples` for the first, `::examples-2` for the second);
this is unstable across reorderings but acceptable for the rare case
of intentional duplicate sibling names.

**Layer 2 — Secondary content fingerprints.** Stored on every anchor
record in `vectors.db`. Never appear in user-facing IDs. Used at
re-index time for **inline rebase** (see §3.3).

| Anchor type | Fingerprint |
|---|---|
| `section` | `(normalized_heading_path, content_hash)` where `content_hash` is SHA-256 over canonicalized body (see §5.4) |
| `code` | `(qualified_scope, signature_hash, ast_subtree_fingerprint)` |
| `code_in_doc` | `(language, declaration_name_or_signature, code_hash)` |

**Escape hatch.** Authors can pin a slug independently of structural
location with an HTML comment immediately under a heading or code
block:

```markdown
## Rule Structure
<!-- scry-id: rule-structure -->
```

When present, the pinned slug overrides the path-derived slug. Heading
or code rewrites then never invalidate the link.

Duplicate `scry-id` values within a single document are an index-time
validation error.

### 3.3 Inline rebase on re-index (no migration log)

When `scry index` runs, it walks the repo and builds the new anchor
set with new primary IDs. For each anchor whose primary ID is missing
from the prior index but whose Layer 2 fingerprint matches a missing
old anchor, scry **rebases links forward in place**:

- Existing links pointing at the old ID get appended `upsert` records
  (see §3.5) with the new ID and updated `from/to_content_hash`.
- The original `link_id` is preserved.
- The link's `drift_status` reflects the actual content change
  (likely `code-changed` or `spec-changed`; rarely `fresh`).

No `aliases.jsonl` file or migration log is written — the JSONL
event-record stream IS the migration history.

**Fuzzy fingerprint match algorithm:**

1. **Embedding cosine similarity** — find candidate matches in the
   missing-from-new set using the same embedding model used for
   retrieval. Threshold default `0.85`, configurable.
2. **SimHash / Jaccard confirmation** — for each candidate, compute
   SimHash over the canonicalized content and confirm Jaccard
   similarity ≥ `0.7` (configurable) before committing the rebase.
3. **Same-file constraint by default** — rebase only considers
   candidates within the same file. Cross-file rebase (catches `git
   mv` + heading rename combos) is opt-in via
   `index.cross_file_rebase: true`.
4. **No match → broken-source / broken-target** — if no candidate
   passes, the link surfaces as broken on next drift check; user
   re-links manually.

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
  ─ NOT stable IDs (regenerated on re-chunk; not used externally)
  ─ embedding of just that sub-chunk text
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

A typed, directed edge between two anchors. Stored in
`.scry/links.jsonl` as **append-only event records**:

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
  "commit_sha": "abc123def456...",       // git HEAD at upsert time
  "worktree_dirty": false,                // was working tree dirty at upsert?
  "supersedes": "lnk_old_xyz",            // optional, set on rebase
  "evidence": "Optional pull-quote",
  "ts": "2026-05-06T12:34:56Z"
}

// Delete
{
  "op": "delete",
  "link_id": "lnk_abc",
  "ts": "2026-05-06T13:00:00Z",
  "reason": "manual"
}
```

**Reader semantics.** Active link table = replay of the JSONL with
last-write-wins by `link_id`; tombstones skip on load.

**Writer semantics.** Always append. Never mutate existing lines.
Inline rebase (§3.3), drift-aware hash refresh, and link curation all
write new `upsert` records that supersede prior ones.

**Merge.** Ship a `.gitattributes` line installing the `union` merge
driver for `.scry/links.jsonl`. Concurrent appends from different
branches union cleanly; conflicts only arise on the same line, which
is impossible by construction.

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
inverse name applied — purely a query-time view. Storage stays
canonical. This eliminates the double-counting drift bug that would
arise if both `implements` and `specifies` existed for the same
logical relationship.

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
  2. Run two retrievals in parallel over ALL embeddings + FTS index:
       - Vector ANN over (parent overview + sub-chunks + code anchors)
       - BM25 (FTS5) over (parent overview + sub-chunks + code anchors)
  3. RRF-fuse the two ranked lists into one rank list:
       fused_score(anchor_or_chunk) = Σ_{list ∈ {vec, bm25}} 1 / (k + rank_in_list)
       (k from retrieval.rrf_k, default 60)
  4. Group sub-chunk ranks by parent anchor; aggregate via RRF:
       parent_score = Σ_{chunk ∈ parent_chunks} 1 / (k + chunk_rank_in_fused)
       (parent's overview embedding participates as one "chunk" in this sum)
  5. Sort parents by aggregated RRF score; take top_k
  6. For each result, populate the anchor packet (§4.2):
       - Pull the best-matching sub-chunk excerpt for the evidence field
       - Pull all 1-hop neighbors from the link graph (filtered by config caps)
       - Compute drift_status for each link
```

**No graph-traversal influence on ranking.** Graph context comes via
the `links` field on each result (§4.2). This dissolves the `α`
tuning problem entirely (no parameter exists), keeps the
score-fusion primitive consistent (RRF everywhere), and gives agents
the graph context they need without a tuning knob.

If real-world usage shows graph-influenced ranking is genuinely
needed, the architecture leaves room to add it later as an additional
RRF rank list.

### 4.2 The Anchor Packet

Every search result is an **anchor packet**:

```jsonc
{
  "anchor": {
    "id": "docs/POLICY_ENGINE.md::policy-engine::rule-structure",
    "type": "section",
    "path": "docs/POLICY_ENGINE.md",
    "heading_path": ["Policy Engine", "Rule Structure"],
    "content": "...full section markdown...",
    "content_hash": "sha256:..."
  },
  "score": 0.83,
  "evidence_excerpt": "The matched sub-chunk text",
  "links": [
    {"to": "python/hailstone/policy/engine.py:PolicyRule",
     "to_type": "code", "type": "implements", "drift_status": "fresh"},
    {"to": "docs/EXECUTION_PIPELINE.md::policy-middleware",
     "to_type": "section", "type": "references", "drift_status": "n/a"}
  ],
  "index_state": "fresh"  // see §7
}
```

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
| `code-changed` | Code endpoint content hash changed (own AST + transitive closure via LSP — see §5.3) |
| `both-changed` | Both endpoints' content hashes changed |
| `broken-source` | Source anchor no longer exists (and no rebase candidate matched) |
| `broken-target` | Target anchor no longer exists (and no rebase candidate matched) |
| `semantic-drift` | For `mirrors` links only — both endpoints' hashes refreshed but cosine distance between their embeddings exceeds `drift.semantic_drift_threshold` (default 0.25). Caught even when text-hash drift was reconciled. |

**Status precedence** (highest first):
`broken-source` / `broken-target` > `both-changed` > `spec-changed` /
`code-changed` > `semantic-drift` > `fresh`

### 5.2 Drift score and coverage score

Two scores computed by `scry check`:

**Drift score** (normalized, 0–100):
```
drift_score = 100 × (1 - Σ(weight_c × count_c) / max(1, total_links))
```

Default weights (configurable in `drift.scoring`):
| status | weight |
|---|---|
| `broken-*` | 1.0 |
| `both-changed` | 0.5 |
| `spec-changed` | 0.3 |
| `code-changed` | 0.3 |
| `semantic-drift` | 0.2 |

A repo with 100% broken links → 0. No findings → 100. 10%
spec-changed → 97. **Independent of repo size.**

**Coverage score** (0–100):
```
coverage_score = 100 × (linked_code_anchors / total_code_anchors)
```

Surfaces the "no links" case that `drift_score` alone hides. CI can
gate on either or both independently.

**Always emitted alongside the scores: raw counts** so CI policy can
gate on counts directly:

```jsonc
{
  "drift_score": 92.3,
  "coverage_score": 67.5,
  "counts": {
    "broken_source": 1, "broken_target": 1,
    "both_changed": 0, "spec_changed": 8,
    "code_changed": 1, "semantic_drift": 0,
    "fresh": 145, "total": 156
  },
  "by_anchor_type": {...}
}
```

### 5.3 Code transitive drift via LSP

`code` anchor `content_hash` includes both the anchor's own
canonicalized AST text **and** sorted hashes of all
transitively-reachable definitions in the same repo, computed using
the language's LSP server (see §6).

This catches behavioral drift that pure-AST hashing misses:
- A helper function `validate_field` in the same or imported file changed
- A base class method scry's anchor inherits and depends on changed
- An imported constant the anchor uses changed

Closure walks `callHierarchy/outgoingCalls` until the module/repo
boundary; external library code (in `node_modules`, `site-packages`,
`vendor/`, etc.) is excluded.

### 5.4 Content-hash canonicalization

`content_hash` is SHA-256 over canonicalized content. Canonicalization
steps applied before hashing:

1. Strip UTF-8 BOM if present
2. Normalize all line endings to LF
3. Trim trailing whitespace per line
4. Collapse trailing newlines at EOF to a single `\n`

What is **deliberately not** done:
- No paragraph reflow (would hide real edits)
- No collapse of internal whitespace runs (significant in code blocks)
- No Unicode NFC normalization (too disruptive for non-ASCII content)

Same canonicalization applies to `section`, `code_in_doc`, and `code`
anchor content hashes. For `code`, canonicalization runs over the
canonicalized AST subtree text plus the transitive-closure suffix
(§5.3).

### 5.5 Reconciliation (LLM-assisted, opt-in)

`scry reconcile <link_id>` runs a structured AI loop:

1. **Reconstruct the baseline** from stored `commit_sha`. If reachable
   in git history, diff `git show <commit_sha>:path` vs current file
   for both endpoints. If not reachable (rebased away), fall back to
   "cannot reconstruct original; showing current state vs
   `content_hash` mismatch only." If `worktree_dirty` was true at
   upsert, caveat the diff with that fact.
2. **Use an LLM** to propose: "spec is correct, code needs change X" OR
   "code is correct, spec needs change Y" OR "both need an update"
3. **Output the proposed patch** as a unified diff for human review.
4. **On accept**, write a new `upsert` record with refreshed hashes
   and the new `commit_sha`.

The deterministic `check` path never calls an LLM. AI is opt-in,
contained, and never blocks correctness signals.

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
  lsp:
    python:   { command: pyright-langserver, args: [--stdio] }
    typescript: { command: typescript-language-server, args: [--stdio] }
    zig:      { command: zls, args: [] }
    # Per-language overrides: set `command` to override binary discovery

# Embedding provider
embeddings:
  provider: local              # local | openai | voyage | custom (OpenAI-compatible)
  model: BAAI/bge-small-en-v1.5
  dimensions: 384

# Retrieval
retrieval:
  rrf_k: 60                    # RRF k-parameter (BM25+vector fusion AND sub-chunk-into-parent)
  bm25:
    enabled: true
  links_per_result:
    outgoing: 5
    incoming: 5

# Drift detection
drift:
  semantic_drift_threshold: 0.25
  scoring:
    broken: 1.0
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
  max_file_size_bytes: 5242880  # 5 MB; files larger than this skipped with warning
```

### 6.1 Per-file frontmatter overrides

Frontmatter is **optional** — repos with well-organized docs rely on
globs alone. When present, frontmatter overrides config:

```markdown
---
scry:
  type: spec                 # overrides classify result
  id: AUTH-LOGIN              # overrides path-derived ID; uniqueness validated at index time
  exclude: false              # opt out per-file
---
```

---

## 7. Persistence layout and provenance

```
<repo>/
├── .scry/
│   ├── config.yaml          ← user-authored; required
│   ├── links.jsonl          ← append-only event log; committed to git
│   ├── vectors.db           ← embeddings + content hashes; gitignored
│   ├── cache/               ← extracted anchors, file hashes; gitignored
│   └── stats.json           ← drift score history; gitignored
└── (repo files)
```

Vectors and cache live in the repo (not in a global data dir like
scrybe). This is deliberate:
- Switching branches → switching index → no cross-branch leakage
- Multiple agent sessions in different repos → no daemon coordination
- Easy to gitignore the cache; easy to ship the index in CI artifacts

### 7.1 Index provenance metadata

`vectors.db` stores a single `index_metadata` row tracking what
produced the current state:

```jsonc
{
  "indexed_git_head": "abc123def456...",
  "indexed_git_tree_hash": "tree_hash...",
  "indexed_file_manifest": {"path/file.md": "sha256:..."},
  "config_hash": "sha256:...",
  "embedding_provider": "local",
  "embedding_model": "BAAI/bge-small-en-v1.5",
  "embedding_dimensions": 384,
  "tokenizer_version": "..."
}
```

### 7.2 Auto-reconcile on startup (branch switches, file edits)

On every `scry mcp` startup and every CLI command that touches
`vectors.db`:

1. **Compare stored `index_metadata` vs current repo state.**
2. **Mismatch in `embedding_*` (provider, model, dimensions, tokenizer):**
   **HARD ERROR.** Refuse to serve. Tell user to run `scry index --force`.
   This is deliberate and overrides the auto-reconcile pattern below —
   model upgrades are too expensive and consequential to do silently.
3. **Mismatch in `indexed_git_head` / `indexed_file_manifest`:**
   Auto-incremental-reconcile transparently. Reindex changed/new files,
   delete anchors for removed files. Suppress with `--no-auto-reconcile`.
4. **MCP responses include `"index_state"`:** one of `"fresh"`,
   `"stale-reconciling"`, `"stale-warned"`. Agents can react.

### 7.3 Two-tier embedding consistency invariants

To prevent the "stale sub-chunk attached to fresh parent hash" race:

1. Every chunk row stores its parent's `content_hash`.
2. Re-index of a parent = single SQLite transaction:
   delete ALL existing chunks for that parent, insert ALL new chunks
   tagged with the new parent hash, update parent record.
3. Search returns only chunks whose stored `parent_content_hash`
   equals the current parent's `content_hash`. Stale orphans are
   invisible.
4. SQLite opened in WAL mode explicitly (single writer + concurrent
   readers).
5. `scry watch`, when implemented, writes a PID/lock file so `scry
   mcp` knows to re-read after change notifications.

---

## 8. MCP tool surface

| Tool | Purpose |
|---|---|
| `search(query, types?, top_k?)` | Hybrid retrieval; returns ranked anchor packets |
| `get_anchor(id)` | Full content of an anchor by ID |
| `get_links(anchor_id, link_types?, direction?)` | Bidirectional link enumeration; inverse names rendered for `direction=incoming` |
| `find_drift(scope?, status_filter?)` | List anchors/links with drift status > `fresh` |
| `propose_link(from_id, to_id, link_type, evidence?)` | Stages a link for human review (writes to `.scry/links.proposed.jsonl`) |
| `accept_link(proposed_id)` | Promotes proposed → committed link (appends `upsert` record to `links.jsonl`) |
| `repo_summary()` | One-shot orientation: file tree, classified docs, top symbols, drift + coverage scores |
| `reindex(scope?)` | Force re-extraction (default is incremental on file change) |

CLI surface mirrors the MCP tools 1:1.

The MCP server lazy-loads the embedder model on first call to `search`
or any tool that requires query embeddings. `get_anchor`, `get_links`,
`find_drift`, `repo_summary` do not pay the model-load tax.

---

## 9. CLI surface

| Command | What it does |
|---|---|
| `scry init` | Wizard: choose embedding provider, write `.scry/config.yaml`, install `.gitattributes` union driver for `links.jsonl`. With `--register-global` flag, also registers MCP entry in `~/.claude.json` / `~/.cursor/mcp.json` (default: prints the JSON snippet for manual paste). |
| `scry index [--force]` | Build/refresh the vector store and AST cache |
| `scry watch` | Sit on a file watcher; reindex on change; coordinates with running `scry mcp` via lock file |
| `scry check [--format json\|md] [--ci]` | Drift + coverage scores; `--ci` exits non-zero on configurable thresholds |
| `scry search "<query>" [--top-k N] [--type section\|code]` | Same as MCP `search`, prints to stdout |
| `scry link <from> <to> --type <link_type> [--evidence "..."]` | Author a link from the CLI |
| `scry suggest-links [--scope <path>] [--accept-all]` | AI-augmented batch link suggestions (opt-in; requires LLM provider) |
| `scry reconcile <link_id>` | AI-assisted patch proposal for drifted links (opt-in) |
| `scry mcp` | Run the stdio MCP server (no daemon required) |

---

## 10. Process model

**Single, short-lived process per agent session.** No daemon by default.
Per-repo `.scry/` cache is durable; the index doesn't need a long-lived
process to stay warm.

When `scry mcp` starts:
1. Load `.scry/config.yaml`
2. Verify `.scry/vectors.db` exists; auto-incremental-reconcile (§7.2)
   or hard-error on embedding-model mismatch
3. Open vector store (read-only by default; read-write if `scry watch`
   is also running, coordinated via lock file)
4. Serve MCP tools over stdio
5. Lazy-load embedder model on first `search` call

When the agent disconnects, the process exits. No state to persist
beyond what the disk already holds.

For workflows where MCP servers are recycled per request and per-call
cold-start latency matters, an opt-in daemon mode is on the
implementation order (§12 Wave 6).

---

## 11. Tech stack

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Tree-sitter Python bindings polyglot; fastembed Python-first; brain-vault familiarity |
| Distribution | `pip install scry-cli` (`scry` binary) | Single `uv tool install` works on Win/Mac/Linux |
| MCP framework | FastMCP | Battle-tested |
| Markdown parser | `markdown-it-py` + custom frontmatter handler | CommonMark-compliant, AST gives heading boundaries cleanly |
| Code parsing | `tree-sitter-language-pack` | One wheel, ~30 langs cross-platform |
| Code resolution | Existing LSP servers as subprocesses (pyright, typescript-language-server, zls, gopls, rust-analyzer) | Production-quality semantic resolution per language; transitive drift via `callHierarchy` |
| Embeddings (default) | `fastembed` with `BAAI/bge-small-en-v1.5` | Local ONNX, ~30 MB, no API key |
| Embedding providers (opt-in) | LiteLLM-style abstraction (OpenAI/Voyage/Cohere/local) | Same pattern brain-vault uses |
| Vector store | `sqlite-vec` extension | Single file, no server, mmap-fast at our scale |
| Keyword search | SQLite FTS5 | In stdlib; combined with vectors via RRF |
| Hashing | SHA-256 over canonicalized content (§5.4); SimHash for fuzzy match | Deterministic, fast |
| Tests | `pytest` | Standard |

**Total LOC target:** ~5,000 lines core for initial release (larger
than the original ~3,500 estimate due to the LSP subprocess
infrastructure).

---

## 12. Implementation Order

This is the sequence we'll build in — not a release schedule. The
first published release of scry contains the full surface defined in
this document. The order below sequences risk: each wave builds on
the previous and unblocks the next.

### Wave 1 — Anchor foundation
- Markdown + tree-sitter extraction (section, code_in_doc, code)
- Two-tier embedding model with sub-chunking
- Layer 1 + Layer 2 anchor IDs with HTML-comment escape hatch
- `.scry/config.yaml` loader with classify list validation
- §15 edge-case extraction rules + test fixtures

### Wave 2 — Storage and retrieval
- `vectors.db` with sqlite-vec + FTS5 + WAL mode + `index_metadata` row
- Two-tier consistency invariants (parent_content_hash on chunks,
  transactional reindex, read-side hash-equality filter)
- `links.jsonl` event-record reader/writer with `.gitattributes union` driver
- Hybrid retrieval algorithm (BM25 + vector + RRF + sub-chunk RRF aggregation)
- MCP server skeleton + `search`, `get_anchor`, `get_links`,
  `repo_summary`, `reindex` tools
- Lazy embedder model loading

### Wave 3 — LSP-resolved code anchors
- LSP subprocess infrastructure (JSON-RPC over stdio, lifecycle, caching)
- Per-language LSP adapters (pyright, typescript-language-server, zls)
- `callHierarchy/outgoingCalls` for transitive closure hashing
- Hard error on missing LSP for an indexed language

### Wave 4 — Drift detection
- Per-link `drift_status` (fresh, spec-changed, code-changed,
  both-changed, broken-source, broken-target, semantic-drift)
- Drift score + coverage score formulas
- `find_drift` MCP tool, `scry check` CLI with `--ci` exit codes
- Auto-reconcile-on-startup with `index_state` field in MCP responses
- Inline rebase on rename (re-index detection of section/symbol moves
  via embedding similarity + SimHash confirmation)

### Wave 5 — Curation and reconciliation
- `propose_link` / `accept_link` MCP tools with
  `.scry/links.proposed.jsonl` staging
- `scry link` CLI for direct human-authored links
- `scry suggest-links` AI batch suggestions (opt-in)
- `scry reconcile <link_id>` AI patch proposals using stored
  `commit_sha` to reconstruct baselines via `git show`

### Wave 6 — Polyglot + ergonomics
- LSP adapters for additional languages (gopls, rust-analyzer, etc.)
- `scry watch` hot-reindex with `scry mcp` lock-file coordination
- Optional in-process daemon mode for very large repos / per-call
  MCP recycling workflows
- Reverse-link queries leveraging the LSP-built call graph
  (`get_callers`, `get_subclasses`)

---

## 13. Open implementation questions (to revisit during build)

1. **LSP performance budget** — `callHierarchy` queries can be slow on
   large repos. Cache aggressively; budget per index pass; consider
   keeping a single LSP alive for the whole indexing run vs. spawn
   per call (huge perf difference).
2. **LSP multi-root workspaces** — LSPs need an init root. Monorepos
   spanning multiple package roots may need per-package LSP instances.
3. **External-library boundary for transitive closure** — closure walks
   outgoing calls; should stop at repo root (default), or also walk
   into specific dependency directories (opt-in)?
4. **LSP version pinning** — pyright/ts-server/etc. APIs evolve. Test
   matrix and document compatibility.
5. **Code-in-doc detection thresholds** — when is a fenced code block
   substantial enough to be its own anchor? Default: ≥ 5 lines OR an
   explicit language tag with ≥ 1 named declaration. Performance impact
   when running tree-sitter on every code block during indexing.
6. **Frontmatter `id` uniqueness** — validate at index time across all
   files. What does the error look like?
7. **`scry init --register-global`** — write strategy for
   `~/.claude.json` / `~/.cursor/mcp.json` (existing keys, JSON merge
   semantics, backup). Default behavior is print-and-let-user-paste.

---

## 14. What scry deliberately is not (revisited)

| Feature | Why not |
|---|---|
| Web UI / docs portal | Agents are the consumer; humans use the CLI for hygiene |
| Chat / LLM responses | The agent harness owns that |
| Multi-project registry | One repo, one `.scry/` directory, one process per session |
| Branch-aware indexing as a separate model | Per-checkout `.scry/` already isolates; auto-reconcile handles switches |
| Auto-fixing drift | Always human-in-the-loop; agents propose, humans accept |
| Spec authoring tools | Devs write markdown however they want; scry just indexes |
| Custom LSP work per language | Re-use existing language servers; we are not in the language-server business |

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

### 15.2 Code block extraction

| Input | Behavior |
|---|---|
| **Block with language tag containing a named declaration** | Becomes a `code_in_doc` anchor with `<section-id>::<declaration-name>` suffix |
| **Block with language tag, no declaration** | Skipped as `code_in_doc`; remains part of the parent section's content |
| **Block without a language tag** | Skipped as `code_in_doc`; remains part of the parent section's content (conservative: don't guess language) |
| **Malformed / unclosed code fence** | markdown-it-py default: code block extends to EOF; we honor that |
| **Nested fenced blocks** (` ```` `, `~~~`) | Parse per markdown-it-py; treat outer fence as the boundary |

### 15.3 ID derivation

| Input | Behavior |
|---|---|
| **HTML-comment ID present** (`<!-- scry-id: foo -->`) | Pinned slug overrides path-derived slug |
| **Sibling-heading slug collision** (two H3 "Examples" siblings) | First gets bare slug, subsequent get `-2`, `-3`, ... (deterministic by file order); unstable across reorderings, acceptable for rare case |
| **Same code declaration name in same section** (two `Config` blocks) | First gets bare name, subsequent get `@2`, `@3`, ... |
| **Anonymous code block** (block without extractable declaration) | `::block-<short-content-hash>` (8-char prefix) |
| **Symbol overload** (TypeScript `function f(x:string); function f(x:number);`) | `:f@<sig-hash>` suffix on each |
| **Duplicate `scry-id` HTML comments in same document** | Index-time validation error |
| **Duplicate frontmatter `id` across different files** | Index-time validation error |

### 15.4 File-level filters

| Input | Behavior |
|---|---|
| **Empty markdown file** | Skipped (no extractable content); not surfaced in retrieval |
| **Frontmatter-only file** (frontmatter present, body empty) | Skipped |
| **File exceeds `index.max_file_size_bytes`** (default 5 MB) | Skipped with warning logged; no anchors created |
| **Binary file with `.md` extension** (UTF-8 decode fails) | Skipped with warning |
| **File matched by `exclude` glob** | Skipped silently |
| **File matched by no `classify` rule** | Excluded from indexing (must classify to participate); warning if include glob matched |

### 15.5 Reference-style markdown links

| Input | Behavior |
|---|---|
| **Inline link** (`[text](path/file.md#heading)`) | Resolved at extraction; if target is a known anchor, contributes a `references` link |
| **Reference-style link** (`[text][ref]` + `[ref]: path/file.md`) | Resolve `[ref]: ...` definitions per CommonMark, then process as inline |
| **Link to anchor that doesn't exist at index time** | Recorded as a candidate `references` link with `drift_status: broken-target` |

---

*Last updated: 2026-05-07*
*Status: design v2 — incorporates all 20 swarm-review decisions*
