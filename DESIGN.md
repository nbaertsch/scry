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
rebase upsert is written, the new record carries two additional
fields — `prior_from_content_hash` and `prior_to_content_hash` — that
record what each endpoint hashed to *before* the rename, so subsequent
drift checks can still see rename-with-edit as drift, not as `fresh`.
Per-endpoint fields (rather than a single `prior_content_hash`) are
required because a rebase may detect a fingerprint match on one
endpoint while the other endpoint legitimately changed; collapsing
both into a single field would discard the disambiguating signal.

The §5.1 drift algorithm consults these fields explicitly: see the
"prior-hash override" rule in §5.1 — if either prior field is set
and differs from the current canonicalized content hash for the
corresponding endpoint, the link's drift status is escalated to
`spec-changed` / `code-changed` / `both-changed` rather than
collapsing to `fresh`.

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
| **Overlay** | `.scry/overlays/<overlay-slug>.jsonl` | Per-branch session state (rebases, proposals, pending changes) | Gitignored |

**Active link table = `replay(baseline) ⊕ replay(overlay-for-current-ref)`.**

#### 3.5.0 Overlay slug derivation

The `<overlay-slug>` is derived from the current git ref via these
rules (always resolved through this function — never from the raw
branch string):

1. Resolve the active ref via `git symbolic-ref --short -q HEAD`. If
   that returns a branch name, use it as input. If it fails (detached
   HEAD), use `detached-<short-sha>` where `<short-sha>` is the first
   12 characters of the current commit SHA.
2. Apply `urllib.parse.quote(name, safe='')` to the input. This URL-
   encodes every character that would be problematic in a flat
   filename (`/`, `\`, `:`, `*`, `?`, `"`, `<`, `>`, `|`, etc.). The
   result is a single flat filename with no path separators.
3. If the URL-encoded result exceeds 200 characters (filesystem path
   length safety margin), replace with `long-<sha256[:12]-of-input>`.
4. Append `.jsonl`.

Examples:
- `main` → `main.jsonl`
- `feature/auth-login` → `feature%2Fauth-login.jsonl`
- `dependabot/npm/foo` → `dependabot%2Fnpm%2Ffoo.jsonl`
- detached HEAD at `abc123def456...` → `detached-abc123def456.jsonl`

**Implications for case-insensitive filesystems** (macOS HFS+/APFS
default, NTFS): two ref names that differ only in case (`feature/AUTH`
vs `feature/auth`) collide on a single overlay file. This matches
git's own behavior on these filesystems and is not separately
mitigated.

**Detached HEAD overlay records are session-local.** When HEAD is
detached, `propose_link` / `accept_link` / `commit-links` succeed
normally and write to the SHA-tagged overlay; CI workflows running
`scry check --ci` produce no overlay writes and so are safe across
parallel jobs. Overlay garbage collection for stale `detached-*`
files is handled by `scry vacuum` (deferred per §13).

The overlay layer means scry can produce session-state link mutations
(auto-rebases, proposals, pending accepts) without ever mutating the
committed baseline file. Branch switching is cheap: the baseline file
swaps automatically on `git checkout` (it's a tracked file), and the
overlay file for the new ref is loaded from `.scry/overlays/`.

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
  "prior_from_content_hash": "sha256:...",   // optional; set on rebased `from` endpoint
  "prior_to_content_hash":   "sha256:...",   // optional; set on rebased `to` endpoint
  "commit_sha": "abc123def456...",       // git HEAD at upsert time
  "worktree_dirty": false,                // working tree dirty at upsert?
  "event_id": "evt_01HXY...",             // immutable per-record UUIDv7; distinct from link_id
  "supersedes": "evt_01HXW...",           // REQUIRED if link_id already exists in this file or in baseline; references the prior record's event_id
  "evidence": "Optional pull-quote",
  "ts": "2026-05-08T12:34:56Z"
}

// Delete (tombstone)
{
  "op": "delete",
  "link_id": "lnk_abc",
  "event_id": "evt_01HXZ...",             // immutable per-record UUIDv7
  "supersedes": "evt_01HXY...",           // REQUIRED; references the prior record's event_id
  "ts": "2026-05-08T13:00:00Z",
  "reason": "manual"
}
```

#### 3.5.2 Replay rules

Records have two distinct identifiers:
- **`link_id`** — the logical edge identity. Stable across the whole
  history of a link; `propose_link` and `accept_link` reuse it.
- **`event_id`** — an immutable per-record UUIDv7. Every `upsert` and
  `delete` carries one. `supersedes` always references an `event_id`,
  never a `link_id`.

This split is what makes the supersedes chain implementable: a
logical link can have any number of records over its lifetime, and
each record can pinpoint *which prior record* it overrides.

1. **Within a single file**, file order = ordering. Last record for a
   given `link_id` wins.
2. **Replay order**: baseline first, then overlay. Overlay records
   can supersede baseline records via `supersedes`.
3. **Tombstones are absorbing within the same file** — once a `delete`
   for `link_id L` appears in a file, a subsequent `upsert` for L in
   that same file is a validation error at write time.
4. **Different files (baseline vs overlay) can revive**: an overlay
   `upsert` after a baseline `delete` is allowed and treated as "user
   re-authored after deletion." The revival upsert MUST carry
   `supersedes: <baseline-tombstone-event-id>` to chain back to the
   prior canonical state — even though the prior `event_id` lives in
   a different file. This is the **cross-file supersedes** case.
5. **`supersedes` is required** on every `upsert` whose `link_id`
   already exists in the file being read OR in the baseline (when
   writing to overlay). Validated at write time. Missing `supersedes`
   on a duplicate `link_id` = error.
6. **Post-union-merge of baseline**: the `supersedes` chain provides
   logical ordering even when file order is non-semantic. If the chain
   is well-formed, the active state is deterministic. If broken (two
   upserts both claim to supersede the same prior `event_id`, or the
   referenced `event_id` is not present in baseline ⊕ overlay), surface
   as a `merge-conflict` event in `scry status`.

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

**Two-phase atomicity** (crash- and disk-full-safe):

1. Append the promoted records to `links.jsonl`; write a
   `.scry/commit-links.<txn-id>.marker` file recording the promoted
   `event_id`s and the source overlay path. `fsync` both files.
2. Remove the promoted records from the overlay file (atomic write
   via `tempfile + rename`). `fsync` the directory.
3. Delete the marker file. `fsync` the directory.

If scry crashes between steps 1 and 3, startup recovery reads the
marker, detects whichever steps remain, and finishes them. The
invariants are: (a) once step 1 succeeds, the promoted records are
considered baseline; (b) the marker prevents duplicate promotion if
the user runs `scry commit-links` twice covering the same records.

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
     in each list. **After deduplication, each list is re-ranked
     1..N by parent before applying the RRF formula in step 4.**
     Re-ranking is critical: without it, a long parent's many
     sub-chunks crowd out single-chunk parents in the lower rank
     positions, re-introducing length bias in inverse form.
  4. RRF-fuse the two parent-ranked lists:
       parent_score = Σ_{list ∈ {vec, bm25}} 1 / (k + parent_rank_in_list)
       (k from retrieval.fusion_rrf_k, default 60; parent_rank_in_list
        is the post-promotion 1..N rank from step 3)
  5. Sort parents by RRF score; take top_k
  6. For each result, populate the anchor packet (§4.2):
       - Pull the evidence excerpt from the **vector** list's
         best-matching sub-chunk for that parent. (When the BM25
         and vector best-chunks differ, the vector pick is preferred
         because it captures semantic relevance; the BM25 chunk is
         available via `get_anchor(id)?evidence_for=bm25`.)
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
closure hashes.

**Status (one value):**

| status | When |
|---|---|
| `fresh` | Both endpoints' content hashes match the stored values |
| `spec-changed` | Spec/doc endpoint content hash changed |
| `code-changed` | Code endpoint content hash changed (own AST + transitive closure where supported — see §5.3) |
| `both-changed` | Both endpoints' content hashes changed |
| `broken-source` | Source anchor no longer exists (and no rebase candidate matched) |
| `broken-target` | Target anchor no longer exists (and no rebase candidate matched) |
| `merge-conflict` | The link's `supersedes` chain has multiple latest upserts post-merge, or references an unknown `event_id`. User resolves via `scry status`. |
| `drift-unknown` | LSP errored on a code anchor whose closure would have been needed to evaluate this link. Text hashes match (so not `code-changed`), but the closure-derived signal is missing. CI policy default: **fail** (treat as drift), so silent passes from broken LSPs are surfaced. Override with `scry check --ignore-lsp-error`. |

**Status precedence** (highest first):
`merge-conflict` > `broken-source` / `broken-target` >
`both-changed` > `spec-changed` / `code-changed` > `drift-unknown` >
`fresh`

**Rationale for precedence**: `merge-conflict` ranks above `broken-*`
because a conflict means the link's *active state itself* is not
trustworthy — until resolved, scry cannot even assert which endpoint
ID is current. A `broken-*` status, by contrast, is a confidently-
known state that can be repaired through normal flow.

**Prior-hash override** (§3.3 consumption rule): on every link
evaluation, before assigning the precedence-resolved status:
- If `prior_from_content_hash` is set on the latest upsert and
  `current_from_content_hash != prior_from_content_hash`, treat
  the `from` endpoint as changed (escalate to `code-changed` or
  `spec-changed` per the endpoint's anchor type).
- Same rule for `prior_to_content_hash` and the `to` endpoint.
- If both prior fields are set and both endpoints have changed,
  escalate to `both-changed`.

This rule is what makes the rebase-with-edit case visible: after a
rebase upsert, the `from_content_hash` and `to_content_hash` reflect
the new (post-rename) state, so a naive `current == stored` check
would resolve to `fresh`. The `prior_*` fields preserve the pre-
rename signal so legitimate concurrent edits aren't masked.

**Semantic-drift is a separate boolean flag**, not part of the
precedence ladder. For `mirrors` links, scry computes
`semantic_drift: true` when both endpoints' embeddings have cosine
distance > `drift.semantic_drift_threshold` (default 0.25), regardless
of the textual `status`. The flag is emitted on every `mirrors` link
in the anchor packet:

```jsonc
{"to": "...", "type": "mirrors",
 "drift_status": "code-changed",
 "semantic_drift": true,
 "transitive_hash_status": "complete"}
```

This is a deliberate change from earlier designs that placed
`semantic-drift` in the precedence ladder: a `mirrors` link with
both hash drift AND embedding drift was reported as `code-changed`,
*hiding the strongest available signal*. As an independent flag, the
behavior-divergence signal is always visible alongside whatever the
text-based status resolved to.

**Cross-language `mirrors` semantic drift**: when the two `mirrors`
endpoints are in different languages, embeddings live in different
regions of the model's space and a 0.25 cosine threshold is not
calibrated. Scry detects this case (endpoint anchor types' resolved
languages differ) and emits `semantic_drift: null` with a warning in
`scry doctor` output. Users wanting cross-language semantic drift
must configure a `drift.cross_language_threshold` explicitly.

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
| `merge-conflict` | 1.0 |
| `broken-*` | 1.0 |
| `both-changed` | 0.5 |
| `spec-changed` | 0.3 |
| `code-changed` | 0.3 |
| `drift-unknown` | 0.3 |
| `semantic_drift: true` (flag, mirrors only) | 0.2 (added on top of base status weight) |

Score is clamped to `[0, 100]`. Independent of repo size.

**Coverage score** (0–100, or `null` for repos with no code anchors):
```
coverage_score = 100 × (linked_code_anchors / max(1, total_code_anchors))
               (returns `null` when total_code_anchors == 0)
```

`null` distinguishes "no code indexed" from "0% of code linked." CI
policy should treat `null` as not-applicable.

**The `max(1, ...)` guard is for code clarity, not arithmetic
necessity** — the early `null` return in both formulas means
denominator-zero is unreachable. Keep the guard so the algebraic
form stays self-documenting.

**Always emitted alongside the scores: raw counts** so CI policy can
gate on counts directly:

```jsonc
{
  "drift_score": 92.3,
  "coverage_score": 67.5,
  "counts": {
    "broken_source": 1, "broken_target": 1, "merge_conflict": 0,
    "both_changed": 0, "spec_changed": 8,
    "code_changed": 1, "drift_unknown": 0,
    "semantic_drift_flagged": 0,
    "fresh": 145, "total": 156
  },
  "by_anchor_type": {...}
}
```

Empty-repo emission shape (for clarity to CI tooling):

```jsonc
{
  "drift_score": null,
  "coverage_score": null,
  "counts": {
    "broken_source": 0, "broken_target": 0, "merge_conflict": 0,
    "both_changed": 0, "spec_changed": 0,
    "code_changed": 0, "drift_unknown": 0,
    "semantic_drift_flagged": 0,
    "fresh": 0, "total": 0
  }
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
| `complete` | LSP returned a complete `callHierarchy` result; closure includes all reachable in-repo callees (per the closure-boundary rule below) |
| `partial` | At least one symbol in the closure was unresolvable: LSP returned `null`, the symbol resolved outside the include set, OR a recursion-depth cap was hit. The closure hash incorporates whatever was resolved. **A leaf function with zero outgoing calls is `complete` (vacuously), not `partial`.** |
| `unsupported` | LSP doesn't implement `callHierarchy` for this symbol's language (probed at startup); only the anchor's own AST text is hashed |
| `lsp_unavailable` | The LSP binary is not installed on PATH; per-language probe failed; only the anchor's own AST text is hashed |
| `lsp_error` | LSP query failed at runtime (crash, timeout, malformed response); falls back to AST-only hash. Drift evaluation surfaces `drift-unknown` (§5.1) when this anchor participates in a link. |

Agents seeing `drift_status: "fresh"` with `transitive_hash_status:
"unsupported"` know the signal is weaker than `complete`.

**Closure boundary**: walks `callHierarchy/outgoingCalls` until a
called symbol is matched by the config `exclude:` globs OR is not
matched by any `include:` glob. This ties the closure boundary to
the same include/exclude system as anchor extraction — no separate
hardcoded list of `node_modules` etc.

**Cycle detection**: the closure walk maintains a per-anchor visited
set keyed by symbol ID. When a symbol is reached that is already in
the visited set, the walk terminates that branch (the cycle does not
contribute additional hashes to the closure). Maximum recursion depth
is bounded by `code_anchors.transitive_max_depth` (default 32) to
defensively cap pathological cases that escape simple cycle detection
(e.g., LSP returning fresh symbol IDs for what should be the same
function across overload resolutions).

**Status propagation across the closure**: the `transitive_hash_status`
of an anchor incorporates the worst status of any callee in its
closure. Concretely:

```
A.transitive_hash_status =
    min(A.self_lsp_status,
        min(callee.transitive_hash_status for callee in A.closure))
```

ordering `complete > partial > unsupported > lsp_error` (best to
worst). So if A successfully queries its own callees but one of them
(B) returned `partial`, A's reported status is `partial` — the
weakest link in the chain governs the trust signal.

**`unsupported` vs `lsp_unavailable` vs `lsp_error`**: the enum has
four values to distinguish causes that demand different user actions:

| value | Meaning | User action |
|---|---|---|
| `complete` | LSP returned a complete `callHierarchy` result; closure includes all reachable in-repo callees | None |
| `partial` | LSP returned partial results for at least one symbol in the closure | Investigate via `scry doctor`; may be a capability gap, may be transient |
| `unsupported` | The configured LSP for this language declared it does not support `callHierarchy` (probed at startup) | None — accept weaker drift signal, or switch LSPs |
| `lsp_unavailable` | LSP binary not installed on PATH; `transitive_hash_status` cannot be computed at all for this language | Run `scry doctor` for install instructions |
| `lsp_error` | LSP query failed at runtime (crash, timeout, malformed response) | Surface in `scry doctor`; may be a recoverable transient or a bug |

Agents seeing `drift_status: "fresh"` with `transitive_hash_status:
"unsupported"` know the signal is weaker than `complete`.

**Field placement** — `transitive_hash_status` lives on the **anchor**
(stored in `vectors.db` per code anchor) and is *projected* onto each
link in the anchor packet. For non-code link targets (`section`,
`code_in_doc` to non-code), the field is **omitted entirely from the
JSON** (not `null`) so agents can use `"transitive_hash_status" in link`
as a presence check.

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
  bm25:
    enabled: true
    index_table_cells: true    # if true, markdown table cell text is included in FTS5
  links_per_result:
    outgoing: 5
    incoming: 5
  content_preview_tokens: 500  # cap on the anchor packet `content` field

# Code anchor closure depth (defensive cap; cycle detection runs
# regardless — see §5.3)
code_anchors_extra:
  transitive_max_depth: 32

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
   (~3-5ms warm on Linux/macOS local-disk; 10-50ms cold or on Windows;
   100ms+ on network filesystems) and `git symbolic-ref --short -q HEAD`
   (returns branch name or fails for detached HEAD — handled per §3.5.0).
2. Compare to `index_metadata.indexed_git_head` and `indexed_branch`.
3. **If branch changed** (including detach/reattach transitions): swap
   the active overlay file via the §3.5.0 slug derivation, and treat
   the file set as fully changed (run the same diff path as a HEAD
   change, since the working tree contents may differ across branches).
4. **If HEAD changed within the same branch**: detect changed files
   via `git diff --name-only <indexed_head> <current_head>` and
   incrementally reindex only those files. If the prior `indexed_head`
   is no longer reachable in git (force-push, garbage-collected
   commit), fall back to a full file-manifest comparison.
5. **If config_hash changed**: full reindex (config affects extraction
   semantics). The hash is computed over the *parsed and canonicalized*
   YAML dict, not raw bytes, so cosmetic reformatting does not trigger
   reindex.
6. **MCP responses include `"index_state"`**: one of `"fresh"`,
   `"stale-reconciling"` (incremental reindex in progress for this
   process; results may reflect partial state), `"stale-no-write-lock"`
   (reconcile required but the leader is unavailable for writes;
   results reflect last-good index), `"stale-warned"` (reconcile would
   exceed `index.auto_reconcile_max_changed_files` (default 500); user
   must explicitly run `scry index` to opt in to a large reindex —
   this prevents agent-triggered branch switches from blocking on
   30-second indexes).

**Per-process HEAD cache** with configurable refresh interval
(default 30 seconds, tunable via `index.head_poll_interval_seconds`).
The cache is **invalidated immediately** on any tool call that may
mutate state (`propose_link`, `accept_link`, `commit_links`,
`reindex`) so writes never go to a stale overlay. Read-only tools
(`search`, `get_anchor`, `get_links`, `find_drift`, `repo_summary`,
`status`) honor the cache to keep tight loops fast. Set the interval
to `0` to disable caching entirely.

**Dirty-worktree polling**: at the same poll, scry runs
`git status --porcelain=v1 -uno --no-renames` (10-50ms) to detect
uncommitted changes to indexed files. Modified files trigger an
incremental reindex of just those paths. This catches the most common
dev scenario: the user edits a spec, then immediately asks the agent
to search — without dirty-worktree polling, the agent sees pre-edit
state until commit. Disable via `index.poll_dirty: false` for
ultra-low-latency setups.

**Polling rationale**: multi-user consistency — every collaborator
gets the same behavior without per-clone hook installation. Polling
is also defensive against IDE git operations that may bypass hooks.

#### 7.2.1 Embedding-model mismatch (HARD ERROR + `--reembed` recovery)

On any mismatch in `embedding_*` fields (provider, model, dimensions,
tokenizer): **HARD ERROR.** Refuse to serve. Tell user:

> *"Embedding configuration changed. Run `scry index --reembed` to
> re-embed existing anchors with the new model (preserves anchors,
> fingerprints, and links). Use `scry index --force` only for
> suspected data corruption."*

`scry index --reembed` is the surgical migration path:

1. Resolves the new model + provider; computes new
   `embedding_dimensions`. If `new_dim != stored_dim`, the sqlite-vec
   virtual table is incompatible (column dimensionality is fixed at
   table creation), so reembed performs a **drop+recreate** of the
   vector table inside the same SQLite transaction as the metadata
   update. Anchor rows, fingerprints, FTS5 index, links.jsonl, and
   overlays are preserved across the drop/recreate.
2. **Source-text storage**: anchor records in `vectors.db` persist
   the canonicalized anchor text in a `content_text` column (not just
   the hash and embedding). This allows reembed to operate without
   reading from disk and survives source-file deletion. Disk-cost
   estimate: +30-50% on `vectors.db` for typical specs; tradeoff
   accepted for migration robustness.
3. **Prune-then-reembed**: before re-embedding, scry runs the
   anchor-existence check against current files. Anchors whose source
   file no longer exists in the working tree are deleted (and any
   links pointing at them are marked `broken-source` / `broken-target`
   in the overlay). Then reembed processes the remaining anchors in
   batches.
4. Updates `index_metadata.embedding_*` fields atomically with the
   final batch.

**Crash safety**: every batch is its own transaction; on crash mid-
reembed, partial progress is durable and `scry index --reembed` can
be resumed. The metadata fields are only updated in the *final*
batch's transaction, so a partial reembed leaves the old `embedding_*`
fields intact and will fail the model-mismatch check on next startup
(causing the user to re-run `--reembed`, which resumes).

Cost: only the embedding work plus the prune scan; skips parsing,
AST extraction, LSP queries, and link processing. **For local
embedders** (the default `fastembed` path), this is roughly 2-3×
faster than `--force` on a hailstorm-scale repo. For cloud embedding
providers (OpenAI, Voyage, Cohere) where embedding latency dominates,
the savings are smaller (skipping parsing/LSP saves seconds; the
embedding API calls take minutes). Rebase capability is fully
preserved because anchors and fingerprints never disappear.

**Concurrent edits during `--reembed`**: scry takes the leader-write
lock (§10) for the duration of reembed. Followers are read-only
during this period and serve from the pre-reembed embedding column
until the final transaction commits. After commit, follower reads
automatically see the new embeddings via WAL.

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
| `search(query, types?, top_k?, exclude_tests?)` | Hybrid BM25 + vector retrieval; returns ranked anchor packets |
| `get_anchor(anchor_id)` | Full content of an anchor by ID (no truncation) |
| `get_links(anchor_id, link_types?, direction?)` | Bidirectional link enumeration; inverse names rendered for `direction=incoming` |
| `find_drift(scope?, status_filter?, since?)` | Evaluate section-level drift for active links; `since` accepts a git ref for diff-scoped results |
| `propose_link(from_id, to_id, link_type, evidence?)` | Stages a link in the overlay (§3.5.4) |
| `accept_link(proposed_id)` | Marks an overlay-staged proposal as accepted (still overlay; promote with `commit_links`) |
| `commit_links(scope?)` | Promote accepted overlay records to the baseline `links.jsonl` |
| `unlink(link_id, reason?)` | Tombstone a link; the `link_id` is permanently reserved |
| `status()` | Return pending overlay records, merge conflicts, index state |
| `repo_summary()` | One-shot orientation: anchor counts, drift + coverage scores |
| `reindex(scope?, force?)` | Force re-extraction (default is incremental on file change) |
| `get_callers(anchor_id, max_depth?)` | LSP-backed: symbols that call a given code anchor |
| `get_subclasses(anchor_id)` | LSP-backed: classes that extend a given class anchor |
| `suggest_links_candidates(scope?, source?, limit?)` | Surface (code, doc) pair candidates + classifier prompt for agent-side LLM classification |
| `apply_link_suggestions(suggestions, pair_payloads, min_confidence?, apply?)` | Apply or preview agent-classified link suggestions |

All write tools accept an `idempotency_token` parameter for safe retries.
All tools carry explicit `readOnlyHint` / `destructiveHint` / `idempotentHint`
annotations so MCP clients can distinguish safe queries from mutations.

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
| **Leader** | Holds `.scry/leader.lock` (PID + IPC endpoint URI + boot epoch token). Owns all writes to `vectors.db` and overlay files. Exposes IPC endpoint (Unix socket on macOS/Linux; Windows named pipe). Performs auto-reconcile. Performs scry index operations. Serves MCP tools normally to its agent. **Internally serializes all overlay-write operations via an asyncio lock keyed by overlay path** so concurrent IPC requests cannot interleave bytes in the same JSONL file. |
| **Follower** | Detects leader alive via `.scry/leader.lock`. Opens `vectors.db` read-only (each tool call wraps its query in a fresh short read transaction so it sees the leader's latest commit; long-lived read transactions are forbidden by convention). Serves all read tools (`search`, `get_anchor`, `get_links`, `find_drift`, `repo_summary`, `status`) directly from the read-only DB. Forwards write operations (`propose_link`, `accept_link`, `commit_links`, `reindex`) to the leader via IPC; reads back results. Polls git context like the leader (so search results reflect current branch correctly). |

Followers see leader writes immediately because `vectors.db` is shared
in WAL mode and concurrent readers see committed writes — provided
they refresh their read transaction per call (see Follower row above).

**Worktrees**: scry detects the worktree topology via
`git rev-parse --git-common-dir` vs `--git-dir`. The
`.scry/` directory lives at the *worktree* root (not the common dir)
so each worktree has independent `vectors.db`, overlays, and leader.
This is a deliberate cost-vs-isolation tradeoff: per-worktree state
means no cross-worktree contention, at the cost of duplicated
embedding work for anyone running multiple worktrees off the same
repo. A future flag `scry init --share-vectors` may opt into a shared
`vectors.db` at the common dir; for the initial release, every
worktree pays its own indexing cost. Documented in `scry doctor`
output when worktrees are detected.

### 10.2 Leader election and failover

On startup, every `scry mcp` instance:

1. **Acquire the leader lock first** (`fcntl.flock` / `msvcrt.locking`
   on `.scry/leader.lock`). The OS-level exclusivity is the source of
   truth for "who is the leader" — *not* the PID written in the lock
   file. This ordering eliminates the cold-start race where DB
   verification and auto-reconcile happen before lock acquisition.
2. **If lock acquired**: this instance is the leader.
   a. Bind the IPC endpoint (Unix socket or named pipe).
   b. Once the listener accept loop is running, write
      `{pid, endpoint_uri, boot_epoch_token, scry_version}` to
      `.scry/leader.lock`. The boot-epoch token (random UUIDv4
      generated at process start) lets stale-lock detection
      distinguish "PID 12345 is alive but it's now `notepad.exe`"
      from "PID 12345 is the same scry process from before."
   c. Run DB verification + auto-reconcile (§7.2) under the held
      lock. Followers attempting to connect during this window get
      a transient "leader-warming" error and back off.
   d. Mark the leader as ready; followers can now connect.
3. **If lock held by another**: this instance is a follower. Read the
   lock file to find the leader's IPC endpoint URI. Connect.
4. **Stale lock detection**: scry treats the OS lock itself as the
   primary liveness signal — locks are released automatically by the
   kernel when a process dies, so `flock`/`msvcrt.locking` will
   succeed for the next process if the prior leader crashed. If the
   lock is held but the recorded `pid` no longer exists OR the
   recorded `boot_epoch_token` does not match the current process's
   token after a known scry-version rotation, log a warning and let
   the OS lock decision stand. **Do not force-steal locks based on
   PID alone** — PID recycling on Windows (and overflow on Linux)
   makes PID-only checks unreliable.

On leader exit:
- Followers detect the IPC endpoint is gone (connection failure).
- Each follower attempts to acquire the leader lock (§10.2 step 1).
  The first one wins; it becomes the new leader.
- The transition is transparent to agent harnesses (their `scry mcp`
  process keeps serving; only the internal role changed).
- **In-flight writes lost across leader transitions**: if a follower's
  `propose_link` was acknowledged by the old leader but not yet
  durably written when the leader crashed, the follower's IPC call
  returns an error AND the new leader has no record of the write. The
  follower's MCP client surface returns an error to the agent harness;
  the agent decides whether to retry. Idempotency tokens (§10.3)
  prevent duplicate writes when the agent retries.

### 10.3 IPC protocol

Lightweight JSON-over-stream over Unix socket / Windows named pipe.
Request/response shapes mirror the MCP tool surface for write tools:

```jsonc
// Follower → Leader
{
  "id": 42,                           // per-connection request sequence
  "op": "propose_link",
  "args": {...},
  "idempotency_token": "tok_01HXY...", // required for write ops
  "protocol_version": 1
}

// Leader → Follower
{"id": 42, "ok": true,  "result": {...}}
{"id": 42, "ok": false, "error": "...", "error_type": "..."}
```

**Idempotency** (write operations): every follower-originated write
(`propose_link`, `accept_link`, `commit_links`, `reindex`) carries a
`idempotency_token` (UUIDv7) that the agent harness or follower
generates per logical operation. The leader maintains an LRU cache of
the most recent ~10,000 tokens with their results. A repeat token
returns the cached result without re-executing the write. This means:
- A timed-out follower retry is safe (idempotent).
- An agent that replays the same `propose_link` after a transient
  error gets the same `link_id` back, not a duplicate.
- The cache survives within a leader's lifetime; after leader
  failover, the new leader has an empty cache (so retries across
  failover may run the operation a second time — but `propose_link`
  is itself naturally idempotent on `from + to + type`, and
  `commit_links` is idempotent on `event_id` set).

**Per-operation timeouts**: short-running ops (`propose_link`,
`accept_link`, `status`) default to 5 seconds. Long-running ops
(`commit_links`, `reindex`) default to "no timeout, but with a
heartbeat every 10 seconds." The follower keeps the connection open
for the duration; if the heartbeat lapses for >30 seconds, the
follower assumes the leader is hung and returns an error. Per-op
timeouts are configurable via `ipc.timeouts.<op>` in
`.scry/config.yaml`.

**Endpoint URI conventions** (cross-platform):

| Platform | Scheme | Example |
|---|---|---|
| Linux / macOS | `unix:` | `unix:.scry/scry.sock` (path is repo-relative; resolved against the repo root) |
| Windows | `pipe:` | `pipe:scry-<sha256[:16]-of-repo-path>` (the pipe path becomes `\\.\pipe\scry-<sha256[:16]>`) |

The leader writes the URI to `.scry/leader.lock`; followers parse the
scheme prefix to dispatch to the right transport.

**Endpoint security**:
- Unix sockets: created with mode `0600` (owner read+write only). The
  socket file lives inside `.scry/` which inherits the repo's user
  ownership. The leader rejects any connection whose `SO_PEERCRED`
  UID does not match its own UID.
- Windows named pipes: created with a restrictive DACL granting
  access only to the current user's SID (via `pywin32`'s
  `CreatePipe` + `SECURITY_ATTRIBUTES`). The leader rejects
  cross-user connections at accept time.

This blocks the multi-user-Windows scenario (RDP/Citrix/dev VM)
where another user's process could discover the pipe via the
`leader.lock` file and forge write requests.

### 10.4 Cold start (no other process running)

When the only process running is the leader, behavior is:

1. Load `.scry/config.yaml`
2. **Acquire leader lock first** (§10.2 step 1). This is the
   serialization barrier — no DB writes happen before this.
   - **2b. Stale-PID recovery (Windows):** if `try_acquire` fails,
     read the lock metadata and check if the recorded PID is alive
     via `os.kill(pid, 0)`. If the PID is dead (process was
     force-killed or orphaned by a parent `uv.exe` termination),
     remove the stale lock file and re-acquire. This prevents
     dangling locks from blocking subsequent startups after unclean
     shutdowns (e.g. Copilot CLI session kill).
3. Bind IPC endpoint (Unix socket or Windows named pipe) per §10.3.
4. Open vector store read-write
5. Verify `.scry/vectors.db` exists; auto-reconcile (§7.2) or
   hard-error on embedding-model mismatch. Both happen under the
   held leader lock so a concurrent `scry mcp` startup cannot
   interleave writes.
6. Once auto-reconcile is complete, mark the lock file as ready
   (write the leader metadata: PID, endpoint URI, boot epoch token,
   scry version).
7. Start the IPC accept loop.
8. Serve MCP tools over stdio.
9. Lazy-load embedder model on first `search` call.

When the leader exits, its lock is released. The next `scry mcp` to
start is a fresh leader.

### 10.5 LSP binary spawning on Windows

The LSP allowlist in §6.2 lists binary *names* (not full paths). On
Windows, name resolution must handle two complications:

1. **Extension resolution**: `shutil.which` honors `PATHEXT`, but the
   resolved path may be `.exe`, `.cmd`, or `.bat`. Allowlist matching
   is done against the **stem** (filename without extension), so
   `pyright-langserver.cmd` matches the `pyright-langserver` allowlist
   entry.
2. **Shim spawning**: `.cmd` and `.bat` shims (npm-installed
   `typescript-language-server` is the common case) cannot be spawned
   directly via `subprocess.Popen([resolved_path, ...])` —
   `CreateProcess` rejects them with WinError 193. Scry detects
   `.cmd`/`.bat` extensions and spawns via `cmd.exe /C "<shim>" <args>`
   transparently. The pre-spawn argument validation enforces the
   allowlist so this does not weaken the security model.

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
- Anchor `content_text` column persisted (per §7.2.1 reembed
  requirement)
- Two-tier consistency invariants (parent_content_hash on chunks,
  transactional reindex with metadata in same transaction,
  read-side hash-equality filter, advisory write lock)
- `links.jsonl` baseline + `.scry/overlays/<overlay-slug>.jsonl`
  per-branch overlay; event-record reader/writer with
  `.gitattributes union` driver; replay rules from §3.5.2 (incl.
  `event_id` and cross-file supersedes)
- `scry commit-links` two-phase atomicity protocol (§3.5.4)
- Hybrid retrieval algorithm (BM25 + vector + RRF with
  best-chunk-per-parent promotion AND post-promotion re-ranking)
- MCP server skeleton: leader election (lock-first ordering, §10.2),
  IPC listener, follower mode, idempotency token cache
- Tools (Wave 2): `search`, `get_anchor`, `get_links`,
  `repo_summary`, `reindex`, `status`, `propose_link`, `accept_link`,
  `commit_links`, `scry link`
- **Section-level drift** (`spec-changed`, `broken-source`,
  `broken-target`, `merge-conflict`): no LSP needed; ships in Wave 2.
  `repo_summary` in Wave 2 returns `drift_score` based on these
  status values only (the `code-changed` and `drift-unknown` counts
  remain `0` until Wave 4 activates them); the response includes
  `"drift_coverage": "section-only"` so consumers know the score is
  partial.
- Polling-based git-context detection at every tool call:
  HEAD + branch + dirty-worktree polling (§7.2); cache invalidation
  on write tools
- `index_state` field on every MCP response (Wave 2 hardcodes
  `"fresh"`; Wave 3 activates the other values)
- Lazy embedder model loading
- `scry doctor`, `scry validate`

### Wave 3 — LSP-resolved code anchors
- LSP subprocess infrastructure (JSON-RPC over stdio, lifecycle, caching)
- LSP binary allowlist (§6.2); `--allow-untrusted-lsp-config` flag
- Windows `.exe`/`.cmd` shim spawning (§10.5)
- Per-language LSP adapters (pyright-langserver, typescript-language-server, zls)
- `callHierarchy/outgoingCalls` for transitive closure hashing with
  per-anchor cycle detection and `transitive_max_depth` cap (§5.3)
- `transitive_hash_status` enum on every code anchor with closure
  propagation rule (§5.3)
- LSP capability probing in `scry doctor`; distinguishes
  `unsupported` (LSP doesn't implement callHierarchy for the
  language) vs `lsp_unavailable` (binary not on PATH); each produces
  a distinct error message and recovery suggestion at indexing time

### Wave 4 — Code-level drift + auto-reconcile
- `code-changed` drift status (uses Wave 3 transitive closure)
- `drift-unknown` drift status for `lsp_error` cases (§5.1)
- `semantic_drift` boolean flag for `mirrors` links
  (embedding-distance check; cross-language emits `null` per §5.1)
- `find_drift` MCP tool
- `scry check` CLI with `--ci` exit codes; coverage_score; raw counts;
  `--require-fresh-embedder` flag; `--ignore-lsp-error` flag (§5.1)
- Auto-reconcile-on-startup activates the non-`"fresh"` `index_state`
  values (`"stale-reconciling"`, `"stale-no-write-lock"`,
  `"stale-warned"`)
- Inline rebase on rename via embedding similarity + SimHash
  confirmation; rebase records written to overlay
- `scry index --reembed` migration path with drop+recreate vector
  table on dimension change, prune-then-reembed for deleted source
  files (§7.2.1)
- `prior_from_content_hash` / `prior_to_content_hash` fields on
  rebased upserts so drift reflects rename-with-edit signal (§3.3
  + §5.1 prior-hash override rule)

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
   version in `index_metadata`.
4. **Code-in-doc detection thresholds** — when is a fenced code block
   substantial enough to be its own anchor? Default: ≥ 5 lines OR an
   explicit language tag with ≥ 1 named declaration. Performance impact
   when running tree-sitter on every code block during indexing.
5. **IPC perf characteristics** — JSON-over-stream Unix socket / Windows
   pipe round-trip latency for write forwarding; ensure follower
   write operations don't degrade noticeably vs leader-direct.
6. **`scry init --register-global`** — write strategy for
   `~/.claude.json` / `~/.cursor/mcp.json` (existing keys, JSON merge
   semantics, backup, file locking on Windows where Claude Desktop
   holds a watch). Default behavior is print-and-let-user-paste.
7. **Anonymous code-block hash collisions** — 8-char prefix has 2^32
   space; collisions become realistic at ~65k anchors (birthday).
   Evaluate during Wave 1 fixture work whether to extend default to
   16 chars.
8. **JSONL replay performance at scale** — for repos with thousands
   of links and many overlay records, replay cost grows. May need
   `scry vacuum` to compact baseline + drop superseded chains, GC
   stale `detached-*` overlay files (§3.5.0), and optionally GC
   overlays for branches that no longer exist locally.
9. **FTS5 tokenizer for non-ASCII content** — default `unicode61`
   handles diacritics but doesn't segment CJK. Document the
   limitation; consider language-aware tokenizer in future.
10. **Polling cost on network-mounted repos** — `git rev-parse HEAD`
    can take 100ms+ on NFS/SMB-mounted `.git` directories. The 30s
    HEAD cache mitigates, but `index.head_poll_interval_seconds`
    and `index.poll_dirty: false` are the user-facing knobs.
11. **Idempotency cache size** — leader maintains LRU of ~10k tokens.
    Unbounded write rate from a runaway agent could displace
    legitimate retries. Evaluate during Wave 2 testing.
12. **`config_hash` canonicalization** — hash the parsed-and-
    canonicalized YAML dict (sorted keys, normalized types) rather
    than the file bytes so cosmetic reformatting doesn't trigger
    full reindex.
13. **`code_anchors.granularity: file`** — option exists in §6 config
    but anchor IDs, drift semantics, and LSP integration are all
    symbol-oriented. Either spec the file-level path (file-level
    anchor ID = `<path>`, no LSP, hash = whole file hash) or remove
    the option before Wave 1 fixtures.

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
| **Malformed YAML frontmatter** | File indexed without frontmatter; warning surfaced via `scry validate`; index proceeds (do not block on bad frontmatter) |
| **File exceeds `index.max_file_size_bytes`** (default 5 MB) | Skipped with warning logged; no anchors created |
| **Binary file with `.md` extension** (UTF-8 decode fails) | Skipped with warning |
| **UTF-16 LE/BE encoded `.md` file** | Detected via BOM (`\xFE\xFF` BE / `\xFF\xFE` LE); skipped with a "transcode to UTF-8 to index" warning |
| **CRLF-only line endings** (Windows-authored) | Canonicalized to LF before hashing (§5.4); indexed normally |
| **Mixed CRLF + LF in same file** | Canonicalization order: replace `\r\n` → `\n`, then bare `\r` → `\n`. Indexed normally. |
| **CR-only line endings** (legacy Mac OS 9 / some embedded tools) | Canonicalized to LF (bare `\r` → `\n` step in §5.4); indexed normally |
| **File matched by `exclude` glob** | Skipped silently. Frontmatter `skip: false` cannot override hard safety excludes (e.g., `secrets/**`). |
| **File matched by frontmatter `skip: true`** | Skipped, even if matched by `include:` |
| **File matched by no `classify` rule** | Excluded from indexing (must classify to participate); warning if matched by `include:` |
| **Symlink to file inside repo** | Followed; if the inode resolves to a file already indexed under another path, deduplicated; the canonical (non-symlink) path wins |
| **Symlink to file outside repo root** | Skipped with warning (security boundary) |
| **Multiple `index_metadata` rows in vectors.db** (corruption case) | Hard error at startup; suggest `scry index --force` |
| **Markdown table cells** | Per `retrieval.bm25.index_table_cells` (default `true`): table cell text is included in the FTS5 index, with pipe (`|`) characters treated as word separators by the tokenizer. Set to `false` to exclude tables (cleaner BM25 scoring at the cost of recall on tabular protocol/enum specs). |

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
| **Circular `derives-from` chain** (A→B→C→A) | Detected at link-write time; `scry link`, `accept_link`, `propose_link` all validate that the new edge does not create a cycle in the `derives-from` subgraph; error surfaced as validation failure. **Post-merge cycles** (two branches each adding non-cyclic edges that union to a cycle) are detected on overlay/baseline replay and surfaced via `scry status` as a `merge-conflict`. |
| **`upsert` with a `link_id` that already exists in the file but no `supersedes` field** | Validation error at write time |
| **`upsert` after a `delete` for the same `link_id` in the same file** | Validation error at write time |
| **`upsert` after a baseline `delete` (cross-file revival, §3.5.2 rule 4)** | Allowed; revival upsert MUST carry `supersedes: <baseline-tombstone-event-id>`. Missing supersedes = validation error. |
| **`supersedes` references an `event_id` not present in baseline ⊕ overlay** | `merge-conflict` drift status; user resolves by writing a new `upsert` superseding both heads. |
| **Two competing latest upserts for one `link_id` post-merge** | `merge-conflict` drift status; surfaced via `scry status`; user resolves by writing a new `upsert` superseding both |

---

*Last updated: 2026-05-08*
*Status: design v3.1 — incorporates all swarm-3 v3.1 patch decisions (7 BLOCKING + 7 HIGH + 6 HIGH-SINGLE)*
