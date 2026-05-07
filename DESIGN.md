# scry — Design

> *"Scry into your codebase."*
> A local-first, hybrid-retrieval MCP server that links spec/doc text
> to AST symbols via a typed graph and detects drift between them.

---

## 1. What it is

Scry indexes a repository at three granularities and links them as a graph:

1. **Spec / doc sections** — heading-bounded markdown blocks
2. **AST symbols** — tree-sitter-extracted functions, classes, types, etc.
3. **Code blocks inside specs** — fenced blocks within a spec section,
   treated as a hybrid first-class anchor type

These three anchor types share one embedding space and one link graph.
Hybrid retrieval combines vector cosine similarity with typed graph
traversal across the links between them. An MCP server exposes the
result to coding agents (Claude Code, Copilot CLI, Cursor, etc.) so
the agent can reason over both *what the spec says* and *what the
code does* in one query.

Drift between linked anchors is detected and surfaced — without an LLM
in the deterministic path — so engineers and agents see when a spec
and its implementing code have diverged.

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

| Anchor type | Source | Example ID |
|---|---|---|
| `section` | A heading-bounded markdown block | `docs/POLICY_ENGINE.md::policy-engine::rule-structure` |
| `code_in_doc` | A fenced code block inside a `section` | `docs/POLICY_ENGINE.md::policy-engine::rule-structure::block-0` |
| `code` | An AST symbol from a source file | `python/hailstone/policy/engine.py:PolicyRule` |

**Stable IDs** are derived from path + structural location, never from
line numbers. They survive line-shuffles, comment additions, and
reformatting; they only change when the heading is renamed (`section`)
or the symbol is renamed/moved (`code`).

### 3.2 Two-tier embedding (the section-chunk solution)

Sections vary in size from 30 words to 8,000+ words. Naive sub-chunking
breaks links. Scry separates **link granularity** from **embedding
granularity**:

```
Parent anchor (link target):
  docs/POLICY_ENGINE.md::policy-engine::rule-structure
  ─ stable ID: yes
  ─ overview embedding (heading + first ~200 tokens)
  ─ content_hash for drift
  ─ this is what links point at and what users/agents see

Sub-chunks (internal cache, retrieval optimization):
  ::rule-structure#chunk-0, #chunk-1, ...
  ─ NOT stable IDs (regenerated on re-chunk)
  ─ embedding of just that sub-chunk text
  ─ used INTERNALLY for retrieval scoring; NEVER returned as a link target
  ─ NOT a separate drift unit
```

Sub-chunking is triggered when section text exceeds
`max_section_tokens` (default 600). Split priority:
1. Sub-headings inside the section (deeper level than the section's own)
2. Fenced-code-block boundaries (never split inside a code block)
3. Paragraph boundaries (blank lines)
4. Sentence boundaries (final fallback)

Each sub-chunk inherits a configurable overlap (default 50 tokens) with
its predecessor.

### 3.3 The Link

A typed, directed edge between two anchors:

```jsonc
{
  "id": "lnk_<hash>",
  "from": "docs/POLICY_ENGINE.md::policy-engine::rule-structure",
  "from_type": "section",
  "to": "python/hailstone/policy/engine.py:PolicyRule",
  "to_type": "code",
  "link_type": "specifies",          // see §3.4
  "from_content_hash": "sha256:...", // hash at link-creation time
  "to_content_hash": "sha256:...",
  "created_at": "2026-05-06",
  "evidence": "Optional pull-quote or note"
}
```

Links are stored in `.scry/links.jsonl` — append-only, version-controlled,
human-reviewable. One link per line.

### 3.4 Link types (initial vocabulary, extensible)

| Link type | Direction | Meaning |
|---|---|---|
| `specifies` | spec → code | This spec section governs this symbol's behavior |
| `implements` | code → spec | This symbol implements this spec section (inverse of `specifies`) |
| `tests` | code → code | This test exercises this implementation symbol |
| `tests` | code → spec | This test verifies this spec section |
| `examples` | code_in_doc → code | This code-block-in-doc is an example of this symbol |
| `mirrors` | code_in_doc → code | This code-block-in-doc is a contract for this symbol — drift between them is a hard signal |
| `references` | section → section | Cross-document reference (auto-derived from markdown links) |
| `derives-from` | section → section | This spec section is downstream of a higher-level spec |

The vocabulary is stored in `.scry/config.yaml` and can be extended
per-repo. Tool-internal logic only depends on a small subset:
`mirrors` and `specifies` get the strongest drift checks; others are
informational.

---

## 4. The retrieval graph

```
                      ┌──────────────┐
                      │   AGENT      │   (Claude Code, Copilot CLI, etc.)
                      └──────┬───────┘
                             │ MCP stdio
                             ▼
                      ┌──────────────┐
                      │ scry mcp     │   (short-lived, one per session)
                      └──────┬───────┘
                             │ in-process
                             ▼
                      ┌──────────────┐
                      │ scry engine  │
                      ├──────────────┤
                      │ retriever    │ ← hybrid: vector + graph traversal
                      │ link-graph   │ ← typed edges between anchors
                      │ vector-store │ ← embeddings of all anchors + sub-chunks
                      │ ast-extractor│ ← tree-sitter, polyglot
                      │ md-extractor │ ← CommonMark + frontmatter
                      │ drift-checker│ ← deterministic, no LLM
                      └──────┬───────┘
                             │
                  ┌──────────┴────────────┐
                  ▼                       ▼
          ┌──────────────┐        ┌──────────────┐
          │ .scry/       │        │ repo files   │
          │  vectors.db  │        │  *.md, *.py, │
          │  links.jsonl │        │  *.ts, ...   │
          │  config.yaml │        │              │
          │  cache/      │        │              │
          └──────────────┘        └──────────────┘
```

### 4.1 Hybrid retrieval algorithm

```
search(query, anchor_types?, traverse_links=true, top_k=10):
  1. Embed query
  2. Vector cosine search across ALL embeddings
     (parent overview + sub-chunks + code anchors)
  3. Group sub-chunk hits by parent anchor; sum scores into parent
     (parent now has score = max(overview_score, sum(sub_chunk_scores)))
  4. (Optional) traverse links from the top-N seed anchors:
     - 1-hop neighbors get a graph_bonus = α × parent_score × edge_weight
     - edge_weight: mirrors=1.0, specifies/implements=0.8, tests=0.5, ...
     - bonus added to neighbor's existing score (or 0 if not yet ranked)
  5. Final ranking = vector_score + graph_bonus
  6. Return top_k anchors with:
     - the anchor itself (id, type, content)
     - best matching sub-chunk excerpt (when applicable) as evidence
     - linked neighbors (filtered by link types, capped)
```

Parameters tunable via `.scry/config.yaml`: vector top-N seed,
traversal hops (default 1), traversal cap, edge weights per link type.

### 4.2 What an agent sees

A search response is an **anchor packet**:

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
     "type": "specifies", "drift_status": "fresh"},
    {"to": "docs/EXECUTION_PIPELINE.md::policy-middleware",
     "type": "references", "drift_status": "n/a"}
  ]
}
```

The agent now has the spec, the linked code, and per-link drift status
in a single response.

---

## 5. Drift detection

Drift is checked **deterministically** at the **parent anchor level** —
sub-chunks don't have independent drift state.

### 5.1 Drift signals

For each link, drift = (from_hash_changed?, to_hash_changed?):

| from changed | to changed | status | meaning |
|---|---|---|---|
| no | no | `fresh` | nothing changed since link created |
| yes | no | `spec-changed` | doc/spec edited; verify code still satisfies |
| no | yes | `code-changed` | code edited; verify spec still describes it |
| yes | yes | `both-changed` | both moved; full reconciliation needed |
| n/a | yes (gone) | `broken-target` | linked symbol/section no longer exists |
| yes (gone) | n/a | `broken-source` | source anchor was deleted |

For `mirrors` links specifically, an extra signal is checked:
**embedding distance** between the two endpoints. If the cosine
distance exceeds a threshold (configurable, default 0.25) the status
is `semantic-drift` even if hashes match — catches the case where
both endpoints were edited "in sync" but their meanings actually
diverged.

### 5.2 Drift score

`scry check` returns a per-repo score:

```
score = 100 - 10×broken - 5×both-changed - 3×spec-changed - 3×code-changed - 2×semantic-drift
       (floored at 0; capped per category)
```

JSON output for CI gates; markdown output for humans.

### 5.3 Reconciliation (v0.2)

`scry reconcile <link_id>` is the AI-augmented loop:
1. Show the diff for both endpoints since `created_at`
2. Use an LLM to propose: "spec is correct, code needs change X" OR
   "code is correct, spec needs change Y" OR "both need an update"
3. Output the proposed patch as a unified diff for human review
4. On accept, refresh the link's hashes after the edit lands

The deterministic `check` path never calls an LLM. AI is opt-in,
contained, and never blocks correctness signals.

---

## 6. Configuration model

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

# Per-glob classification of markdown files
classify:
  "docs/**.md": spec
  "README.md": doc
  "**/*.md": doc           # default for unclassified

# Anchor extraction tuning
sections:
  max_tokens: 600
  overlap_tokens: 50
  min_section_tokens: 0    # 0 = embed even tiny sections; raise to 30 to skip stubs

code_anchors:
  granularity: symbol      # symbol | file
  symbol_kinds:            # tree-sitter node kinds to extract per language
    python: [function_definition, class_definition]
    typescript: [function_declaration, class_declaration, interface_declaration, type_alias_declaration]
    zig: [FnProto, ContainerDecl]

# Embedding provider
embeddings:
  provider: local          # local | openai | voyage | custom
  model: BAAI/bge-small-en-v1.5
  dimensions: 384

# Retrieval
retrieval:
  hybrid:
    enabled: true
    rrf_k: 60
  graph:
    traverse: true
    max_hops: 1
    max_neighbors_per_seed: 5
    edge_weights:
      mirrors: 1.0
      specifies: 0.8
      implements: 0.8
      tests: 0.5
      references: 0.3

# Drift detection
drift:
  semantic_drift_threshold: 0.25
  scoring:
    broken: 10
    both_changed: 5
    spec_changed: 3
    code_changed: 3
    semantic_drift: 2
```

### 6.1 Per-file overrides via frontmatter

Frontmatter is **optional** — repos that already have well-organized
docs can rely entirely on globs in `.scry/config.yaml`. When present,
frontmatter overrides config:

```markdown
---
scry:
  type: spec
  id: AUTH-LOGIN              # overrides path-derived ID
  exclude: false              # opt out per-file
---
```

---

## 7. Persistence layout

```
<repo>/
├── .scry/
│   ├── config.yaml          ← user-authored; required
│   ├── links.jsonl          ← user-curated; the link graph (committed to git)
│   ├── vectors.db           ← embeddings + content hashes (gitignored)
│   ├── cache/               ← extracted anchors, file hashes (gitignored)
│   └── stats.json           ← drift score history (gitignored)
└── (repo files)
```

Vectors and cache live in the repo (not in a global data dir like
scrybe). This is deliberate:
- Switching branches → switching index → no cross-branch leakage
- Multiple agent sessions in different repos → no daemon coordination
- Easy to gitignore the cache; easy to ship the index in CI artifacts

**Trade-off:** more disk usage (one cache per repo). Acceptable for
the simplicity gain.

---

## 8. MCP tool surface (initial set)

| Tool | Purpose |
|---|---|
| `search(query, types?, top_k?)` | Hybrid retrieval; returns ranked anchor packets |
| `get_anchor(id)` | Full content of an anchor by ID |
| `get_links(anchor_id, link_types?, direction?)` | Bidirectional link enumeration |
| `find_drift(scope?)` | List anchors/links with drift > `fresh` |
| `propose_link(from_id, to_id, link_type, evidence?)` | Stages a link for human review (writes to `.scry/links.proposed.jsonl`) |
| `accept_link(proposed_id)` | Promotes proposed → committed link |
| `repo_summary()` | One-shot orientation: file tree, classified docs, top symbols, drift score |
| `reindex(scope?)` | Force re-extraction (default is incremental on file change) |

CLI surface mirrors the MCP tools 1:1.

---

## 9. CLI surface (initial set)

| Command | What it does |
|---|---|
| `scry init` | Wizard: choose embedding provider, write `.scry/config.yaml`, register MCP entry in `.mcp.json` / `~/.claude.json` / `~/.cursor/mcp.json` |
| `scry index [--force]` | Build/refresh the vector store and AST cache |
| `scry watch` | Daemon-free: sit on a file watcher; reindex on change |
| `scry check [--format json\|md] [--ci]` | Drift score; `--ci` exits non-zero on any non-`fresh` link |
| `scry search "<query>" [--top-k N] [--type section\|code]` | Same as MCP `search`, prints to stdout |
| `scry link <from> <to> --type <link_type> [--evidence "..."]` | Author a link from the CLI |
| `scry mcp` | Run the stdio MCP server (no daemon required) |

---

## 10. Process model

**Single, short-lived process per agent session.** No daemon (vs scrybe).
This is deliberate:
- Per-repo `.scry/` cache is durable; the index doesn't need a long-lived
  process to stay warm
- File watcher (`scry watch`) is opt-in for users who want hot-reindex
- CI usage just runs `scry index && scry check` — no daemon to wrangle

When `scry mcp` starts:
1. Load `.scry/config.yaml`
2. Verify `.scry/vectors.db` exists; if not, suggest `scry index`
3. Open vector store read-only (or read-write if file watcher is on)
4. Serve MCP tools over stdio

When the agent disconnects, the process exits. No state to persist
beyond what the disk already holds.

---

## 11. Tech stack

| Concern | Choice | Why |
|---|---|---|
| Language | **Python 3.11+** | Tree-sitter Python bindings are the most polyglot, fastembed is Python-first, brain-vault familiarity |
| Distribution | `pip install scry-cli` (`scry` binary) | Single `uv tool install` works on Win/Mac/Linux |
| MCP framework | FastMCP | Same as brain-vault; battle-tested |
| Markdown parser | `markdown-it-py` + custom frontmatter handler | CommonMark-compliant, AST gives us heading boundaries cleanly |
| Code parsing | `tree-sitter-language-pack` | One wheel, ~30 langs cross-platform |
| Embeddings (default) | `fastembed` with `BAAI/bge-small-en-v1.5` | Local ONNX, ~30 MB, no API key |
| Embedding providers (opt-in) | LiteLLM-style abstraction (OpenAI/Voyage/Cohere/local) | Same pattern brain-vault uses |
| Vector store | `sqlite-vec` extension | Single file, no server, mmap-fast at our scale |
| Keyword/BM25 | SQLite FTS5 | In stdlib; combine with vectors via RRF fusion |
| Hashing | SHA-256 over canonicalized content | Deterministic, fast |
| Tests | `pytest` | Standard |

**Total LOC target:** ~3,500 lines core for v0.1.

---

## 12. Roadmap

### v0.1 — Read-only retrieval (MVP)
- Anchor extraction (section, code_in_doc, code)
- Two-tier embedding with sub-chunks
- Hybrid retrieval (vector + RRF + graph traversal)
- MCP server + CLI for `init`, `index`, `search`, `mcp`
- Basic `links.jsonl` storage, no curation tooling beyond manual edit
- Tree-sitter for Python + TypeScript + Zig (covers hailstorm)

### v0.2 — Drift detection
- `scry check` deterministic drift scoring
- `scry watch` for hot-reindex
- Per-link `drift_status` in MCP responses
- Broken-anchor detection (`broken-source`, `broken-target`)

### v0.3 — Link curation
- `scry link` CLI for authoring links
- `propose_link` MCP tool for agent-suggested links (always staged, never auto-committed)
- `scry suggest-links` AI-augmented batch suggestions (opt-in)

### v0.4 — Reconciliation loop
- `scry reconcile <link_id>` AI-assisted patch proposal
- Embedding-distance-based `semantic-drift` detection for `mirrors` links
- CI integration recipes (GitHub Actions, pre-commit)

### v0.5+ — Polyglot expansion + ecosystem
- More tree-sitter languages with thoughtful symbol-kind defaults
- Reverse-link queries (`get_callers`, `get_subclasses`) leveraging
  the AST graph already built
- Optional in-process daemon mode for very large repos

---

## 13. Open questions (to revisit during implementation)

1. **Code-in-doc detection thresholds** — when is a fenced code block
   substantial enough to be its own anchor? Default: ≥ 5 lines OR an
   explicit language tag with ≥ 1 named declaration.
2. **Symbol address stability across refactors** — moving a method
   between classes changes its address. Should we track move history,
   or surface as `broken-target` and let the user re-link?
3. **Multi-file anchors** — some specs describe a "module" or
   "subsystem" that has no single AST node. Open question whether to
   add an `aggregate` anchor type or use multiple links instead.
4. **Embedding provider migration** — switching models means re-embedding
   the whole repo. Is there value in storing multiple embedding columns?
   Probably not for v0.1.
5. **Branch-awareness** — scrybe has it; do we need it? Initial answer:
   no, scry runs per-checkout. Revisit if user feedback says otherwise.

---

## 14. What scry deliberately is not (revisited)

| Feature | Why not |
|---|---|
| Web UI / docs portal | Agents are the consumer; humans use the CLI for hygiene |
| Chat / LLM responses | The agent harness owns that |
| Multi-project registry | One repo, one `.scry/` directory, one process per session |
| Branch-aware indexing | Switch branches → switch checkout → switch index |
| Auto-fixing drift | Always human-in-the-loop; agents propose, humans accept |
| Spec authoring tools | Devs write markdown however they want; scry just indexes |

---

*Last updated: 2026-05-06*
*Status: design complete, ready for implementation kickoff*
