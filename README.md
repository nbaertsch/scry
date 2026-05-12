# scry

> *Scry into your codebase.*

A local-first MCP server that links spec/doc text to AST symbols and detects
drift between them.

**Status: v0.0.1 — fully functional, all 6 implementation waves shipped
(W0a–W6e).** Dogfooded on this repo and surfaced through a 5-persona
swarm user-testing pass.  See [DESIGN.md](DESIGN.md) for the full
architecture.

## What scry does

Most coding agents grep. Scry gives them a typed graph between your spec
sections, your code symbols (extracted via tree-sitter), and the code
blocks embedded inside your specs. Hybrid retrieval combines vector
cosine similarity (sqlite-vec) with BM25 keyword search (FTS5) and a
typed link graph across those anchors. When a spec and its implementing
code drift apart, scry surfaces it.

## Install

> All commands below use `uv run scry`. If you'd prefer bare `scry` invocations,
> install with `pip install -e .` instead (adds the `scry` binary to your PATH).

```bash
git clone https://github.com/nbaertsch/scry
cd scry
uv sync          # installs scry + tree-sitter + sqlite-vec + fastembed
uv run scry --help
```

## Quick start

```bash
# In the repo you want to track:
uv run scry init          # writes .scry/config.yaml + .gitignore + .gitattributes
uv run scry index         # builds vectors.db (incremental on subsequent runs)
uv run scry search "your query"
uv run scry doctor        # health check
```

## Core workflow

```bash
# 1. Build (or refresh) the local index
uv run scry index

# 2. Link a spec section to its implementing code
uv run scry link "docs/auth.md::## Authentication" "src/auth.py:login" --type implements

# 3. Check for drift between linked spec and code
uv run scry check

# 4. Promote overlay links to the shared baseline
uv run scry commit-links
```

Run `uv run scry COMMAND --help` for full options on each command.

To enable the MCP server in your editor (Claude Code, Cursor, OpenCode):

```jsonc
{
  "mcpServers": {
    "scry": {
      "command": "scry",
      "args": ["mcp"]
    }
  }
}
```

(`uv run scry init` prints this snippet for you.)

## Capabilities

* **Anchor extraction** — markdown sections, code symbols (Python, TypeScript,
  Go, Rust, Zig), and code blocks embedded inside markdown
* **Hybrid retrieval** — vector cosine + BM25 fused via RRF, scoped by anchor
  type and link traversal
* **Typed links** — `mirrors`, `implements`, `tests`, `examples`,
  `derives-from`, `references` between any anchor pair
* **Drift detection** — `fresh` / `code-changed` / `spec-changed` /
  `both-changed` / `drift-unknown` / `merge-conflict`, plus
  `semantic_drift` (cosine on `mirrors` links)
* **Transitive code drift** via LSP `callHierarchy/outgoingCalls` —
  optional, requires pyright / typescript-language-server / gopls /
  rust-analyzer / zls on PATH
* **MCP server** — 12 tools: `search`, `get_anchor`, `get_links`,
  `find_drift`, `propose_link`, `accept_link`, `commit_links`, `status`,
  `repo_summary`, `reindex`, `get_callers`, `get_subclasses`
* **Leader / follower coordination** via Unix socket / Windows named
  pipe IPC; multiple agent harnesses can share one indexed view of a
  repo
* **`scry watch`** — file-watcher hot-reindex coordinated through the
  leader
* **AI-assisted curation** (opt-in, requires LLM) — `scry suggest-links`
  and `scry reconcile <link_id>` via OpenAI / Anthropic / Ollama / LiteLLM

## Roadmap

* [x] **Wave 0–6** — implementation complete (see commit history)
* [x] **Dogfooding** — caught & fixed 10 bugs running scry on itself
* [x] **Swarm user-testing** — caught & fixed 20 bugs across 5 simulated
  user personas (first-time, drift workflow, MCP integration,
  multi-process, polyglot)
* [ ] Polish: surface a few remaining MEDIUM/LOW UX gaps tracked in the
  bug-tracker
* [ ] Distribute via PyPI

## License

MIT
