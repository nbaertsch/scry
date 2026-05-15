# scry

> *Scry into your codebase.*

A local-first MCP server that links spec/doc text to AST symbols and detects
drift between them.

**Status: v0.0.1 — fully functional, all 6 implementation waves shipped
(W0a–W6e).** See [DESIGN.md](DESIGN.md) for the full architecture.

## What scry does

Most coding agents grep. Scry gives them a typed graph between your spec
sections, your code symbols (extracted via tree-sitter), and the code
blocks embedded inside your specs. Hybrid retrieval combines vector
cosine similarity (sqlite-vec) with BM25 keyword search (FTS5) and a
typed link graph across those anchors. When a spec and its implementing
code drift apart, scry surfaces it.

---

## Quick start (zero install — recommended)

Run scry in any git repo using [`uvx`](https://docs.astral.sh/uv/guides/tools/)
— no clone, no virtualenv, no PATH setup:

```bash
cd your-project/

# One-time setup
uvx --from "scry-cli @ git+https://github.com/nbaertsch/scry" scry init
uvx --from "scry-cli @ git+https://github.com/nbaertsch/scry" scry index

# Search
uvx --from "scry-cli @ git+https://github.com/nbaertsch/scry" scry search "authentication"
```

**Tip — create a shell alias** to avoid repeating the `--from` flag:

```bash
# bash / zsh — add to ~/.bashrc or ~/.zshrc
alias scry='uvx --from "scry-cli @ git+https://github.com/nbaertsch/scry" scry'

# PowerShell — add to $PROFILE
function scry { uvx --from "scry-cli @ git+https://github.com/nbaertsch/scry" scry @args }
```

Then use `scry` as a bare command anywhere:

```bash
scry init
scry index
scry search "your query"
scry check
```

## Alternative: install globally with `uv tool`

```bash
uv tool install "scry-cli @ git+https://github.com/nbaertsch/scry"
scry --help   # now on PATH permanently
```

## Alternative: clone + dev install

```bash
git clone https://github.com/nbaertsch/scry
cd scry
uv sync
uv run scry --help
```

---

## Core workflow

```bash
# 1. Build (or refresh) the local index
scry index

# 2. Link a spec section to its implementing code
scry link "docs/auth.md::authentication" "src/auth.py:login" --type implements

# 3. Check for drift between linked spec and code
scry check

# 4. Promote overlay links to the shared baseline
scry commit-links
```

Run `scry COMMAND --help` for full options on each command.

## MCP server setup

Add scry as an MCP server in your editor (Claude Code, Copilot, Cursor, etc.):

```jsonc
// .mcp.json (in your project root)
{
  "mcpServers": {
    "scry": {
      "command": "uvx",
      "args": ["--from", "scry-cli @ git+https://github.com/nbaertsch/scry", "scry", "mcp"]
    }
  }
}
```

Or if you installed globally via `uv tool install`:

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

(`scry init` prints these snippets for you.)

## Capabilities

* **Anchor extraction** — markdown sections, code symbols (Python, TypeScript,
  Go, Rust, Zig, Jest/Mocha test constructs), and code blocks inside markdown
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
* **MCP server** — 15 tools: `search`, `get_anchor`, `get_links`,
  `find_drift`, `propose_link`, `accept_link`, `commit_links`, `unlink`,
  `status`, `repo_summary`, `reindex`, `get_callers`, `get_subclasses`,
  `suggest_links_candidates`, `apply_link_suggestions`
* **Leader / follower coordination** via Unix socket / Windows named
  pipe IPC; multiple agent harnesses can share one indexed view of a
  repo
* **`scry watch`** — file-watcher hot-reindex coordinated through the
  leader
* **AI-assisted curation** (opt-in, requires LLM) — `scry suggest-links`
  and `scry reconcile <link_id>` via OpenAI / Anthropic / Ollama / LiteLLM

## License

MIT
