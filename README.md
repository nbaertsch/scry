# scry

> *Scry into your codebase.*

A local-first MCP server that links spec/doc text to AST symbols and
detects drift between them.

**Status: design phase.** See [DESIGN.md](DESIGN.md) for the full
architecture. Implementation starts in v0.1.

## Quick concept

Most coding agents grep. Scry gives them a typed graph between
your spec sections, your code symbols (extracted via tree-sitter),
and the code blocks embedded inside your specs. Hybrid retrieval
combines vector cosine similarity with graph traversal across those
links. When a spec and its implementing code drift apart, scry
surfaces it.

## Status

- [x] Design — [DESIGN.md](DESIGN.md)
- [ ] v0.1 — read-only retrieval (anchor extraction, embeddings, MCP)
- [ ] v0.2 — drift detection
- [ ] v0.3 — link curation
- [ ] v0.4 — AI-assisted reconciliation

## License

MIT (planned)
