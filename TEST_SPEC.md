# TEST_SPEC — UAT-R5-5 Simulation Spec

> This file is used ONLY for the UAT-R5-5 CI gate evaluation. It links to real scry
> implementation code and is intentionally modified during PR simulations.

---

## CLI Check Command

The `scry check` command evaluates all active links for drift and reports
scores to stdout. It must support `--ci`, `--json`, `--strict`,
`--since`, `--ignore-lsp-error`, and `--require-fresh-embedder` flags.
Exit codes are: 0 = clean, 1 = drift detected, 2 = operational error.
**CHANGED FOR PR-B SIM**: Added a new requirement — the check command must
emit a machine-readable summary even when no links are scoped by --since.

---

## Embedder Interface

The embedder subsystem provides a unified interface for turning anchor
text into float vectors. A `StubEmbedder` is available for CI
environments that should not trigger a model download. The `make_embedder`
factory function selects the concrete embedder from config.

---

## Link Store

The `LinkStore` class owns the append-only event log of link records.
It exposes `replay()` to materialise the current active link table from
an ordered sequence of upsert/delete events. The baseline layer lives in
`.scry/links.jsonl`; per-branch overlays live under `.scry/overlays/`.

---

## Hybrid Search

The `hybrid_search` function combines BM25 keyword scoring with vector
cosine similarity using Reciprocal Rank Fusion (RRF). It returns a list
of ranked anchor packets that expose both the section text and the
computed score fusion breakdown.

---

## Drift Evaluation

The `evaluate_link_drift` function takes a link record and both anchor
content hashes and returns a `DriftStatus` enum value. It is the core
primitive for diff-aware CI checks and MCP `find_drift` calls.
