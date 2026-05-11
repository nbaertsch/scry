# UAT2 — Spec Author's Take on scry

**Evaluator:** UAT2 — Tech lead, maintains DESIGN.md / RFC documents  
**Test subject:** scry v0.0.1 dogfooded on its own repository  
**Platform:** Windows 11 / PowerShell, Python 3.11.13  
**Date:** 2026-05-10  

---

## 1. The Promise vs the Reality

Scry's pitch — "make spec drift visible" — hits the right nerve for anyone who has shipped a system where DESIGN.md stopped reflecting the code by v1.1. The promise is a living link graph between spec sections and code symbols, with deterministic drift detection surfaced on every `scry check`. In practice, the core loop (index → search → link → check) works and produces a real signal: after authoring three links against DESIGN.md sections and their Python implementations, `scry check` immediately reported drift_score=100 and coverage_score=0.4%. That 0.4% is honest and actionable — it says "you've linked 10 of your 2419 code anchors." What it doesn't yet do is show me *which* spec sections are completely unlinked, or flag the fact that I've spent 152 seconds indexing a small repo without LSP. The proof-of-concept works; the ergonomics of daily use are rougher than the spec implies.

## 2. Setup Ergonomics

The `.scry/config.yaml` was pre-configured for dogfooding, which removes the `scry init` step from the critical path. That said, `scry index --force` ran for **152 seconds** (2.5 minutes) on a repo of 85 files and 2,483 anchors, emitting only two lines of output before going silent for the duration. There's no progress bar, no ETA, and no indication of whether it's embedding anchors or doing something else. The LSP warning (`No allowlisted LSP binary found for 'python'; tried: pyright-langserver, pylsp, basedpyright-langserver`) appeared immediately at startup but `scry doctor` gives no platform-specific install instructions — on Windows, "install pyright-langserver" is not a one-liner and the tool doesn't help you get there. The fact that code transitive drift (`§5.3`) is completely locked behind a working LSP means you go from "100% drift coverage promised" to "section-only drift only" the moment setup hits the first unresolved binary — which, on a fresh Windows machine, is guaranteed.

## 3. The Link Command

`scry link <from_id> <to_id> --type implements` is syntactically clean. The saving grace is that `scry search` returns exact anchor IDs as part of its output — `src/scry/retrieve.py:hybrid_search` and `DESIGN.md::scry-design::4-retrieval::4-1-hybrid-retrieval-algorithm` are legible if long — so I got all three links right on the first try by copy-pasting from search results. What tripped me up was direction: DESIGN.md §3.6 defines `implements` as canonically `code → spec`, but the seven pre-existing links in `links.jsonl` are all stored `spec → code` (e.g., `from: DESIGN.md::scry-design::5-drift-detection`, `to: src/scry/drift.py:DriftDetectionError`, `type: implements`). The design says inverse inputs "are accepted at write time but normalized to canonical form (from/to swapped)." That normalization is either not implemented or the baseline links were authored before it landed. Either way, `scry check` doesn't flag the inconsistency, so a new user can't tell if their direction is right. A `scry validate` warning on inverted `implements` links would close this gap.

## 4. The Check Command

`scry check` is the most production-ready thing in scry. The output format is clean markdown, `--format json` works, and the counts table (fresh, spec_changed, code_changed, broken_source, etc.) maps directly to CI gate logic described in the spec. Two friction points: first, `--verbose` currently outputs only "(verbose) all links fresh." — nothing more granular when links exist and are fresh. For a spec author reviewing a PR, I want verbose to show me each linked pair with its drift status, even when fresh, so I can sanity-check the coverage at a glance. Second, coverage_score=0.4% is useful as a global metric but tells me nothing about *which spec sections* are uncovered. A "most-unlinked spec sections" list in check output would be the high-value addition here.

## 5. Suggest-Links

`scry suggest-links` is non-functional without a local Ollama instance, which is not installed on this machine. What's worse, the command doesn't fail-fast: it silently iterates through every candidate pair (800+ on this repo in batches of 20), printing the same "Ollama is not reachable at http://localhost:11434" warning for every batch before eventually stopping. This is a severe UX bug — the tool detected the LLM was down on the very first batch call but continued processing for several minutes with zero useful output. Running with `--limit 5` did properly error and exit after one batch, but the unlimited default is a silent time-sink. I cannot evaluate the quality of suggestions themselves; what I can say is that the candidate-pair selection phase (embedding similarity scanning) clearly works — the candidate count of 800+ for a 2,483-anchor repo suggests the similarity threshold is finding real pairs. The architecture (semantic pre-filter → LLM classification) is the right design for signal-to-noise.

## 6. Spec Author's Missing Pieces

The single most-wanted feature I don't see: **"spec section → linked anchors" CLI browse**. There's `get_links(anchor_id)` as an MCP tool but it's absent from the CLI surface, even though DESIGN.md §8 says "CLI surface mirrors the MCP tools 1:1." After authoring a link, I have no CLI way to ask "what code is currently linked to §4.1?" without going through MCP. Second: there's no **change-aware notification** — scry detects drift after-the-fact but doesn't integrate with pre-commit hooks or git `post-commit` to flag "you just modified a file with linked anchors." Third: there's no **spec-section coverage breakdown** — the global 0.4% coverage score doesn't help me identify which sections of DESIGN.md are completely orphaned from any code links. Fourth: `scry reconcile` requires a running LLM and, in this case, produced "No drifted links to reconcile" — it's deferred to Wave 5 and the LLM-dependency makes it effectively a second-class citizen on any machine without cloud API credentials configured.

## 7. Vocabulary

The six canonical link types (`implements`, `tests`, `examples`, `mirrors`, `derives-from`, `references`) cover the most common spec↔code relationships and the `mirrors` type with its separate `semantic_drift` boolean is genuinely clever — distinguishing "structurally equivalent" from "semantically drifted" is exactly the right cut for contract-style code blocks in specs. What's missing: **`supersedes`** (spec section A replaces section B — useful for tracking spec evolution without deleting the old section), **`validates`** (config parsing or schema enforcement code that enforces a spec constraint — distinct from `implements`), and **`deferred`** (spec section X has no implementation yet, intentionally — a first-class "acknowledged gap" link type would let `scry check` distinguish "not yet linked" from "never will be linked"). The vocabulary is extensible per-repo (`.scry/config.yaml`), which is good, but the design doesn't document the extension mechanism clearly enough for new users to discover it.

## 8. Doc-Driven Workflow

Scry's mental model is nominally bidirectional — `scry search` returns both spec sections and code anchors for any query — but the authoring workflow is de facto **code-first**: you run `scry link <code-anchor> <spec-anchor> --type implements` with the code as `from`. A spec author thinks "what implements §4.1?" not "which spec section does `hybrid_search` implement?" The `specifies` inverse (rendered at query time) is promised in the design but not surfaced in any CLI command. If I could run `scry link DESIGN.md::...::4-1 <blank> --type specifies` and have scry ask me to fill in the code anchor with a fuzzy search prompt, that would be doc-first. Right now it's doc-searchable (you can find spec sections via `scry search`) but not doc-anchored as a link source. The workflow implicitly assumes the user knows which function to look for — which is a code-reader's privilege, not a spec-author's.

## 9. Would You Use This on Your Specs at Work?

Honest answer: **not yet, but close.** The core `search → link → check` loop is sound enough to produce real value. If `scry check` ran in CI with `--drift-min 90 --coverage-min 50` against a well-linked DESIGN.md, I would trust it to catch regressions. The two blockers for production use on my team are: (a) the 2.5-minute index time on a small repo is a developer-experience problem at any frequency beyond weekly, and (b) `suggest-links` being gated on a local LLM setup means the onboarding story for new contributors is "install Ollama, pull a 4GB model, then scry suggest-links will help you find what to link" — that's too much friction for optional tooling. I'd trial it on a repo where I already maintain a living DESIGN.md with clear section boundaries, run `scry index` once, manually link 15–20 key pairs, and use `scry check` in CI as the gate. That narrower use case works today.

## 10. Top 3 Specific UX Suggestions

**1. `suggest-links` must fail-fast when LLM is unreachable.**  
Currently the command iterates through all 800+ candidate pairs (2–5 minutes) printing identical "Ollama not reachable" warnings before exiting. The fix: probe the LLM on batch index 0 failure — if the error is a connectivity error (not a rate limit), abort immediately with a single clear message and non-zero exit code. The `--limit 5` flag does cause early exit, but the unlimited default must not silently burn minutes doing useless work.

**2. Add `scry get-links <anchor-id>` and `scry get-anchor <anchor-id>` as CLI commands.**  
DESIGN.md §8 states "CLI surface mirrors the MCP tools 1:1", but `get_anchor` and `get_links` are MCP-only today. For a spec author, `scry get-links DESIGN.md::scry-design::4-retrieval::4-1-hybrid-retrieval-algorithm` (showing all code anchors linked to that section with drift status) is the single most-useful query after authoring links. Without it, you either invoke MCP directly or use `scry check --verbose` which currently shows nothing per-link even when links exist. This is a straightforward CLI wrapper around the existing MCP handler.

**3. Add a spec-section coverage breakdown to `scry check`.**  
The global `coverage_score: 0.4%` is useful for CI gating but doesn't help spec authors prioritize which sections to link first. A `scry check --uncovered` flag that lists spec-section anchors with zero incoming or outgoing links (sorted by section length, as a proxy for importance) would let a spec author triage the gap in one command. Alternatively, add a `by_section` block to the existing check output that lists each spec file's per-section link count — no LLM required, purely deterministic, high signal for the doc-first workflow.

---

## Verdict: **TRY-IF-IMPROVED**

The core primitives are the right ones. Hybrid search works, `scry check` is genuinely useful, and the append-only link graph with baseline/overlay split is a thoughtful design. The blockers are operational: slow indexing, no LSP without manual setup on Windows, a `suggest-links` that burns minutes on LLM errors, and a CLI surface that's missing the `get-links` command spec authors will reach for first. Fix those three and this becomes a daily-driver for any team that writes living specs.
