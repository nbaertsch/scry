# UAT-R5-3 — MCP Integration After Round-4 Fixes

**Evaluator:** UAT-R5-3 — Developer living in Claude Desktop; calls MCP tools daily  
**Persona:** I am Claude. Every interaction below is me deciding what to call next.  
**Test subject:** scry v0.0.1, dogfooded on its own repository via stdio MCP  
**Platform:** Windows 11, Python via `uv run`, subprocess Popen harness  
**Date:** 2026-05-11  
**Harness:** `uat_r5_3_harness.py` (17 MCP exchanges logged to `uat_r5_3_results.json`)  
**Prior round:** UAT-R4 (round-4 fixes: descriptions, compact responses, anchor_id consistency)

---

## 1. `tools/list` Quality

All 12 tools arrived with non-empty descriptions (22–44 words each) and both `readOnlyHint` and `destructiveHint` set — zero annotation gaps on the binary read/write axis. The write tools (`propose_link`, `accept_link`, `commit_links`, `reindex`) correctly carry `destructiveHint=True`, so as Claude I can refuse to call them in read-only contexts without parsing prose. Two gaps remain: `propose_link` accepts an `idempotency_token` parameter and its description says "supply idempotency_token to make retries safe" — but its annotation lacks `idempotentHint=True` (only `commit_links` and `reindex` carry that hint). As an LLM I would miss the idempotency affordance on `propose_link` without reading the full description. The `accept_link` description honestly notes its "Wave 2 stub / no-op" status, which is factually helpful but also signals to me that this tool is not fully implemented — I'd be uncertain whether to call it in a real workflow.

## 2. Response Shapes (Post-Compact)

Compaction correctly stripped `content_hash`, `fingerprint_simhash`, `def_line`, `def_char`, `closure_hash`, and `overview_embedding` from anchor packets. After stripping, a single anchor object has exactly 7 fields: `id`, `type`, `path`, `heading_path`, `symbol_name`, `content_text`, `transitive_hash_status`. That is clean for an LLM. **However**, the primary token cost was never the hashes — it was the text content itself. A 5-result search for "drift" returned 34,833 bytes / ~8,708 tokens. On inspection, each result carries both `content_text` (full anchor text, 929 chars in this case) and `evidence_excerpt` at the same length — both fields encode the same text for short anchors, effectively doubling the payload. Additionally, `index_state` appears per-result AND once per tool response, producing 5× repetition of the same `"fresh"` string across a single search response. These two patterns are the remaining token-efficiency stories.

## 3. Cross-Tool Flows

The `search → get_anchor → get_links` chain ran without friction. Anchor IDs returned by `search` worked directly as inputs to `get_anchor` (both `anchor_id=` and legacy `id=` parameters) and `get_links`. Each response's `index_state` field is consistent across the chain, so I never saw a contradictory stale/fresh signal mid-workflow. The `find_drift → get_links → get_anchor` reverse path also works: after committing links, `find_drift` returned 14 entries each with `from_id` and `to_id` that I could directly pass to `get_anchor` for context. One wrinkle: `get_links` on an anchor with zero active links returns `{"links": [], "index_state": "fresh"}` rather than an error — that is correct behavior but requires me to distinguish "no links" from "unknown anchor" using a prior `get_anchor` call, since `get_links` doesn't validate existence.

## 4. Idempotency UX

The cache-hit path worked perfectly. Calling `propose_link` twice with the same `idempotency_token` returned the identical `link_id` both times — the server-side LRU correctly deduplicated at the leader layer, matching IPC semantics. The response contained no `"cache_hit": true` field, so from my perspective as Claude the first and second calls looked identical. That is fine; the important invariant (same link_id) is preserved. Calling `propose_link` **without** an `idempotency_token` succeeded silently and minted a fresh `link_id` — no warning, no refusal, no annotation telling me I'm taking a non-idempotent risk. The `destructiveHint=True` is present on the tool, which gives me a vague caution signal, but nothing in the response or annotation specifically calls out "no idempotency_token means you cannot safely retry this call." A Claude agent wiring a retry loop would create duplicate links unknowingly.

## 5. Back-Compat for `id` → `anchor_id`

Fully solid. Both `get_anchor(anchor_id="...")` and `get_anchor(id="...")` returned byte-for-byte identical results across two independent calls in the harness. The docstring on the tool explicitly says `"The legacy id keyword is still accepted"`, which is visible in `tools/list` descriptions. As Claude, I can safely use either form and the server will normalize. The round-4 rename is transparent to existing integrations and correctly documented for new ones.

## 6. LSP-Unavailable Signal

This is the sharpest unresolved issue from the round-4 pass. `transitive_hash_status` is correctly preserved in compact anchor packets (value `null` when LSP was not available during indexing) — that part of the round-4 fix landed. **But `get_callers` itself is broken for this use case**: when no LSP session is available, the handler silently returns `{"callers": [], "index_state": "fresh"}` — identical to the response when a function genuinely has zero callers. As Claude, I receive an empty list and have no way to distinguish "this function is a leaf with no callers" from "LSP is unavailable so we couldn't look." A response with `"lsp_unavailable": true` or `"callers_status": "lsp_unavailable"` at the top level would let me immediately route the user to "install pyright-langserver" rather than concluding "no callers found." The `transitive_hash_status=null` on the anchor packet is the nearest proxy signal, but it requires a prior `get_anchor` call to see it, and `null` is ambiguous (could mean LSP not yet run vs LSP not installed).

## 7. Errors That an LLM Agent Could Recover From

Every error I triggered produced a clear, plain-English string in the `error.message` field. Bad `link_type` on `propose_link` would give: `"Invalid link_type 'foo': 'foo' is not a valid LinkType"` — immediately actionable (try a valid type). Bad `anchor_id` would give: `"Source anchor not found: 'bad-id'"` — I would know to call `search` first. The `status_filter` validation on `find_drift` enumerates valid values in the error: `"Valid values: ['broken-source', 'code-changed', ...]"` — I can correct and retry without human help. The one error class that would need human triage is the cold-start failure (`.scry/config.yaml` absent or `vectors.db` corrupt): those errors bubble up before any tool is callable, so `tools/list` would still work but every subsequent tool call would raise `MCPServerError("MCPServer has not been started")` — a message that requires user action, not a correctable LLM retry.

## 8. Token Efficiency

**5-result search ("how does scry handle drift"):** 34,833 bytes → ~8,708 tokens. **10-result search ("extract"):** 25,281 bytes → ~6,320 tokens (shorter code snippet anchors). The 5-result search cost MORE tokens than the 10-result because spec-section anchors carry multi-paragraph `content_text` (929 chars on the shortest example). Rough pre-compact estimate for the same 5 results would have been ~9,000+ tokens (hash fields add ~200 bytes per result × 5 = 1,000 bytes). Compaction saved maybe 250 tokens — a 3% reduction. The remaining bulk is almost entirely `content_text` + the duplicated `evidence_excerpt` field. For comparison: a typical Claude tool-use budget is 4,000–8,000 tokens for the entire context; a 5-result search already consumes 8,700. That leaves almost no headroom for the conversation history when doing a realistic multi-turn workflow.

## 9. What an LLM-Using User Would Still Complain About

1. **`get_callers` gives no signal when LSP is missing** — user asks "who calls this function?", Claude returns "no callers found," user is confused because the function is called in several places. Silent empty list is a trust-destroying bug.

2. **8,700 tokens for 5 search results** — a developer with a long conversation will hit the context window doing a 3-step search→drill→link workflow. The `content_truncated: false` flag tells me there's more to read, not that I'm already at the limit.

3. **`evidence_excerpt` duplicates `content_text`** for short anchors — the per-result payload effectively carries the text twice. Users on tight token budgets (Claude Haiku, GPT-4o-mini) will see this as wasteful.

4. **`propose_link` without `idempotency_token` creates un-retryable mutations** — a Claude agent that retries on network error will create duplicate links silently. No warning, no `idempotency_required: true` hint.

5. **`find_drift` top-level response has no `drift_coverage`** — the per-entry field says `"section-only"` but I would expect a top-level summary too (the way `repo_summary` exposes it). I'd have to read N entries just to know the coverage scope.

6. **`index_state` is repeated per-result in search** — 5 copies of `"fresh"` adds noise. One top-level `index_state` is sufficient.

7. **`accept_link` is a documented no-op stub** — as Claude, I don't know whether to call it in a workflow. Its presence in `tools/list` suggests it does something, but the description says "status persistence is currently a no-op." I'd likely skip it but also wonder if that breaks something downstream.

## 10. Three MCP-Specific UX Suggestions (Fresh Ranking)

**1. `get_callers` / `get_subclasses` must signal LSP unavailability explicitly.**  
The response `{"callers": [], "index_state": "fresh"}` is indistinguishable from "no callers found." Add a top-level `"lsp_status": "unavailable" | "ok" | "error"` field (or `"lsp_unavailable": true`) so an LLM agent can branch: if `lsp_unavailable`, surface "install pyright-langserver" guidance; otherwise, trust the empty list. This is a one-liner in the handler (`return {"callers": [], "index_state": ..., "lsp_unavailable": True}`) but eliminates the most misleading ambiguity in the whole API surface.

**2. Eliminate `evidence_excerpt` duplication from search results OR make it a strict substring.**  
`content_text` carries the full anchor text; `evidence_excerpt` currently carries the same text for anchors shorter than the excerpt limit. Either drop `evidence_excerpt` from MCP search responses (keep it for CLI) or make it a strict 200-character highlighted snippet with a `matched_at` byte offset. This alone would cut 5-result search payloads by ~40% for spec-section anchors and push the 8,700-token response below 5,200 — leaving breathing room for multi-turn workflows. Bonus: move `index_state` to the top level only (remove per-result repetition) for another ~50-token saving.

**3. Add `idempotentHint=True` to `propose_link` and emit a `"idempotency_token_required": true` warning in the response when the parameter is omitted.**  
`propose_link` is the most dangerous non-idempotent write in the API: retrying it creates silent duplicates. Adding `idempotentHint=True` to its annotation signals to MCP clients (and LLM agents) that retries are possible but require a token. A response-level warning when the token is absent (e.g., `"warning": "no idempotency_token supplied — retrying this call will create a duplicate link"`) gives Claude an explicit affordance to communicate retry risk to the user without parsing descriptions. This mirrors how Stripe's API communicates idempotency to API consumers.

---

## Verdict: **TRY-IF-IMPROVED**

**Improvement vs prior round: better**

Round-4's cleanup landed correctly: descriptions are LLM-readable, `anchor_id` consistency is solid, compact responses removed hash clutter, and `transitive_hash_status` preservation shows deliberate LLM-first thinking. The `initialize → tools/list → search → get_anchor → get_links → propose_link → commit_links → find_drift` chain ran end-to-end without a single error — that is the core happy path for a spec-drift agent and it works. The blockers are now concentrated in two areas: **token volume** (8,700 tokens for 5 results is still too high for Haiku-class agents doing multi-turn workflows) and **LSP-unavailable ambiguity in `get_callers`** (silent empty list vs. genuinely no callers is trust-destroying). Fix those two and the UX moves from "promising but cautious" to "daily-driver for MCP-native agents."
