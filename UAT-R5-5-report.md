# UAT-R5-5 — Can scry Be a Required PR Gate?

**Role:** Platform engineer owning CI/CD pipeline  
**scry version:** 0.0.1  
**Date:** 2026-05-11  
**Repo under test:** scry's own repo (dogfood)  
**Simulation baseline:** 10 committed spec↔code links (DESIGN.md → src/scry/), plus
5 new links (TEST_SPEC.md → cli.py / embed.py / store/links.py / retrieve.py / drift.py)
committed via `scry link` + `scry commit-links`.

---

## Setup Executed

```
scry index --force                        # fresh vector store from scratch
# Created TEST_SPEC.md with 5 headings
scry link TEST_SPEC.md::...::cli-check-command  src/scry/cli.py:check  --type implements
scry link TEST_SPEC.md::...::embedder-interface src/scry/embed.py:make_embedder  --type implements
scry link TEST_SPEC.md::...::link-store         src/scry/store/links.py:LinkStore --type implements
scry link TEST_SPEC.md::...::hybrid-search      src/scry/retrieve.py:hybrid_search --type implements
scry link TEST_SPEC.md::...::drift-evaluation   src/scry/drift.py:evaluate_link_drift --type implements
scry commit-links                         # promoted 20 records to baseline
git commit -m "UAT baseline"
```

PR simulations:

| PR | Change | Command | Observed Exit |
|----|--------|---------|---------------|
| A  | SCRATCH.txt (unlinked) | `--ci --since HEAD~1` | **0** ✅ |
| B  | TEST_SPEC.md spec text | `--ci --strict --since HEAD~1` | **1** ✅ (`spec_changed: 1`) |
| C  | `drift.py:evaluate_link_drift` body | `--ci --strict --since HEAD~1` | **1** ✅ (`code_changed: 4`) |
| D  | 41 unrelated test files | `--ci --strict --since HEAD~1` | **0** ✅ (`drift_score: 100.0`) |

---

## 1. Diff-aware `--since` correctness — false positives? false negatives?

`--since HEAD~1` correctly scoped every PR to only the links whose endpoint files
appeared in `git diff --name-only HEAD~1..HEAD`. PR-A (unlinked file) returned
`drift_score: null` — a well-designed null-means-pass that avoids spurious failures
when a PR legitimately touches no linked files. PR-D (41 noisy test files) returned
`drift_score: 100.0` with `total: 6`; the six links that touch test files all showed
`fresh` because the added module-level comments did not change any symbol-level
content hash, confirming scry correctly hashes at symbol granularity, not file
granularity. No false positives were observed. The one false-negative risk: if the
index is stale (unrun between commit and check), `--since` may scope correctly to the
right files but report `fresh` because content hashes haven't been updated; the
`fs_stale_warning` field surfaces this, but CI won't block on it. The fix is to always
run `scry index` before `scry check` in the pipeline.

---

## 2. Exit code reliability — 0/1/2 matched expectations?

All three codes behaved exactly as documented. Exit 0 was returned for clean states
(PR-A, PR-D, baseline), exit 1 for drift when `--strict` or `--drift-min` was active
(PR-B, PR-C), and exit 2 for operational failures (missing `vectors.db`, invalid
`--since` git ref, embedding model mismatch via `--require-fresh-embedder`). **One
critical trap**: `--ci` alone, without `--drift-min` or `--coverage-min`, **always
exits 0** — the CI gate is entirely opt-in at the threshold level. Reading the CLI
source confirms this: `if drift_min is not None and effective_drift_score < drift_min`
— if you forget `--drift-min`, the condition never fires. In practice, CI gates
should use either `--strict` (zero-tolerance) or `--ci --drift-min 100` for the same
effect; `--ci` alone is effectively a reporting mode, not a gate.

---

## 3. JSON output stability — would you parse it in a downstream Action? Any version markers?

JSON output is byte-identical across consecutive invocations on the same index state;
two back-to-back `scry check --json` calls produced `True` for PowerShell's
`ConvertFrom-Json` equality check. The schema has five top-level keys:
`drift_score` (float|null), `coverage_score` (float|null), `counts` (object with
nine named sub-keys), `drift_coverage` (string enum), and `fs_stale_warning`
(string|absent). Adding `--verbose` appends a `drifted_links` array with
`link_id`, `drift_status`, `from_id`, `to_id`, and `type` — useful for
a downstream Action that comments on a PR with the specific links that drifted.
**There is no `scry_version` field in the output.** This is a meaningful gap for
downstream consumers: if the schema changes in a future release, nothing in the
JSON payload signals that the parser needs updating. I would parse this output in a
downstream Action today, but I'd pin the scry version explicitly in the workflow
and add a schema-validation step against a committed JSON Schema file.

---

## 4. Cold-start cost in CI — how long does index take on a fresh runner?

Measured on a Windows dev box with no prior fastembed model cache: `scry index --force`
completed in **864 seconds** (~14.4 minutes) on a 90-file, 2578-anchor repo; the
dominant cost was fastembed downloading the `BAAI/bge-small-en-v1.5` model (~420 MB)
from HuggingFace at first run. With the model cached, the same full re-index took
**243 seconds** (~4 minutes). Incremental indexing of 1–2 changed files took
**3–18 seconds**. On a GitHub-hosted Linux runner (`ubuntu-latest`), network speed
is faster but CPU is slower; a realistic estimate for warm-cache incremental CI is
**20–40 seconds** for `scry index` followed by **10–20 seconds** for `scry check`.
Without any caching, a fresh runner for a mid-size repo would add **10–15 minutes**
to every PR run — completely unacceptable for a required gate.

---

## 5. Caching strategy — would `.scry/vectors.db` survive across CI runs? Should it?

Yes, you must cache `.scry/vectors.db` or the tool is unusable as a PR gate.
Cache two directories: (1) `.scry/vectors.db` (27 MB for this repo — grows with
codebase size) keyed on `scry_version + embedding_model + hash(.scry/links.jsonl)`
so cache is invalidated on model migration or link baseline changes; and (2)
`~/.cache/fastembed` (the HuggingFace model weights, ~420 MB) keyed on model name
and version. With both caches warm, CI reduces to an incremental `scry index` (~20s)
plus `scry check` (~15s). The `.scry/vectors.db` should be stored as a **read-only
artifact** from the `main` branch daily build, with PR runners pulling it as a base
and running incremental updates; this avoids every PR re-indexing independently.
**Do not commit `vectors.db` to git** — it is large, binary, and regenerable.

---

## 6. Failure modes — what failure modes would block all PRs across the company?

Three failure modes would cause company-wide PR blockage. First, **HuggingFace
network outage**: if `fastembed` cannot download the model on a cold runner and the
model cache is empty, `scry index` fails with an opaque exception rather than a
clean exit-2 message; this would block all PRs until the outage resolves or caches
are pre-warmed. Second, **SQLite write-lock timeout**: `scry index` and `scry check`
both use a file-based write lock; if a zombie lock file from a crashed job is left
behind, every subsequent run fails with `Could not acquire write lock within the
timeout` — observed during this UAT. The fix (removing the lock file) requires
manual runner intervention or a `pre-step` that clears stale locks. Third,
**links.jsonl corruption or schema migration**: if a scry update changes the link
record schema and old `links.jsonl` files fail to deserialize, every check across
every repo fails until the baseline is re-committed.

---

## 7. Operational concerns — where would scry break in a multi-runner / multi-fork CI?

Multi-runner environments have three classes of breakage. First, **concurrent index
runs**: if two runners for the same repo run `scry index` simultaneously, one will
time out waiting for the write lock — there is no distributed lock, only a local
file lock. This is safe (no corruption) but wasteful and confusing. Solution: run
`scry index` only in a dedicated "pre-check" step, not in parallel matrix jobs.
Second, **fork PRs**: fork PRs run in a different checkout with a different
`.scry/` directory; they cannot inherit the base repo's `vectors.db` cache unless
the workflow explicitly downloads it as an artifact. Without this, fork PRs pay the
full cold-start cost and may hit network failures in restricted fork environments.
Third, **Windows/Linux cross-platform**: SQLite databases are portable, but path
separators in anchor IDs (`DESIGN.md::scry-design::...` vs file system paths) use
forward slashes internally regardless of OS — this was consistent in testing, but
the `--since` diff output on Windows uses backslashes; the `_link_touches_diff`
function splits on `:` not `/`, so `from_id.split(":", 1)[0]` returns the full
platform path. If git on Windows returns `src\scry\cli.py` in the diff output but
the anchor ID stores `src/scry/cli.py`, `--since` would silently miss the link.
**This was not triggered in testing but represents a latent cross-platform bug.**

---

## 8. GitHub Actions YAML

```yaml
name: scry-drift-gate

on:
  pull_request:
    branches: [main]

jobs:
  scry-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 2          # need HEAD~1 for --since

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install scry
        run: pip install "scry-cli==0.0.1"   # pin version explicitly

      - name: Restore vectors.db cache
        id: cache-vectors
        uses: actions/cache@v4
        with:
          path: |
            .scry/vectors.db
            ~/.cache/fastembed
          key: scry-${{ hashFiles('.scry/links.jsonl') }}-${{ runner.os }}
          restore-keys: |
            scry-

      - name: Index (incremental or fresh)
        run: |
          if [ "${{ steps.cache-vectors.outputs.cache-hit }}" = "true" ]; then
            scry index          # incremental — only changed files
          else
            scry index --force  # cold start — full embed (slow, ~4 min)
          fi

      - name: Drift check (diff-aware, strict)
        run: |
          scry check \
            --ci \
            --strict \
            --ignore-lsp-error \
            --since HEAD~1 \
            --json \
            --verbose 2>&1 | tee scry-report.json

      - name: Upload drift report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: scry-drift-report
          path: scry-report.json

      - name: Comment on PR (on drift failure)
        if: failure()
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const report = JSON.parse(fs.readFileSync('scry-report.json', 'utf8'));
            const lines = (report.drifted_links || []).map(l =>
              `- \`${l.link_id}\` **${l.drift_status}**: \`${l.from_id}\` → \`${l.to_id}\``
            ).join('\n');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## ⚠️ scry drift detected\n\n${lines || '(see artifact for details)'}\n\n_Run \`scry check --verbose\` locally for more._`
            });
```

**Notes for the YAML:**
- `fetch-depth: 2` is required for `--since HEAD~1`; `--since origin/main` would need `fetch-depth: 0`
- `--ignore-lsp-error` is critical on Linux runners where `pyright-langserver` is not pre-installed
- `--strict` is required because `--ci` alone without `--drift-min` always exits 0
- Cache key includes `links.jsonl` hash so a `scry commit-links` push invalidates the cache

---

## 9. Gating verdict — REQUIRED for all PRs at your company?

**Not yet.** The feature correctness is solid — `--since` scopes accurately,
`--strict` gates on any drift, exit codes are reliable, and JSON output is stable
enough to parse. However, two operational blockers prevent making this REQUIRED
company-wide today. First, cold-start cost is 10–15 minutes on a fresh runner;
even with caching, the first run after a model upgrade or cache eviction silently
adds 15 minutes to your slowest PR. Until there is a `--no-download` flag that
fails cleanly when the model is absent (rather than downloading it mid-job),
runners with network restrictions will silently stall. Second, the `--ci` flag's
threshold-free behavior is a footgun: any team that writes `scry check --ci` without
also adding `--strict` or `--drift-min 100` gets a green check that means nothing.
This would happen across every repo at onboarding time. **Verdict: OPT-IN.**
Make it available as a non-blocking advisory check in the shared CI template, with
easy escalation to `--strict` per-repo via a `.scry/config.yaml` field or a
team-level override. Promote to REQUIRED gate after: (a) `scry check --ci` defaults
to `--strict` or requires an explicit `--drift-min`, and (b) the lock/cold-start
failure modes have a clean non-blocking exit path.

---

## 10. Three CI-Specific UX Suggestions (ranked)

### 1. Add `scry_version` to JSON output (highest priority)

Every downstream consumer — Action parsers, monitoring dashboards, compliance
tools — needs to know which schema version it is reading. The current output has
no `scry_version` field; when the schema evolves, silent parse failures are
guaranteed. Add `"scry_version": "0.0.1"` as the first key in every JSON response.
This is a one-line change with zero backward-compatibility cost and would
immediately unblock enterprise adoption of `--json` in pipelines.

### 2. Default `--ci` to `--strict` or require explicit `--drift-min`

The biggest footgun found: `--ci` alone never exits 1. New users who read "CI mode"
and write `scry check --ci` expect a gate that fails on drift. The simplest fix is
to make `--ci` imply `--strict` unless `--drift-min` is explicitly overridden. An
alternative is to print a `warning: --ci without --strict or --drift-min is a
no-op gate; all PRs will pass` message to stderr when neither is supplied. Either
approach would have caught at least 50% of the misconfiguration bugs we'd see
across a multi-team rollout.

### 3. Add `--lock-timeout-exit-code 0` flag for CI resiliency

The current lock behavior on timeout is to exit 2 (operational error). In CI,
a lock timeout on `scry index` means a runner crashed mid-job and left the lock
behind. Exiting 2 fails the PR check and blocks the developer, even though the
error is infrastructural, not content-related. A `--lock-timeout-exit-code 0` flag
(or `--ci-resilient`) that clears stale locks and continues would prevent
lock-file artifacts from blocking company-wide PRs, at the cost of slightly
looser guarantees (the stale index warning already communicates this risk via
`fs_stale_warning`).

---

## Verdict

**OPT-IN**

`scry check --since` is genuinely the killer feature for CI: it transforms a
whole-repo drift audit (impractical as a PR gate) into a scoped, fast, accurate
check that touches only files in the PR diff. The null-score-means-pass design
for zero-link PRs is elegant. The `--strict` + `--ignore-lsp-error` + `--since`
combination gives a realistic, actionable gate for repos without full LSP setup.

The feature is not yet ready to be **REQUIRED** company-wide because: cold-start
cost can silently add 15 minutes to CI, `--ci` alone is a footgun that always
exits 0, and the write-lock failure mode requires runner-level intervention. Fix
those three issues and this becomes one of the most defensible required PR gates
in the spec-driven tooling ecosystem — because it is the only one that is both
diff-aware and content-hash-deterministic.

---

*Simulation commits (local only, not pushed):*
```
c3f9dc6  test(uat-r5-5): add TEST_SPEC.md + 5 spec->code links for CI gate simulation
1111a4f  test(uat-r5-5): PR-A clean commit (unlinked file)
0d9bad6  test(uat-r5-5): PR-B spec-only edit (TEST_SPEC.md::cli-check-command changed)
e8f9f1f  test(uat-r5-5): PR-C code-only edit (evaluate_link_drift body changed)
679549c  test(uat-r5-5): PR-D noisy diff (50 unrelated test files)
```
