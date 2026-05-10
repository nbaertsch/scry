"""SR2 — Concurrency Adversary experiments for the scry project.

Run with: uv run python sr2_experiments.py
"""
from __future__ import annotations

import asyncio
import ctypes
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

# Make src/scry importable
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

IS_WINDOWS = sys.platform == "win32"

RESULTS: dict[str, dict] = {}


# ─── Experiment 2a: IPC idempotency TOCTOU ───────────────────────────────────

async def _exp2a_idem_toctou() -> dict:
    """
    Two concurrent asyncio tasks send the SAME idempotency token to
    _run_dispatch_logic.  Per §6.5 / §10.3 the handler MUST run exactly once.
    
    The bug: between cache.get(token) [miss] and cache.put(token, resp),
    the handler awaits — yielding control.  A second coroutine can sneak in,
    also see a cache miss, and also execute the handler.
    """
    from scry.process.ipc import (
        IPCRequest,
        IPCResponse,
        _IdempotencyCache,
        _run_dispatch_logic,
    )
    from scry.models import IPCConfig

    call_log: list[float] = []

    async def slow_handler(req: IPCRequest) -> IPCResponse:
        call_log.append(time.monotonic())
        await asyncio.sleep(0.05)   # yields — lets the second task enter
        return IPCResponse(request_id=req.request_id, ok=True, result=len(call_log))

    cache = _IdempotencyCache(maxsize=1000)
    config = IPCConfig()
    token = "tok_SR2RACE0001AAAA"

    req1 = IPCRequest(request_id=1, op="propose_link", args={}, idempotency_token=token)
    req2 = IPCRequest(request_id=2, op="propose_link", args={}, idempotency_token=token)

    r1, r2 = await asyncio.gather(
        _run_dispatch_logic(req1, slow_handler, cache, config),
        _run_dispatch_logic(req2, slow_handler, cache, config),
    )

    handler_ran = len(call_log)
    race_detected = handler_ran > 1

    return {
        "experiment": "2a — IPC _run_dispatch_logic idempotency TOCTOU",
        "handler_ran": handler_ran,
        "expected": 1,
        "r1_result": r1.result,
        "r2_result": r2.result,
        "race_detected": race_detected,
        "verdict": "FAIL" if race_detected else "PASS",
    }


# ─── Experiment 2b: _leader_idem_cache in MCPServer TOCTOU ──────────────────

async def _exp2b_leader_idem_cache_toctou() -> dict:
    """
    Two concurrent calls to MCPServer._dispatch with the SAME idempotency token.
    Per UT3-4 comment in server.py the cache is meant to deduplicate these.
    
    The bug: between the cache.get() check and the cache.put() store,
    `await handler(ctx, **args)` yields — a second coroutine can see a
    cache miss and also execute the handler, creating a duplicate write.
    """
    import asyncio

    # We directly test the _leader_idem_cache pattern extracted from MCPServer._dispatch
    leader_idem_cache: dict = {}
    WRITE_OPS = frozenset({"propose_link", "accept_link", "commit_links", "reindex"})
    handler_call_count = 0

    async def fake_handler(**kwargs):
        nonlocal handler_call_count
        handler_call_count += 1
        await asyncio.sleep(0.05)   # yields
        return {"link_id": f"lnk_{handler_call_count:04d}"}

    async def dispatch_like_server(op: str, args: dict):
        token_arg = args.get("idempotency_token")
        if op in WRITE_OPS and token_arg:
            cache_key = (op, token_arg)
            cached = leader_idem_cache.get(cache_key)
            if cached is not None:
                return cached

        # --- AWAIT HERE --- second coroutine can enter above check
        result = await fake_handler(**args)

        if op in WRITE_OPS and token_arg:
            cache_key = (op, token_arg)
            leader_idem_cache[cache_key] = result
            if len(leader_idem_cache) > 10_000:
                for key in list(leader_idem_cache)[:100]:
                    del leader_idem_cache[key]
        return result

    token = "tok_SR2RACE0002BBBB"
    r1, r2 = await asyncio.gather(
        dispatch_like_server("propose_link", {"idempotency_token": token}),
        dispatch_like_server("propose_link", {"idempotency_token": token}),
    )

    race_detected = handler_call_count > 1

    return {
        "experiment": "2b — MCPServer._leader_idem_cache TOCTOU",
        "handler_ran": handler_call_count,
        "expected": 1,
        "r1_link_id": r1.get("link_id"),
        "r2_link_id": r2.get("link_id"),
        "same_result": r1 == r2,
        "race_detected": race_detected,
        "verdict": "FAIL" if race_detected else "PASS",
    }


# ─── Experiment 3: SQLite WAL contention (concurrent index) ─────────────────

def _exp3_sqlite_wal_contention() -> dict:
    """
    4 concurrent subprocesses all try to run `scry index` against the same
    .scry/index.db.  The advisory write lock should serialize them; test whether
    any concurrent writes sneak through or if we get a clean "lock held" error.
    
    We run scry via subprocess to get true multi-process behaviour.
    We use a minimal fake repo with no real content so index is near-instant.
    """
    with tempfile.TemporaryDirectory(prefix="sr2_exp3_", ignore_cleanup_errors=True) as tmpdir:
        repo = Path(tmpdir)
        scry_dir = repo / ".scry"
        scry_dir.mkdir()
        overlays_dir = scry_dir / "overlays"
        overlays_dir.mkdir()

        # Minimal config
        (scry_dir / "config.yaml").write_text(
            "include:\n  - '**/*.md'\nexclude:\n  - .scry/**\nembeddings:\n  provider: local\n  model: BAAI/bge-small-en-v1.5\n  dimensions: 384\n"
        )
        # A tiny spec file
        (repo / "spec.md").write_text("# Spec\nSome content.\n")

        # Init git repo
        subprocess.run(["git", "init", "-q", str(repo)], capture_output=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@test.com"], capture_output=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], capture_output=True)
        subprocess.run(["git", "-C", str(repo), "add", "."], capture_output=True)
        subprocess.run(
            ["git", "-C", str(repo), "commit", "-m", "init", "-q"],
            capture_output=True,
        )

        python = sys.executable
        script = f"""
import sys, os
sys.path.insert(0, r'{str(_SRC)}')
import asyncio
from pathlib import Path
from scry.config import load_config
from scry.embed import make_embedder
from scry.git_context import GitContextProvider
from scry.index import Indexer
from scry.store.db import ScryDB

repo = Path(r'{str(repo)}')
cfg = load_config(repo)
db = ScryDB(repo)
git_ctx = GitContextProvider.from_config(repo, cfg.index)
embedder = make_embedder(cfg.embeddings)
indexer = Indexer(repo_root=repo, config=cfg, db=db, embedder=embedder, git_context=git_ctx)
asyncio.run(indexer.index_async(force=True))
print("index_done", flush=True)
"""

        N = 4
        procs = []
        t0 = time.monotonic()
        for _ in range(N):
            p = subprocess.Popen(
                [python, "-c", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(repo),
            )
            procs.append(p)

        results = []
        for p in procs:
            stdout, stderr = p.communicate(timeout=60)
            results.append({
                "rc": p.returncode,
                "stdout": stdout.decode(errors="replace").strip(),
                "stderr": stderr.decode(errors="replace")[-500:].strip(),
            })

        elapsed = time.monotonic() - t0
        done = [r for r in results if r["rc"] == 0 and "index_done" in r["stdout"]]
        errors = [r for r in results if r["rc"] != 0]
        lock_errors = [r for r in errors if "lock" in r["stderr"].lower()]

        return {
            "experiment": "3 — SQLite WAL concurrent index",
            "processes": N,
            "elapsed_s": round(elapsed, 2),
            "succeeded": len(done),
            "failed": len(errors),
            "lock_errors": len(lock_errors),
            "error_samples": [r["stderr"][:300] for r in errors[:2]],
            "verdict": "PASS" if len(done) >= 1 else "FAIL",
        }


# ─── Experiment 4: IPC pipe overflow ────────────────────────────────────────

async def _exp4_ipc_overflow() -> dict:
    """
    Send a 10 MB payload as a single IPC request to the leader.
    The framing layer should reject it with an 'oversized' error, not hang.
    On Windows we use named pipes; on Unix we use the Unix socket path.
    """
    if IS_WINDOWS:
        return {
            "experiment": "4 — IPC pipe overflow (Windows)",
            "verdict": "SKIP",
            "note": "Windows IPCClient.call raises NotImplementedError (deferred W6). "
                    "Test the server-side _WinPipeIO size guard via direct write.",
        }
    # Unix path
    with tempfile.TemporaryDirectory(prefix="sr2_exp4_") as tmpdir:
        repo = Path(tmpdir)
        (repo / ".scry").mkdir()

        from scry.process.ipc import IPCServer, IPCRequest, IPCResponse, parse_endpoint_uri, IPCClient

        async def _echo(req: IPCRequest) -> IPCResponse:
            return IPCResponse(request_id=req.request_id, ok=True, result="ok")

        srv = IPCServer(repo, handler=_echo)
        await srv.start()

        spec = parse_endpoint_uri(srv.endpoint_uri, repo)
        big_payload = "x" * (11 * 1024 * 1024)  # 11 MB value
        msg = json.dumps({"id": 1, "op": "search", "args": {"query": big_payload}, "protocol_version": 1}).encode() + b"\n"

        try:
            reader, writer = await asyncio.open_unix_connection(
                spec.address,
                limit=30 * 1024 * 1024,
            )
            writer.write(msg)
            await writer.drain()

            try:
                line = await asyncio.wait_for(reader.readline(), timeout=5.0)
                if line:
                    resp = json.loads(line)
                    rejected = not resp.get("ok", True) or resp.get("error_type") == "oversized"
                else:
                    rejected = True  # connection closed = rejection
            except (asyncio.TimeoutError, ConnectionError):
                rejected = False
            finally:
                writer.close()

            verdict = "PASS" if rejected else "FAIL"
            return {
                "experiment": "4 — IPC pipe overflow (Unix)",
                "payload_mb": round(len(msg) / 1_048_576, 1),
                "rejected": rejected,
                "verdict": verdict,
            }
        finally:
            await srv.stop()


# ─── Experiment 5: Heartbeat / squat detection ──────────────────────────────

def _exp5_squat_detection() -> dict:
    """
    Start a leader process that holds the lock, then hard-kill it (simulate kill -9
    / TerminateProcess).  Verify that a new leader can take over.
    """
    tmpdir_obj = tempfile.TemporaryDirectory(prefix="sr2_exp5_", ignore_cleanup_errors=True)
    tmpdir = tmpdir_obj.name
    try:
        repo = Path(tmpdir)
        (repo / ".scry").mkdir()

        python = sys.executable
        leader_script = f"""
import sys, os, time
sys.path.insert(0, r'{str(_SRC)}')
from pathlib import Path
from scry.process.leader import LeaderLock

repo = Path(r'{str(repo)}')
lock = LeaderLock.try_acquire(repo)
if lock is None:
    print('LOCK_FAILED', flush=True)
    sys.exit(1)
# Write placeholder metadata so followers can see endpoint
lock.write_metadata(endpoint_uri='pipe:scry-testdeadbeef0000', scry_version='0.0.1')
print('LEADER_READY', flush=True)
time.sleep(30)   # will be killed before this
"""
        # Start leader in subprocess
        p_leader = subprocess.Popen(
            [python, "-c", leader_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Wait for it to acquire the lock
        line = p_leader.stdout.readline()
        if b"LEADER_READY" not in line:
            p_leader.kill()
            p_leader.wait()
            return {
                "experiment": "5 — Squat detection",
                "verdict": "SKIP",
                "note": f"Leader did not start: {line!r}",
            }

        # Verify lock file exists
        lock_file = repo / ".scry" / "leader.lock"
        lock_contents = lock_file.read_text() if lock_file.exists() else ""

        # Hard-kill the leader (TerminateProcess on Windows)
        if IS_WINDOWS:
            ctypes.windll.kernel32.TerminateProcess(int(p_leader._handle), 1)
        else:
            import signal
            os.kill(p_leader.pid, signal.SIGKILL)

        p_leader.wait(timeout=3)
        time.sleep(0.5)  # Let OS release the lock

        # Now try to acquire the lock as a new leader
        from scry.process.leader import LeaderLock, read_leader_metadata_if_present

        t0 = time.monotonic()
        new_lock = None
        for attempt in range(20):
            new_lock = LeaderLock.try_acquire(repo)
            if new_lock is not None:
                break
            time.sleep(0.1)
        elapsed = time.monotonic() - t0

        takeover_ok = new_lock is not None
        if new_lock:
            new_lock.write_metadata(endpoint_uri="pipe:scry-takeover000000", scry_version="0.0.1")
            meta = read_leader_metadata_if_present(repo)
            new_lock.release()
        else:
            meta = None

        return {
            "experiment": "5 — Squat detection (kill-9 / TerminateProcess)",
            "leader_pid": p_leader.pid,
            "old_lock_contents": lock_contents[:200],
            "takeover_ok": takeover_ok,
            "takeover_elapsed_s": round(elapsed, 3),
            "new_metadata": meta.__dict__ if meta else None,
            "verdict": "PASS" if takeover_ok else "FAIL",
        }
    finally:
        tmpdir_obj.cleanup()


# ─── Experiment 1: Leader handoff race ─────────────────────────────────────

def _exp1_leader_handoff_race() -> dict:
    """
    Spawn a leader + N followers using subprocess.
    Kill the leader mid-write (using TerminateProcess / SIGKILL).
    Check if followers detect the IPC disconnect and can elect a new leader.
    """
    # Since full MCP server startup requires a real scry repo with config,
    # we test the lower-level lock handoff directly via a multi-process
    # scenario using LeaderLock primitives.
    with tempfile.TemporaryDirectory(prefix="sr2_exp1_", ignore_cleanup_errors=True) as tmpdir:
        repo = Path(tmpdir)
        (repo / ".scry").mkdir()

        python = sys.executable
        # Follower script: keeps trying to acquire the lock until it gets it
        follower_script = f"""
import sys, time
sys.path.insert(0, r'{str(_SRC)}')
from pathlib import Path
from scry.process.leader import LeaderLock

repo = Path(r'{str(repo)}')
# Try for up to 5 seconds
deadline = time.monotonic() + 5.0
while time.monotonic() < deadline:
    lock = LeaderLock.try_acquire(repo)
    if lock is not None:
        lock.write_metadata(endpoint_uri='pipe:scry-follower000000', scry_version='0.0.1')
        print('FOLLOWER_BECAME_LEADER', flush=True)
        lock.release()
        break
    time.sleep(0.05)
else:
    print('FOLLOWER_TIMEOUT', flush=True)
"""
        leader_script = f"""
import sys, time
sys.path.insert(0, r'{str(_SRC)}')
from pathlib import Path
from scry.process.leader import LeaderLock

repo = Path(r'{str(repo)}')
lock = LeaderLock.try_acquire(repo)
if lock is None:
    print('LOCK_FAILED', flush=True)
    sys.exit(1)
lock.write_metadata(endpoint_uri='pipe:scry-leader00000000', scry_version='0.0.1')
print('LEADER_READY', flush=True)
time.sleep(10)   # hold lock until killed
"""
        # Start leader
        p_leader = subprocess.Popen(
            [python, "-c", leader_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        line = p_leader.stdout.readline()
        if b"LEADER_READY" not in line:
            p_leader.kill()
            return {"experiment": "1 — Leader handoff race", "verdict": "SKIP", "note": f"Leader failed: {line!r}"}

        # Spawn 5 follower processes
        followers = [
            subprocess.Popen(
                [python, "-c", follower_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for _ in range(5)
        ]
        time.sleep(0.2)  # Let followers start their polling loops

        # Kill the leader
        if IS_WINDOWS:
            ctypes.windll.kernel32.TerminateProcess(int(p_leader._handle), 1)
        else:
            import signal
            os.kill(p_leader.pid, signal.SIGKILL)

        t0 = time.monotonic()
        follower_results = []
        for f in followers:
            out, err = f.communicate(timeout=10)
            follower_results.append(out.decode(errors="replace").strip())
        elapsed = time.monotonic() - t0

        p_leader.wait(timeout=2)

        became_leader = [r for r in follower_results if "FOLLOWER_BECAME_LEADER" in r]
        timed_out = [r for r in follower_results if "FOLLOWER_TIMEOUT" in r]

        # Exactly one follower should become the new leader (not all 5)
        exactly_one = len(became_leader) == 1

        return {
            "experiment": "1 — Leader handoff race (5 followers)",
            "became_leader_count": len(became_leader),
            "timed_out_count": len(timed_out),
            "elapsed_s": round(elapsed, 2),
            "exactly_one_new_leader": exactly_one,
            "verdict": "PASS" if exactly_one else "FAIL",
            "note": "All followers should queue; exactly one should win the lock after leader dies",
        }


# ─── Experiment 6: Watcher + leader + follower quiescence ───────────────────

def _exp6_watcher_leader_follower() -> dict:
    """
    Check whether concurrent reads from multiple processes see consistent DB state
    after the leader completes a write.  Tests SQLite WAL visibility.
    
    We use ScryDB directly (no full MCP stack) for speed.
    """
    with tempfile.TemporaryDirectory(prefix="sr2_exp6_", ignore_cleanup_errors=True) as tmpdir:
        repo = Path(tmpdir)
        (repo / ".scry").mkdir()
        (repo / ".scry" / "overlays").mkdir()

        python = sys.executable

        # Writer script: opens DB rw, inserts an anchor, commits
        writer_script = f"""
import sys, time
sys.path.insert(0, r'{str(_SRC)}')
from pathlib import Path
from scry.store.db import ScryDB
from scry.models import Anchor, AnchorType, SubChunk

repo = Path(r'{str(repo)}')
db = ScryDB(repo)
db.acquire_write_lock()
time.sleep(0.1)   # hold write lock briefly
db.upsert_anchor(Anchor(
    id='docs/test.md::section',
    type=AnchorType.SECTION,
    path='docs/test.md',
    content_hash='sha256:deadbeef',
    content_text='Test content for quiescence check',
    heading_path=['Test'],
    embedding=[0.0] * 384,
    sub_chunks=[],
    simhash=12345,
))
db.release_write_lock()
print('WRITE_DONE', flush=True)
"""
        # Reader script: opens DB ro, queries anchors N times
        reader_script = f"""
import sys, time
sys.path.insert(0, r'{str(_SRC)}')
from pathlib import Path
from scry.store.db import ScryDB

repo = Path(r'{str(repo)}')
db = ScryDB(repo, read_only=True)
# Poll until we see the anchor (up to 5s)
deadline = time.monotonic() + 5.0
while time.monotonic() < deadline:
    a = db.get_anchor('docs/test.md::section')
    if a is not None:
        print(f'SAW_ANCHOR {{a.id}}', flush=True)
        break
    time.sleep(0.05)
else:
    print('READER_TIMEOUT', flush=True)
"""
        # Init the DB
        from scry.store.db import ScryDB
        db_init = ScryDB(repo)
        del db_init  # just initializes schema

        # Start 2 reader processes first, then the writer
        readers = [
            subprocess.Popen(
                [python, "-c", reader_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for _ in range(2)
        ]
        time.sleep(0.05)
        writer = subprocess.Popen(
            [python, "-c", writer_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        writer_out, writer_err = writer.communicate(timeout=15)
        reader_results = []
        for r in readers:
            out, err = r.communicate(timeout=10)
            reader_results.append(out.decode(errors="replace").strip())

        write_ok = b"WRITE_DONE" in writer_out
        saw_anchor = sum(1 for r in reader_results if "SAW_ANCHOR" in r)
        timed_out = sum(1 for r in reader_results if "READER_TIMEOUT" in r)

        return {
            "experiment": "6 — WAL read visibility after leader write",
            "write_ok": write_ok,
            "readers_saw_anchor": saw_anchor,
            "readers_timed_out": timed_out,
            "reader_outputs": reader_results,
            "verdict": "PASS" if write_ok and saw_anchor == 2 and timed_out == 0 else "WARN",
            "note": "WAL readers should see committed writes immediately via short read transactions",
        }


# ─── Experiment 7: Static analysis — platform-specific code in shared paths ──

def _exp7_static_analysis() -> dict:
    """
    Statically analyze ipc.py for platform-specific assumptions in shared
    code paths — i.e., code that runs on both platforms but uses OS-specific APIs.
    """
    ipc_path = Path(__file__).parent / "src" / "scry" / "process" / "ipc.py"
    source = ipc_path.read_text()

    findings = []

    # Check 1: _recv_response uses asyncio.open_unix_connection path, which is
    # unavailable on Windows. But IPCClient._connect() calls open_unix_connection
    # unconditionally before checking self._spec.scheme.
    # Wait, actually call() checks scheme: if pipe → _connect_win, else → _connect.
    # That's correct. Let me check for other issues.

    # Check 2: _ConnectionHandler._verify_peer_uid calls os.getuid() (Linux only)
    # The code already guards with `platform.system() != "Linux"` → return True.
    # BUT: on macOS, SO_PEERCRED is not available, and the code returns True
    # unconditionally — relying on socket file mode 0600 as the only guard.
    # This is documented ("On macOS and other Unix systems, the mode-0600 socket
    # file is the primary access guard"). So this is a design choice, not a bug.

    # Check 3: IPCClient.call() has a comment "Windows: raises NotImplementedError"
    # but actually the Windows path IS implemented via _connect_win → _WinPipeIO.
    # Check whether the comment is stale.
    if "NotImplementedError" in source:
        idx = source.index("NotImplementedError")
        snippet = source[max(0, idx-200):idx+200]
        findings.append({
            "location": "IPCClient docstring",
            "issue": "Comment mentions NotImplementedError for Windows IPC, "
                     "but the code now implements it via _WinPipeIO (W6b). "
                     "The docstring is stale — NOT a runtime bug.",
            "snippet": snippet.strip()[:300],
            "severity": "LOW",
        })

    # Check 4: _WinPipeIO._read_buf is instance state; if readline() and write_all()
    # are called from different threads without a lock, the _read_buf could corrupt.
    # The write_lock protects writes but not reads. Is there a risk?
    # _readline_sync accesses self._read_buf (no lock)
    # _write_sync accesses self._handle (with self._write_lock)
    # These are run via asyncio.to_thread — each in a thread pool thread.
    # If two readline() calls happen concurrently (two threads reading from _read_buf)
    # there IS a data race on _read_buf.
    # BUT: IPCClient._call_lock ensures only one call() runs at a time, so
    # two concurrent readline() calls are prevented at the client side.
    # On the server side (_WinConnectionHandler.run()), readline is called
    # sequentially in the run() loop — no concurrency there either.
    # So in practice, _read_buf is only accessed by one thread at a time. SAFE.

    # Check 5: In _WinConnectionHandler.run(), SID verification happens AFTER
    # the first request line is read (required by Windows ImpersonateNamedPipeClient).
    # The first request line is decoded and processed only if SID verifies.
    # But the line has ALREADY been read into memory. If the SID check fails, we
    # just return (close the pipe without responding). The request bytes are in
    # memory but not processed. This is correct — no processing happens before auth.

    # Check 6: _start_windows yields once (await asyncio.sleep(0)) before returning.
    # This ensures the accept loop task starts before write_metadata() is called.
    # But asyncio.sleep(0) only yields ONCE — the accept loop may not have had time
    # to call ConnectNamedPipe yet. However, ConnectNamedPipe is blocking (runs in
    # thread pool), and the PIPE IS ALREADY BOUND (CreateNamedPipe was called
    # synchronously before yielding). So the pipe is ready to accept connections
    # even before ConnectNamedPipe is called. Clients can connect immediately.
    # This is subtly but correctly designed.

    # Check 7: asyncio.to_thread in _readline_sync — thread pool exhaustion.
    # If N connections are all waiting for data simultaneously, N threads are
    # blocked in ReadFile. The default thread pool size is min(32, cpu_count+4).
    # With many followers, this could exhaust the thread pool, causing deadlocks.
    if "asyncio.to_thread" in source and "PIPE_UNLIMITED_INSTANCES" in source:
        findings.append({
            "location": "ipc.py — _WinPipeIO.readline / PIPE_UNLIMITED_INSTANCES",
            "issue": "Windows: each concurrent pipe connection blocks one thread-pool "
                     "thread in ReadFile. asyncio.to_thread's default pool is "
                     "min(32, cpu+4). With many followers, thread starvation is possible. "
                     "PIPE_UNLIMITED_INSTANCES does not bound this.",
            "severity": "MEDIUM",
        })

    return {
        "experiment": "7 — Static analysis (platform-specific shared paths)",
        "findings": findings,
        "verdict": "FINDINGS" if findings else "CLEAN",
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("SR2 — Concurrency Adversary Experiments")
    print("Platform:", sys.platform)
    print("Python:", sys.version)
    print("=" * 70)

    # Run async experiments
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    print("\n[Experiment 2a] IPC _run_dispatch_logic idempotency TOCTOU...")
    r2a = loop.run_until_complete(_exp2a_idem_toctou())
    RESULTS["2a"] = r2a
    print(json.dumps(r2a, indent=2))

    print("\n[Experiment 2b] MCPServer._leader_idem_cache TOCTOU...")
    r2b = loop.run_until_complete(_exp2b_leader_idem_cache_toctou())
    RESULTS["2b"] = r2b
    print(json.dumps(r2b, indent=2))

    print("\n[Experiment 4] IPC pipe overflow...")
    r4 = loop.run_until_complete(_exp4_ipc_overflow())
    RESULTS["4"] = r4
    print(json.dumps(r4, indent=2))

    loop.close()

    print("\n[Experiment 5] Heartbeat / squat detection (kill-9)...")
    r5 = _exp5_squat_detection()
    RESULTS["5"] = r5
    print(json.dumps(r5, indent=2))

    print("\n[Experiment 1] Leader handoff race (5 followers)...")
    r1 = _exp1_leader_handoff_race()
    RESULTS["1"] = r1
    print(json.dumps(r1, indent=2))

    print("\n[Experiment 3] SQLite WAL concurrent index...")
    r3 = _exp3_sqlite_wal_contention()
    RESULTS["3"] = r3
    print(json.dumps(r3, indent=2))

    print("\n[Experiment 6] Watcher + leader + follower WAL quiescence...")
    r6 = _exp6_watcher_leader_follower()
    RESULTS["6"] = r6
    print(json.dumps(r6, indent=2))

    print("\n[Experiment 7] Static analysis (platform-specific shared paths)...")
    r7 = _exp7_static_analysis()
    RESULTS["7"] = r7
    print(json.dumps(r7, indent=2))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for k, v in RESULTS.items():
        verdict = v.get("verdict", "?")
        exp = v.get("experiment", k)
        print(f"  {verdict:8s}  {exp}")


if __name__ == "__main__":
    main()
