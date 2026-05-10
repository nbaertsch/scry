"""LSP subprocess lifecycle manager: spawn, initialize, shutdown, allowlist.

Implements the W3a LSP infrastructure:

* :data:`LSP_ALLOWLIST`   — hardcoded per DESIGN.md §6.2
* :class:`LSPLaunchSpec`  — immutable launch parameters for one language
* :class:`LSPSession`     — one live LSP subprocess + JSON-RPC wires
* :class:`LSPManager`     — lazy-spawn session pool; one session per language

Security
--------
Per DESIGN.md §6.2, only binaries whose *bare name* is listed in
:data:`LSP_ALLOWLIST` may be launched without
``allow_untrusted=True``.  A ``command:`` override in
``.scry/config.yaml`` that is not explicitly opted into via
``--allow-untrusted-lsp-config`` raises :class:`LSPAllowlistViolation`.

Cross-platform spawning (DESIGN.md §10.5)
------------------------------------------
npm-installed tools on Windows (e.g. ``typescript-language-server``) are
``.cmd`` shims that ``CreateProcess`` cannot execute directly.  When the
resolved binary path ends in ``.cmd`` or ``.bat`` and the runtime is
Windows, the manager spawns via ``cmd.exe /C <path> <args>``.

References
----------
DESIGN.md §5.3  — transitive drift via callHierarchy
DESIGN.md §6.2  — LSP binary allowlist
DESIGN.md §10.5 — Windows .exe/.cmd shim spawning
DESIGN.md §11   — JSON-RPC over stdio
"""

from __future__ import annotations

import asyncio
import asyncio.subprocess
import contextlib
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self

from scry.lsp.adapters import get_adapter
from scry.lsp.proto import LSPMessage, LSPProtocolError, LSPStreamReader, LSPStreamWriter
from scry.models import CodeAnchorsConfig

logger = logging.getLogger(__name__)

__all__ = [
    "LSP_ALLOWLIST",
    "LSPAllowlistViolation",
    "LSPInitializeError",
    "LSPLaunchError",
    "LSPLaunchSpec",
    "LSPManager",
    "LSPSession",
]

# ─── Allowlist ────────────────────────────────────────────────────────

# Hardcoded per DESIGN.md §6.2 (post-v3.1).
# Values are bare binary names resolved via PATH.
# W3c will handle Windows .exe/.cmd suffix lookup per adapter.
LSP_ALLOWLIST: dict[str, list[str]] = {
    "python": ["pyright-langserver", "pylsp", "basedpyright-langserver"],
    "typescript": ["typescript-language-server"],
    "tsx": ["typescript-language-server"],
    "javascript": ["typescript-language-server"],
    "jsx": ["typescript-language-server"],
    "zig": ["zls"],
    "go": ["gopls"],
    "rust": ["rust-analyzer"],
}

# ─── Exceptions ───────────────────────────────────────────────────────


class LSPLaunchError(Exception):
    """Raised when the LSP subprocess cannot be started (e.g. ENOENT)."""


class LSPInitializeError(Exception):
    """Raised when the LSP initialize handshake fails."""


class LSPAllowlistViolation(Exception):
    """Raised when config sets ``command:`` without ``allow_untrusted=True``.

    Per DESIGN.md §6.2: arbitrary command execution from a hostile
    ``.scry/config.yaml`` is prevented by requiring the explicit opt-in flag.
    """


# ─── Launch spec ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class LSPLaunchSpec:
    """Immutable parameters for spawning one LSP server.

    ``command`` is the *resolved* binary path (from :func:`shutil.which`
    or the allow-untrusted override).  On Windows, a ``.cmd``/``.bat``
    suffix causes the manager to wrap the invocation in ``cmd.exe /C``.
    """

    language: str
    command: str
    args: list[str]
    cwd: Path


# ─── Session ──────────────────────────────────────────────────────────


@dataclass
class LSPSession:
    """A single live LSP subprocess and its JSON-RPC wires.

    Owns:

    * the ``asyncio.subprocess.Process`` handle
    * a :class:`~scry.lsp.proto.LSPStreamReader` /
      :class:`~scry.lsp.proto.LSPStreamWriter` pair on its stdin/stdout
    * a per-session request-id counter
    * a future map for in-flight requests (``id`` → ``Future``)
    * a background task that reads messages and dispatches them

    Lifecycle::

        await session.start()    # spawns + initialize handshake
        result = await session.request(method, params, timeout=30)
        await session.notify(method, params)
        await session.shutdown()  # shutdown request + exit notification

    ``capabilities`` is populated by :meth:`start` after the initialize
    handshake; use :meth:`supports` for dotted-path capability queries.
    """

    language: str
    spec: LSPLaunchSpec
    capabilities: dict[str, Any] = field(default_factory=dict)
    allow_untrusted: bool = field(default=False)

    # ── Internal state — set by start(), not part of __init__ ──────────
    _proc: asyncio.subprocess.Process | None = field(default=None, init=False, repr=False)
    _stream_reader: LSPStreamReader | None = field(default=None, init=False, repr=False)
    _stream_writer: LSPStreamWriter | None = field(default=None, init=False, repr=False)
    _next_id: int = field(default=1, init=False, repr=False)
    _pending: dict[int | str, asyncio.Future[Any]] = field(
        default_factory=dict, init=False, repr=False
    )
    _reader_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)

    # ── Public API ───────────────────────────────────────────────────────

    async def start(self) -> None:
        """Spawn the LSP process and complete the initialize handshake.

        After this method returns, :attr:`capabilities` is populated and
        the session is ready for :meth:`request` / :meth:`notify` calls.

        Raises
        ------
        LSPLaunchError
            When the subprocess cannot be created (e.g. binary not found).
        LSPInitializeError
            When the initialize handshake fails (timeout, protocol error,
            server returned an error response).
        """
        cmd = _build_spawn_cmd(self.spec)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                cwd=str(self.spec.cwd),
            )
        except OSError as exc:
            raise LSPLaunchError(
                f"Failed to spawn LSP for '{self.language}' (cmd={cmd!r}): {exc}"
            ) from exc

        self._proc = proc
        assert proc.stdin is not None, "stdin should be PIPE"
        assert proc.stdout is not None, "stdout should be PIPE"

        self._stream_reader = LSPStreamReader(proc.stdout)
        self._stream_writer = LSPStreamWriter(proc.stdin)

        # Start background dispatch loop
        self._reader_task = asyncio.create_task(
            self._read_loop(), name=f"lsp-reader-{self.language}"
        )

        # Initialize handshake (LSP §3.1)
        # Look up a per-language adapter (W3c).  If one exists, delegate
        # both the initialize params and the post-initialize workspace
        # settings to it; otherwise fall back to minimal inline params so
        # that languages without a dedicated adapter (go, rust) still work.
        adapter = get_adapter(self.language)
        if adapter is not None:
            init_params: dict[str, Any] = adapter.prepare_initialize_params(
                self.spec.cwd, self.allow_untrusted
            )
        else:
            # Fallback for languages without a W3c adapter (go, rust).
            # Capability shape MUST nest under textDocument per LSP spec
            # (review-w3c MEDIUM fix); a flat top-level "callHierarchy"
            # is silently ignored by servers, weakening §5.3 transitive
            # drift detection on the fallback path.
            init_params = {
                "processId": os.getpid(),
                "rootUri": self.spec.cwd.as_uri(),
                "capabilities": {
                    "textDocument": {
                        "callHierarchy": {"dynamicRegistration": False},
                    },
                },
            }

        try:
            result = await self.request(
                "initialize",
                init_params,
                timeout=30.0,
            )
            if isinstance(result, dict):
                self.capabilities = result.get("capabilities", {})
            await self.notify("initialized", {})

            # Post-initialize: push workspace settings to the server.
            # Sent as a notification (fire-and-forget) per LSP spec.
            # Only dispatched when the adapter provides non-empty settings;
            # ZLS (and fallback adapters) skip this step automatically.
            if adapter is not None:
                settings = adapter.initial_workspace_settings()
                if settings:
                    await self.notify(
                        "workspace/didChangeConfiguration",
                        {"settings": settings},
                    )
        except Exception as exc:
            await self._kill_subprocess()
            raise LSPInitializeError(
                f"LSP initialize handshake failed for '{self.language}': {exc}"
            ) from exc

    async def request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: float = 30.0,
    ) -> Any:
        """Issue a JSON-RPC request and await the response.

        Raises
        ------
        LSPProtocolError
            When the server returned a JSON-RPC error response.
        asyncio.TimeoutError
            When *timeout* seconds elapse with no response.
        """
        if self._stream_writer is None:
            raise LSPProtocolError("LSP session not started")

        req_id = self._next_id
        self._next_id += 1

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Any] = loop.create_future()
        self._pending[req_id] = fut

        try:
            await self._stream_writer.write_message(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "method": method,
                    "params": params,
                }
            )
            return await asyncio.wait_for(asyncio.shield(fut), timeout=timeout)
        except TimeoutError:
            self._pending.pop(req_id, None)
            raise
        except asyncio.CancelledError:
            # Caller task was cancelled — the shielded future would
            # otherwise remain in self._pending forever, leaking memory
            # and risking a stale response landing on a future the
            # caller no longer awaits (review-w3a MEDIUM #2).
            self._pending.pop(req_id, None)
            if not fut.done():
                fut.cancel()
            raise
        except Exception:
            self._pending.pop(req_id, None)
            raise

    async def notify(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Send a JSON-RPC notification (fire-and-forget; no response)."""
        if self._stream_writer is None:
            return
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        await self._stream_writer.write_message(msg)

    async def shutdown(self) -> None:
        """Send ``shutdown`` + ``exit`` and wait for process termination.

        A 5-second grace period is given; the process is killed if it
        does not exit within that window.
        """
        if self._proc is None:
            return

        # 1. shutdown request
        try:
            await self.request("shutdown", {}, timeout=10.0)
        except Exception as exc:
            logger.debug("LSP [%s] shutdown request failed: %s", self.language, exc)

        # 2. exit notification (best-effort)
        try:
            await self.notify("exit")
        except Exception as exc:
            logger.debug("LSP [%s] exit notification failed: %s", self.language, exc)

        # 3. cancel background reader
        if self._reader_task is not None and not self._reader_task.done():
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader_task

        # 4. wait for process with grace period
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=5.0)
        except TimeoutError:
            logger.warning("LSP [%s] did not exit within grace period; killing", self.language)
            self._proc.kill()
            with contextlib.suppress(Exception):
                await self._proc.wait()

    @property
    def is_alive(self) -> bool:
        """Return ``True`` when the subprocess is still running."""
        return self._proc is not None and self._proc.returncode is None

    def supports(self, capability_path: str) -> bool:
        """Return whether ``ServerCapabilities.<dotted.path>`` is truthy.

        Example::

            session.supports("callHierarchyProvider")
            session.supports("textDocumentSync.openClose")
        """
        obj: Any = self.capabilities
        for part in capability_path.split("."):
            if not isinstance(obj, dict):
                return False
            obj = obj.get(part)
            if obj is None:
                return False
        return bool(obj)

    # ── Internal helpers ─────────────────────────────────────────────────

    async def _read_loop(self) -> None:
        """Background task: read messages and dispatch to pending futures."""
        if self._stream_reader is None:
            return
        try:
            while True:
                msg = await self._stream_reader.read_message()
                self._dispatch(msg)
        except asyncio.CancelledError:
            raise
        except LSPProtocolError as exc:
            logger.error("LSP [%s] protocol error: %s", self.language, exc)
        except Exception as exc:
            logger.debug("LSP [%s] reader closed: %s", self.language, exc)
        finally:
            # Resolve all pending futures with a connection-closed error
            err = LSPProtocolError(f"LSP ['{self.language}'] connection closed unexpectedly")
            for fut in list(self._pending.values()):
                if not fut.done():
                    fut.set_exception(err)
            self._pending.clear()

    def _dispatch(self, msg: LSPMessage) -> None:
        """Route one incoming message to the appropriate handler."""
        if msg.id is not None and msg.method is None:
            # Response to one of our requests
            fut = self._pending.pop(msg.id, None)
            if fut is not None and not fut.done():
                if msg.error is not None:
                    fut.set_exception(
                        LSPProtocolError(f"LSP error response [{msg.id}]: {msg.error}")
                    )
                else:
                    fut.set_result(msg.result)
        elif msg.method is not None and msg.id is not None:
            # Server-originated request — reply method-not-found (Wave 3a)
            task = asyncio.create_task(
                self._send_method_not_found(msg.id, msg.method),
                name=f"lsp-mnf-{self.language}",
            )
            task.add_done_callback(lambda t: None)
        elif msg.method is not None:
            # Notification
            self._handle_notification(msg)

    async def _send_method_not_found(self, req_id: int | str, method: str) -> None:
        if self._stream_writer is None:
            return
        try:
            await self._stream_writer.write_message(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}",
                    },
                }
            )
        except Exception as exc:
            logger.debug("LSP [%s] failed to send method-not-found: %s", self.language, exc)

    def _handle_notification(self, msg: LSPMessage) -> None:
        method = msg.method or ""
        if method == "window/logMessage":
            params = msg.params or {}
            level = params.get("type", 4)
            text = str(params.get("message", ""))
            _LOG_LEVEL_MAP: dict[int, Any] = {
                1: logger.error,
                2: logger.warning,
                3: logger.info,
            }
            log_fn = _LOG_LEVEL_MAP.get(level, logger.debug)
            log_fn("LSP [%s]: %s", self.language, text)
        elif method in ("window/showMessage", "telemetry/event"):
            pass  # intentionally ignored per spec
        else:
            logger.debug("LSP [%s] notification: %s", self.language, method)

    async def _kill_subprocess(self) -> None:
        """Cancel the reader task and kill the subprocess."""
        if self._reader_task is not None and not self._reader_task.done():
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader_task
        if self._proc is not None:
            with contextlib.suppress(Exception):
                self._proc.kill()
                await self._proc.wait()


# ─── Manager ──────────────────────────────────────────────────────────


class LSPManager:
    """Owns one :class:`LSPSession` per language; lazy-spawns on first request.

    Usage::

        async with LSPManager(repo_root, config) as mgr:
            session = await mgr.session_for("python")
            if session is not None:
                result = await session.request("callHierarchy/prepare", params)

    Parameters
    ----------
    repo_root:
        Absolute path to the repository root (used as ``rootUri`` and ``cwd``).
    config:
        :class:`~scry.models.CodeAnchorsConfig` from ``.scry/config.yaml``.
    allow_untrusted:
        When ``True``, a ``command:`` override in the ``lsp:`` config block
        is used verbatim instead of being rejected.  Corresponds to the
        ``--allow-untrusted-lsp-config`` CLI flag (DESIGN.md §6.2).
    """

    def __init__(
        self,
        repo_root: Path,
        config: CodeAnchorsConfig,
        *,
        allow_untrusted: bool = False,
    ) -> None:
        self._repo_root = repo_root
        self._config = config
        self._allow_untrusted = allow_untrusted
        self._sessions: dict[str, LSPSession] = {}
        self._failed: set[str] = set()
        # Languages rejected because they use a custom command without
        # --allow-untrusted-lsp-config.  These are tracked separately so
        # status_for() can map them to "unsupported" (not "lsp_unavailable").
        self._untrusted_rejected: set[str] = set()

    async def session_for(self, language: str) -> LSPSession | None:
        """Return a live session for *language*, spawning one if necessary.

        Returns ``None`` when:

        * *language* is configured as ``lsp.<lang>: skip`` (intentional,
          no warning emitted).
        * No allowlisted binary is on ``PATH`` (WARNING logged).
        * The LSP failed to start and the failure is cached (no retry).

        Raises
        ------
        LSPAllowlistViolation
            When ``config.lsp.<language>.command`` is set but
            ``allow_untrusted=False`` (DESIGN.md §6.2).
        """
        # Return cached alive session
        if language in self._sessions:
            session = self._sessions[language]
            if session.is_alive:
                return session
            # Session died unexpectedly; remove it so we don't retry below
            del self._sessions[language]

        # Don't retry previously failed languages
        if language in self._failed:
            return None

        # Intentional skip — configured per-language. Two cases:
        #   1. Explicit `lsp.<lang>: skip` directive
        #   2. Language is NOT in `code_anchors.languages` at all
        #      (review-w3a MEDIUM #3 fix: previously, an unconfigured
        #      allowlisted language could spawn anyway because the check
        #      only looked at the explicit 'skip' value, not absence)
        lang_directive = self._config.languages.get(language)
        if lang_directive is None:
            logger.debug(
                "LSP [%s] not configured in code_anchors.languages; skipping",
                language,
            )
            return None
        if lang_directive == "skip":
            logger.debug("LSP [%s] intentionally skipped (languages: skip)", language)
            return None

        # Resolve binary and build launch spec (may raise LSPAllowlistViolation)
        try:
            spec = self._resolve_spec(language)
        except LSPAllowlistViolation as exc:
            # Per-language rejection: do not abort the entire indexing run.
            # Treat this language as unsupported and continue (DESIGN.md §6.2).
            logger.warning(
                "LSP [%s] config rejected (no --allow-untrusted-lsp-config): %s; "
                "code anchors for this language will use unsupported status",
                language,
                exc,
            )
            self._failed.add(language)
            self._untrusted_rejected.add(language)
            return None
        if spec is None:
            self._failed.add(language)
            return None

        # Spawn and handshake
        session = LSPSession(language=language, spec=spec, allow_untrusted=self._allow_untrusted)
        try:
            await session.start()
        except (LSPLaunchError, LSPInitializeError) as exc:
            logger.warning(
                "LSP [%s] failed to start: %s; code anchors will use lsp_unavailable status",
                language,
                exc,
            )
            self._failed.add(language)
            return None

        self._sessions[language] = session
        logger.info(
            "LSP [%s] ready; callHierarchy=%s",
            language,
            session.supports("callHierarchyProvider"),
        )
        return session

    def status_for(
        self, language: str
    ) -> Literal["available", "skip", "lsp_unavailable", "unknown"]:
        """Return the current availability status for *language* without spawning.

        Does NOT start a new session.  The return value reflects the most
        recent state (i.e. post any previous ``session_for`` call).

        Returns
        -------
        ``"available"``
            A live session already exists for this language.
        ``"skip"``
            ``languages.<lang>: skip`` is configured — intentionally disabled.
            Also returned when the language was rejected because it uses a
            custom ``command:`` without ``--allow-untrusted-lsp-config``; the
            indexer maps both to :attr:`TransitiveHashStatus.UNSUPPORTED`.
        ``"lsp_unavailable"``
            The binary is missing, the LSP failed to start, or a previously
            active session has died.  Set when ``language in self._failed`` (and
            the failure was NOT an untrusted-command rejection).
        ``"unknown"``
            The language is not listed in ``code_anchors.languages`` at all.
        """
        if language in self._sessions and self._sessions[language].is_alive:
            return "available"
        if language in self._untrusted_rejected:
            # Custom command rejected without flag — treat as intentionally
            # unsupported, not as a binary-missing failure.
            return "skip"
        if language in self._failed:
            return "lsp_unavailable"
        lang_directive = self._config.languages.get(language)
        if lang_directive is None:
            return "unknown"
        if lang_directive == "skip":
            return "skip"
        # Configured as "lsp" but session never started (or died without being
        # added to _failed).  Treat as unavailable for the caller's purposes.
        return "lsp_unavailable"

    async def shutdown_all(self) -> None:
        """Gracefully shut down every active LSP session."""
        for session in list(self._sessions.values()):
            try:
                await session.shutdown()
            except Exception as exc:
                logger.debug("Error shutting down LSP [%s]: %s", session.language, exc)
        self._sessions.clear()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await self.shutdown_all()

    # ── Private helpers ──────────────────────────────────────────────────

    def _resolve_spec(self, language: str) -> LSPLaunchSpec | None:
        """Resolve the binary and args for *language*, enforcing §6.2.

        Returns ``None`` when no usable binary is available.

        Raises
        ------
        LSPAllowlistViolation
            When ``command:`` is set in config without ``allow_untrusted``.
        """
        lsp_cfg = self._config.lsp.get(language, {})
        custom_cmd: str | None = lsp_cfg.get("command")
        config_args: list[str] = lsp_cfg.get("args", [])

        if custom_cmd is not None:
            if self._allow_untrusted:
                # Resolve via PATH if possible; otherwise use verbatim
                resolved: str = shutil.which(custom_cmd) or custom_cmd
            else:
                raise LSPAllowlistViolation(
                    f"LSP config for '{language}' sets "
                    f"`command: {custom_cmd}` which is not permitted "
                    f"without --allow-untrusted-lsp-config (DESIGN.md §6.2). "
                    f"Pass that flag to override the allowlist."
                )
        else:
            # Walk the allowlist in order, returning the first hit
            allowlisted = LSP_ALLOWLIST.get(language, [])
            resolved = ""
            for binary in allowlisted:
                found = shutil.which(binary)
                if found is not None:
                    resolved = found
                    break

            if not resolved:
                if allowlisted:
                    logger.warning(
                        "No allowlisted LSP binary found for '%s' "
                        "(tried: %s); code anchors will use lsp_unavailable status",
                        language,
                        ", ".join(allowlisted),
                    )
                else:
                    logger.warning(
                        "Language '%s' has no entry in LSP_ALLOWLIST; "
                        "code anchors will use lsp_unavailable status",
                        language,
                    )
                return None

        return LSPLaunchSpec(
            language=language,
            command=resolved,
            args=config_args,
            cwd=self._repo_root,
        )


# ─── Cross-platform spawn helpers ─────────────────────────────────────


# Characters that have special meaning to Windows ``cmd.exe`` and that we
# refuse to forward through the ``cmd.exe /C`` shim path.  Allowing them
# would let a hostile repo's ``lsp.<lang>.args`` config inject arbitrary
# commands (e.g. ``args: ['--stdio', '&', 'calc.exe']``) — review-w3a
# BLOCKING bug.  Note: ``allow_untrusted=True`` does NOT relax this; it
# only relaxes the ``command:`` constraint.  Args are always
# repo-controlled and always validated through the shim path.
_CMD_SHELL_METACHARS = frozenset('&|<>^"`%\n\r\t()')


def _arg_is_safe_for_cmd_exe(arg: str) -> bool:
    """Return ``True`` if *arg* contains no Windows cmd.exe metacharacters.

    Reject (return ``False``) when the arg contains any of:
        ``&  |  <  >  ^  "  \\`  %  \\n  \\r  \\t  (  )``
    These all have shell-syntactic meaning under ``cmd.exe /C``.  Even
    inside double-quotes a ``"`` ends the quoted region and a ``%``
    triggers env-var expansion, so quoting alone is not a safe escape.
    """
    return not any(c in _CMD_SHELL_METACHARS for c in arg)


def _build_spawn_cmd(spec: LSPLaunchSpec) -> list[str]:
    """Build the full argument list for :func:`asyncio.create_subprocess_exec`.

    On Windows, ``.cmd`` and ``.bat`` shims (common for npm-installed tools
    such as ``typescript-language-server``) cannot be executed directly by
    ``CreateProcess``.  They must be wrapped in ``cmd.exe /C`` (DESIGN.md §10.5).

    Security (review-w3a BLOCKING fix): when wrapping in ``cmd.exe /C`` we
    refuse to forward args containing shell metacharacters, since the
    args list is repo-controlled (from ``.scry/config.yaml``
    ``lsp.<lang>.args``) and the ``--allow-untrusted-lsp-config`` flag
    does NOT cover args (only ``command:``).  Refused args raise
    :class:`LSPLaunchError` which the manager handles by treating the
    language as unavailable.

    Raises:
        LSPLaunchError: if the spec uses a Windows shim AND any arg
            contains a shell metacharacter.
    """
    suffix = Path(spec.command).suffix.lower()
    if sys.platform == "win32" and suffix in (".cmd", ".bat"):
        bad = [a for a in spec.args if not _arg_is_safe_for_cmd_exe(a)]
        if bad:
            raise LSPLaunchError(
                f"refusing to spawn {spec.language!r} LSP: arg(s) "
                f"{bad!r} contain Windows cmd.exe shell metacharacters; "
                f"a .cmd/.bat shim is being wrapped via 'cmd.exe /C' so "
                f"these would inject arbitrary commands. Either rename "
                f"the LSP binary to a real .exe, or remove the offending "
                f"args from .scry/config.yaml lsp.{spec.language}.args."
            )
        return ["cmd.exe", "/C", spec.command, *spec.args]
    return [spec.command, *spec.args]
