"""Claim-level documentation verification for scry.

This package adds factual claim extraction and verification on top of
scry's existing semantic drift detection.  While ``find_drift`` answers
"is this paragraph still *about* the right code?", the claims system
answers "are the *specific facts* in this paragraph still *true*?".

Architecture
------------
1. **Extractor** — parses markdown into atomic, typed ``Claim`` objects
2. **Verifiers** — typed checkers that compare claims against code facts
3. **Store** — persists claims, dependencies, and cached verdicts
4. **Orchestrator** — wires extract → verify → report for CLI/MCP
"""
