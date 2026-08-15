"""Environment variable name verifier using the pre-built RepoIndex."""

from __future__ import annotations

from pathlib import Path

from scry.claims.model import (
    Claim,
    ClaimType,
    Evidence,
    Severity,
    Verdict,
    VerificationResult,
)
from scry.claims.repo_index import RepoIndex
from scry.claims.verifiers import BaseVerifier, register

# ALL_CAPS words that look like env vars but are Dockerfile instructions,
# Kubernetes terms, enum values, or generic labels
_ENVVAR_FALSE_POSITIVE = frozenset({
    "ENTRYPOINT", "WORKDIR", "EXPOSE", "VOLUME", "STOPSIGNAL",
    "HEALTHCHECK", "ONBUILD", "MAINTAINER", "LABEL", "COPY", "ADD",
    "STARTING", "STOPPING", "DESTROYING", "PROVISIONING",
    "UNTRUSTED_AGENT", "UNTRUSTED_GRADING", "TRUSTED_INTERNAL", "SHARED",
    "CONTAINER", "LOCALHOST", "EMPTYDIR",
})


class EnvVarVerifier(BaseVerifier):
    """Verify that an environment variable name exists in the codebase."""

    @property
    def name(self) -> str:
        return "env_var"

    @property
    def supported_types(self) -> frozenset[ClaimType]:
        return frozenset({ClaimType.ENV_VAR_NAME})

    def verify(self, claim: Claim, repo_root: Path, index: RepoIndex) -> VerificationResult:
        var_name = claim.subject
        if not var_name:
            return VerificationResult(
                claim_id=claim.id, verdict=Verdict.ERROR,
                reason="No env var name in claim subject", verifier=self.name,
            )

        # Skip known false positives
        if var_name in _ENVVAR_FALSE_POSITIVE:
            return VerificationResult(
                claim_id=claim.id, verdict=Verdict.CONFIRMED, confidence=0.5,
                reason=f"`{var_name}` is a known term (not an env var claim)",
                verifier=self.name,
            )

        hits = index.lookup_string(var_name)
        # Filter out doc-file hits
        code_hits = [h for h in hits if not h.file_path.startswith("docs/") and not h.file_path.endswith(".md")]

        if code_hits:
            evidence = [
                Evidence(file_path=h.file_path, line=h.line, snippet=h.context)
                for h in code_hits[:5]
            ]
            return VerificationResult(
                claim_id=claim.id, verdict=Verdict.CONFIRMED, confidence=0.9,
                reason=f"Env var `{var_name}` found in code",
                evidence=evidence, verifier=self.name,
            )

        # Prefix matching: SIQE matches SIQE_MODEL, COLOSSEUM_LLM_CRED_ matches prefixed vars
        prefix = var_name.rstrip("_") + "_"
        prefix_hits = [s for s in index.strings if s.startswith(prefix)]
        if prefix_hits:
            return VerificationResult(
                claim_id=claim.id, verdict=Verdict.CONFIRMED, confidence=0.75,
                reason=f"Env var prefix `{var_name}` matches: {', '.join(prefix_hits[:3])}",
                verifier=self.name,
            )

        similar = index.find_similar_strings(var_name)
        if similar:
            return VerificationResult(
                claim_id=claim.id, verdict=Verdict.STALE_TARGET, confidence=0.7,
                severity=Severity.HIGH,
                reason=f"Env var `{var_name}` not found; similar: {', '.join(similar[:5])}",
                expected=var_name, observed=similar[:5], verifier=self.name,
            )

        return VerificationResult(
            claim_id=claim.id, verdict=Verdict.CONTRADICTED, confidence=0.6,
            severity=Severity.MEDIUM,
            reason=f"Env var `{var_name}` not found in codebase",
            expected=var_name, verifier=self.name,
        )


register(EnvVarVerifier())
