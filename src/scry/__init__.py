"""scry — local-first MCP server for code↔spec drift detection.

See DESIGN.md for the architectural specification.
"""

from __future__ import annotations

# UT1-10: silence the AuthlibDeprecationWarning that fastmcp triggers
# via its JWT auth provider import.  authlib's own deprecate.py runs
# ``warnings.simplefilter("always", AuthlibDeprecationWarning)`` at
# import time, which clobbers any prior ignore filter.  We work around
# this by importing authlib's warning class *first* (so authlib runs
# its simplefilter), then registering our ignore filter ON TOP of it.
# When authlib is absent (e.g. fastmcp isn't installed), this is a no-op.
import warnings as _warnings

try:
    from authlib.deprecate import AuthlibDeprecationWarning as _AuthlibWarn

    _warnings.simplefilter("ignore", _AuthlibWarn)
except ImportError:  # pragma: no cover — authlib may be optional
    _warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module=r"authlib(\..*)?",
    )


__version__ = "0.0.1"

__all__ = ["__version__"]
