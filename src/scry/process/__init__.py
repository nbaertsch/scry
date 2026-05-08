"""Process coordination package for scry.

Exposes leader election and lock-file management (W2g) and — once W2h lands —
the IPC transport layer.  See DESIGN.md §10 for the full process model.
"""

from scry.process.leader import (
    LeaderLock,
    LeaderMetadata,
    LeaderState,
    LockTimeout,
    StaleLockError,
    detect_leader_state,
    read_leader_metadata_if_present,
)

__all__ = [
    "LeaderLock",
    "LeaderMetadata",
    "LeaderState",
    "LockTimeout",
    "StaleLockError",
    "detect_leader_state",
    "read_leader_metadata_if_present",
]
