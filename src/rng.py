"""Deterministic RNG management — no global np.random calls."""

from __future__ import annotations

import numpy as np


class SeedBank:
    """Generates deterministic, non-overlapping RNG streams per subsystem name.

    Usage:
        sb = SeedBank(master_seed=42)
        pellet_rng = sb.get_rng('pellets')
        agent_rng = sb.get_rng('agents')
        # pellet_rng and agent_rng are independent but deterministic
    """

    def __init__(self, master_seed: int) -> None:
        self._master = np.random.RandomState(master_seed)
        self._rng_cache: dict[str, np.random.RandomState] = {}
        self._seed_cache: dict[str, int] = {}

    def get_rng(self, name: str) -> np.random.RandomState:
        """Return a deterministic RandomState for the given subsystem name.

        Cached after first call — repeated calls with the same name return
        the same RandomState instance (same internal state).
        """
        if name not in self._rng_cache:
            subsystem_seed = self._master.randint(0, 2**31, dtype=np.int64)
            self._seed_cache[name] = subsystem_seed
            self._rng_cache[name] = np.random.RandomState(subsystem_seed)
        return self._rng_cache[name]

    def get_seed(self, name: str) -> int:
        """Return the raw integer seed for a subsystem (for logging/debugging).

        Does not mutate the RNG state — safe to call at any time.
        """
        if name not in self._seed_cache:
            self.get_rng(name)
        return self._seed_cache[name]
