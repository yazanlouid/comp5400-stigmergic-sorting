"""World simulation — pellets, spatial hashing, boundary physics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .rng import SeedBank

# Canonical colour palette — Arena uses the first *num_colours* entries.
_COLOURS: list[str] = ["red", "blue", "green", "yellow", "orange", "purple"]


@dataclass(eq=False)
class Pellet:
    """A single pellet in the arena.

    Hashable by ``id`` so it can be stored in the SpatialHash buckets.
    """

    id: int
    x: float
    y: float
    colour: str

    def __hash__(self) -> int:
        return hash(self.id)


class SpatialHash:
    """Grid-based spatial hash for O(1) average pellet queries.

    The grid cell size should match the typical sensor radius so that
    a query only inspects the centre cell and its 8 neighbours.
    """

    def __init__(self, cell_size: float) -> None:
        """Initialise with the given cell size."""
        self.cell_size: float = cell_size
        self._buckets: dict[tuple[int, int], set[Pellet]] = {}

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        """Convert world coordinates to grid-cell indices."""
        return (
            int(math.floor(x / self.cell_size)),
            int(math.floor(y / self.cell_size)),
        )

    def add(self, pellet: Pellet) -> None:
        """Insert *pellet* into the correct bucket."""
        key = self._cell(pellet.x, pellet.y)
        self._buckets.setdefault(key, set()).add(pellet)

    def remove(self, pellet: Pellet) -> None:
        """Remove *pellet* from its current bucket."""
        key = self._cell(pellet.x, pellet.y)
        bucket = self._buckets.get(key)
        if bucket is not None:
            bucket.discard(pellet)
            if not bucket:
                del self._buckets[key]

    def query(self, x: float, y: float, radius: float) -> list[Pellet]:
        """Return all pellets within *radius* of *(x, y)*.

        Checks the centre cell and its 8 neighbours, then filters by
        actual Euclidean distance.
        """
        cx, cy = self._cell(x, y)
        # Number of cells to scan in each direction
        span = int(math.ceil(radius / self.cell_size))
        results: list[Pellet] = []
        r2 = radius * radius
        for dx in range(-span, span + 1):
            for dy in range(-span, span + 1):
                bucket = self._buckets.get((cx + dx, cy + dy))
                if bucket is None:
                    continue
                for p in bucket:
                    if (p.x - x) ** 2 + (p.y - y) ** 2 <= r2:
                        results.append(p)
        return results

    def clear(self) -> None:
        """Empty all buckets."""
        self._buckets.clear()

    def rebuild(self, pellets: list[Pellet]) -> None:
        """Clear and re-add all *pellets*."""
        self.clear()
        for p in pellets:
            self.add(p)


class Arena:
    """2-D arena containing pellets with boundary clamping and spatial queries."""

    def __init__(
        self,
        width: float,
        height: float,
        pellet_count: int,
        num_colours: int,
        seed_bank: SeedBank,
    ) -> None:
        self.width: float = width
        self.height: float = height
        self.pellets: list[Pellet] = []
        self._seed_bank: SeedBank = seed_bank
        self._hash: SpatialHash = SpatialHash(cell_size=5.0)

        # Pre-compute colour list
        self._colours: list[str] = _COLOURS[:num_colours]
        self._pellets_per_colour: int = pellet_count // num_colours

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------

    def place_pellets(self) -> None:
        """Deterministically place pellets using the SeedBank."""
        rng: np.random.RandomState = self._seed_bank.get_rng("pellets")
        self.pellets.clear()
        self._hash.clear()

        pid = 0
        for colour in self._colours:
            for _ in range(self._pellets_per_colour):
                x = rng.uniform(0.0, self.width)
                y = rng.uniform(0.0, self.height)
                p = Pellet(id=pid, x=x, y=y, colour=colour)
                self.pellets.append(p)
                self._hash.add(p)
                pid += 1

    # ------------------------------------------------------------------
    # Boundary
    # ------------------------------------------------------------------

    def clamp_position(self, x: float, y: float) -> tuple[float, float]:
        """Clamp *(x, y)* to the arena boundaries."""
        return (max(0.0, min(x, self.width)), max(0.0, min(y, self.height)))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_pellets_in_radius(self, x: float, y: float, radius: float) -> list[Pellet]:
        """Delegate to spatial hash for radius query."""
        return self._hash.query(x, y, radius)

    def get_pellets_by_colour_in_radius(
        self, x: float, y: float, radius: float, colour: str
    ) -> list[Pellet]:
        """Filter radius query result by *colour*."""
        return [p for p in self._hash.query(x, y, radius) if p.colour == colour]

    # ------------------------------------------------------------------
    # Pickup / Drop
    # ------------------------------------------------------------------

    def pickup_pellet(
        self, x: float, y: float, pickup_radius: float = 1.0
    ) -> Optional[Pellet]:
        """Pick up the nearest pellet within *pickup_radius* of *(x, y)*.

        Removes the pellet from the arena and spatial hash.  Returns
        ``None`` when no pellet is nearby.
        """
        candidates: list[Pellet] = self._hash.query(x, y, pickup_radius)
        if not candidates:
            return None
        # Nearest by Euclidean distance
        nearest: Pellet = min(candidates, key=lambda p: (p.x - x) ** 2 + (p.y - y) ** 2)
        self._hash.remove(nearest)
        self.pellets.remove(nearest)
        return nearest

    def drop_pellet(self, x: float, y: float, pellet: Pellet) -> None:
        """Drop *pellet* at *(x, y)* (clamped to arena bounds)."""
        cx, cy = self.clamp_position(x, y)
        pellet.x = cx
        pellet.y = cy
        self.pellets.append(pellet)
        self._hash.add(pellet)

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------

    def get_all_pellet_positions(self) -> list[tuple[float, float]]:
        """Return [(x, y), ...] for every pellet in the arena."""
        return [(p.x, p.y) for p in self.pellets]

    def get_all_pellet_colours(self) -> list[str]:
        """Return [colour, ...] parallel to :meth:`get_all_pellet_positions`."""
        return [p.colour for p in self.pellets]
