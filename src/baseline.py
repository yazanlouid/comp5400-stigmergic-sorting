"""Deneubourg baseline agent — hand-designed stigmergic sorting rules."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from .agents import BaseAgent, Action, SensorReading

if TYPE_CHECKING:
    from .arena import Arena


class DeneubourgAgent(BaseAgent):
    """Agent implementing Deneubourg's stigmergic sorting model.

    Pickup/drop probabilities follow the original formulation:

        P(pickup) = (k1 / (k1 + f))**2    when not carrying
        P(drop)   = (f / (k2 + f))**2     when carrying

    where *f* is the fraction of similar-coloured pellets in the local
    sensor window.
    """

    def __init__(
        self,
        agent_id: int,
        x: float,
        y: float,
        heading_deg: float,
        k1: float = 0.1,
        k2: float = 0.3,
    ) -> None:
        super().__init__(agent_id, x, y, heading_deg)
        self.k1: float = k1
        self.k2: float = k2

    # Decision

    def decide_action(
        self,
        arena: Arena,
        sensor_radius: float,
        rng: np.random.RandomState,
    ) -> Action:
        """Choose action using Deneubourg's probabilistic rules.

        When not carrying, computes *f* as the fraction of the dominant
        colour in the sensor window.  When carrying, *f* is the fraction
        of pellets matching the carried pellet's colour.

        Falls back to a random walk (MOVE / TURN_LEFT / TURN_RIGHT) when
        the probabilistic pickup/drop does not trigger.
        """
        nearby = arena.get_pellets_in_radius(self.x, self.y, sensor_radius)

        if self.carrying is None:
            return self._decide_not_carrying(nearby, rng)
        else:
            return self._decide_carrying(nearby, rng)

    def _decide_not_carrying(
        self,
        nearby: list,
        rng: np.random.RandomState,
    ) -> Action:
        """Pickup decision when the agent is not carrying a pellet."""
        if len(nearby) == 0:
            return Action(rng.randint(0, 3))  # random walk

        # Count pellets per colour
        counts: dict[str, int] = {}
        for p in nearby:
            counts[p.colour] = counts.get(p.colour, 0) + 1

        # f = fraction of the dominant colour
        f = max(counts.values()) / len(nearby)

        # P(pickup) = (k1 / (k1 + f))**2
        p_pickup = (self.k1 / (self.k1 + f)) ** 2

        if rng.random() < p_pickup:
            return Action.PICKUP

        return Action(rng.randint(0, 3))  # random walk

    def _decide_carrying(
        self,
        nearby: list,
        rng: np.random.RandomState,
    ) -> Action:
        """Drop decision when the agent is carrying a pellet."""
        if len(nearby) == 0:
            return Action(rng.randint(0, 3))  # random walk

        # f = fraction of similar-coloured pellets
        my_colour = self.carrying.colour
        similar = sum(1 for p in nearby if p.colour == my_colour)
        f = similar / len(nearby)

        # P(drop) = (f / (k2 + f))**2
        p_drop = (f / (self.k2 + f)) ** 2

        if rng.random() < p_drop:
            return Action.DROP

        return Action(rng.randint(0, 3))  # random walk
