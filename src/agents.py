"""Base agent class — sensors, actuators, and action space for stigmergic sorting."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from math import cos, sin, radians
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .arena import Arena, Pellet
    from .rng import SeedBank


class Action(IntEnum):
    """Discrete actions available to agents."""

    MOVE = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    PICKUP = 3
    DROP = 4

    @classmethod
    def names(cls) -> list[str]:
        """Return list of action names in enum order."""
        return ["MOVE", "TURN_LEFT", "TURN_RIGHT", "PICKUP", "DROP"]


@dataclass
class SensorReading:
    """Local pellet-density snapshot from an agent's sensor."""

    red_density: float
    blue_density: float
    similar_density: float
    dissimilar_density: float


class BaseAgent:
    """Minimal agent with position, heading, and carry state.

    Subclasses override :meth:`decide_action` for behaviour logic.
    The base implementation performs a uniform random walk.
    """

    def __init__(self, agent_id: int, x: float, y: float, heading_deg: float) -> None:
        """Create an agent at *(x, y)* facing *heading_deg* degrees.

        Args:
            agent_id: Unique identifier for this agent.
            x: Horizontal position in arena coordinates.
            y: Vertical position in arena coordinates.
            heading_deg: Facing direction in degrees, normalised to [-180, 180].
        """
        self.id: int = agent_id
        self.x: float = x
        self.y: float = y
        self.heading_deg: float = heading_deg
        self.carrying: Pellet | None = None

    # ------------------------------------------------------------------
    # Sensing
    # ------------------------------------------------------------------

    def sense(self, arena: Arena, sensor_radius: float) -> SensorReading:
        """Query the arena for local pellet densities.

        Args:
            arena: The arena to query.
            sensor_radius: Radius (in world units) of the sensor window.

        Returns:
            SensorReading with red/blue counts and similar/dissimilar
            densities relative to the pellet the agent is carrying.
        """
        nearby = arena.get_pellets_in_radius(self.x, self.y, sensor_radius)
        total = len(nearby)

        red_count = sum(1 for p in nearby if p.colour == "red")
        blue_count = sum(1 for p in nearby if p.colour == "blue")

        if self.carrying is not None:
            my_colour = self.carrying.colour
            similar = sum(1 for p in nearby if p.colour == my_colour)
            dissimilar = total - similar
        else:
            similar = 0.0
            dissimilar = float(total)

        return SensorReading(
            red_density=float(red_count),
            blue_density=float(blue_count),
            similar_density=float(similar),
            dissimilar_density=float(dissimilar),
        )

    # ------------------------------------------------------------------
    # Actuation
    # ------------------------------------------------------------------

    def apply_action(self, action: Action, arena: Arena) -> None:
        """Execute *action* and update agent/arena state.

        Args:
            action: The discrete action to perform.
            arena: The arena the agent interacts with.
        """
        if action == Action.MOVE:
            dx = cos(radians(self.heading_deg))
            dy = sin(radians(self.heading_deg))
            self.x, self.y = arena.clamp_position(self.x + dx, self.y + dy)

        elif action == Action.TURN_LEFT:
            self.heading_deg = self._normalize_heading(self.heading_deg - 30)

        elif action == Action.TURN_RIGHT:
            self.heading_deg = self._normalize_heading(self.heading_deg + 30)

        elif action == Action.PICKUP:
            if self.carrying is None:
                self.carrying = arena.pickup_pellet(self.x, self.y)

        elif action == Action.DROP:
            if self.carrying is not None:
                arena.drop_pellet(self.x, self.y, self.carrying)
                self.carrying = None

    # ------------------------------------------------------------------
    # Decision (override in subclasses)
    # ------------------------------------------------------------------

    def decide_action(
        self,
        arena: Arena,
        sensor_radius: float,
        rng: np.random.RandomState,
    ) -> Action:
        """Choose the next action — default random walk.

        Args:
            arena: The arena (unused by default implementation).
            sensor_radius: Sensor radius (unused by default implementation).
            rng: Seeded RandomState for reproducibility.

        Returns:
            A randomly chosen action from [MOVE, TURN_LEFT, TURN_RIGHT].
        """
        idx = rng.randint(0, 3)
        return Action(idx)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_heading(heading: float) -> float:
        """Wrap *heading* into the range [-180, 180]."""
        return (heading + 180) % 360 - 180


def create_agents(count: int, arena: Arena, seed_bank: SeedBank) -> list[BaseAgent]:
    """Create *count* agents with random positions and headings.

    Args:
        count: Number of agents to create.
        arena: Arena providing boundary dimensions.
        seed_bank: SeedBank for deterministic RNG streams.

    Returns:
        List of BaseAgent instances with sequential IDs starting at 0.
    """
    rng = seed_bank.get_rng("agents")
    agents: list[BaseAgent] = []
    for i in range(count):
        x = rng.uniform(0.0, arena.width)
        y = rng.uniform(0.0, arena.height)
        heading = rng.uniform(-180.0, 180.0)
        agents.append(BaseAgent(agent_id=i, x=x, y=y, heading_deg=heading))
    return agents
