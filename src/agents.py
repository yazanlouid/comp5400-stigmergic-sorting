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

    # Sensing

    def sense(self, arena: Arena, sensor_radius: float) -> SensorReading:
        """Query the arena for local pellet fractions."""
        nearby = arena.get_pellets_in_radius(self.x, self.y, sensor_radius)
        total = len(nearby)

        if total == 0:
            return SensorReading(
                red_density=0.0,
                blue_density=0.0,
                similar_density=0.0,
                dissimilar_density=0.0,
            )

        red_count = sum(1 for p in nearby if p.colour == "red")
        blue_count = sum(1 for p in nearby if p.colour == "blue")

        red_fraction = float(red_count) / total
        blue_fraction = float(blue_count) / total

        if self.carrying is not None:
            my_colour = self.carrying.colour
            similar = sum(1 for p in nearby if p.colour == my_colour)
            dissimilar = total - similar

            similar_fraction = float(similar) / total
            dissimilar_fraction = float(dissimilar) / total
        else:
            similar_fraction = 0.0
            dissimilar_fraction = 0.0

        return SensorReading(
            red_density=red_fraction,
            blue_density=blue_fraction,
            similar_density=similar_fraction,
            dissimilar_density=dissimilar_fraction,
        )

    # Actuation

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

    # Decision (override in subclasses)

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

    # Helpers

    @staticmethod
    def _normalize_heading(heading: float) -> float:
        """Wrap *heading* into the range [-180, 180]."""
        return (heading + 180) % 360 - 180


class EvolvedAgent(BaseAgent):
    """Agent controlled by an evolved NeuralController.

    Senses the local environment, builds a 6-d input vector,
    runs the controller forward pass, and selects an action
    stochastically from the softmax output.
    """

    def __init__(
        self,
        agent_id: int,
        x: float,
        y: float,
        heading_deg: float,
        controller,
    ) -> None:
        """Create an evolved agent with the given controller.

        Args:
            agent_id: Unique identifier.
            x: Horizontal position.
            y: Vertical position.
            heading_deg: Facing direction in degrees.
            controller: NeuralController instance (shared across agents).
        """
        super().__init__(agent_id, x, y, heading_deg)
        self.controller = controller

    def decide_action(
        self,
        arena: Arena,
        sensor_radius: float,
        rng: np.random.RandomState,
    ) -> Action:
        """Choose action via neural controller forward pass.

        1. Sense local pellet densities.
        2. Build 6-d input vector from sensor reading + carry state.
        3. Run controller forward pass → softmax probabilities.
        4. Stochastic action selection from probability distribution.

        Args:
            arena: The arena to sense.
            sensor_radius: Radius of the sensor window.
            rng: Seeded RandomState for stochastic action selection.

        Returns:
            Selected discrete action.
        """
        reading = self.sense(arena, sensor_radius)

        carrying_red = (
            1.0 if self.carrying is not None and self.carrying.colour == "red" else 0.0
        )
        carrying_blue = (
            1.0 if self.carrying is not None and self.carrying.colour == "blue" else 0.0
        )

        input_vec = np.array(
            [
                reading.red_density,
                reading.blue_density,
                reading.similar_density,
                reading.dissimilar_density,
                carrying_red,
                carrying_blue,
            ],
            dtype=np.float64,
        )

        probs = self.controller.forward(input_vec)
        action_idx = int(rng.choice(len(probs), p=probs))
        return Action(action_idx)


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
