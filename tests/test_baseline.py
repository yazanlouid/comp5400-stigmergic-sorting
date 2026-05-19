"""Tests for Deneubourg baseline agent."""

from __future__ import annotations

import numpy as np
import pytest

from src.agents import Action
from src.arena import Arena, Pellet
from src.baseline import DeneubourgAgent
from src.rng import SeedBank


@pytest.fixture
def arena():
    sb = SeedBank(42)
    return Arena(width=100, height=100, pellet_count=10, num_colours=2, seed_bank=sb)


@pytest.fixture
def agent():
    return DeneubourgAgent(agent_id=0, x=50.0, y=50.0, heading_deg=0.0, k1=0.1, k2=0.3)


class TestDeneubourgNotCarrying:
    def test_no_pickup_from_homogeneous_cluster(self, arena, agent):
        """Same-colour cluster → high f → low P(pickup) → random walk."""
        arena.pellets.clear()
        arena._hash.clear()
        for i in range(8):
            p = Pellet(id=i, x=50.0 + i, y=50.0, colour="red")
            arena.pellets.append(p)
            arena._hash.add(p)

        rng = np.random.RandomState(0)
        action = agent.decide_action(arena, sensor_radius=5.0, rng=rng)
        assert action in (Action.MOVE, Action.TURN_LEFT, Action.TURN_RIGHT)

    def test_random_walk_when_no_pellets_nearby(self, arena, agent):
        arena.pellets.clear()
        arena._hash.clear()

        rng = np.random.RandomState(0)
        action = agent.decide_action(arena, sensor_radius=5.0, rng=rng)
        assert action in (Action.MOVE, Action.TURN_LEFT, Action.TURN_RIGHT)

    def test_high_k1_increases_pickup_likelihood(self, arena):
        """With k1=10.0, P(pickup) ≈ (10/11)² ≈ 0.83 even for f=1.0."""
        agent = DeneubourgAgent(0, 50.0, 50.0, 0.0, k1=10.0, k2=0.3)
        for i in range(4):
            p = Pellet(id=i, x=50.0 + i, y=50.0, colour="red")
            arena.pellets.append(p)
            arena._hash.add(p)

        pickup_count = 0
        for seed in range(100):
            rng = np.random.RandomState(seed)
            action = agent.decide_action(arena, sensor_radius=5.0, rng=rng)
            if action == Action.PICKUP:
                pickup_count += 1
        assert pickup_count > 50, f"Expected ~83 pickups, got {pickup_count}"


class TestDeneubourgCarrying:
    def test_drops_when_near_similar_pellets(self, arena, agent):
        agent.carrying = Pellet(id=-1, x=50.0, y=50.0, colour="red")
        for i in range(6):
            p = Pellet(id=i, x=50.0 + i, y=50.0, colour="red")
            arena.pellets.append(p)
            arena._hash.add(p)

        rng = np.random.RandomState(0)
        action = agent.decide_action(arena, sensor_radius=5.0, rng=rng)
        assert action == Action.DROP

    def test_random_walk_when_no_similar_pellets_nearby(self, arena, agent):
        agent.carrying = Pellet(id=-1, x=50.0, y=50.0, colour="red")
        for i in range(4):
            p = Pellet(id=i, x=50.0 + i, y=50.0, colour="blue")
            arena.pellets.append(p)
            arena._hash.add(p)

        rng = np.random.RandomState(42)
        action = agent.decide_action(arena, sensor_radius=5.0, rng=rng)
        assert action in (Action.MOVE, Action.TURN_LEFT, Action.TURN_RIGHT)


class TestDeneubourgParameters:
    def test_k1_k2_stored(self):
        a = DeneubourgAgent(0, 0.0, 0.0, 0.0, k1=0.2, k2=0.5)
        assert a.k1 == 0.2
        assert a.k2 == 0.5

    def test_inherits_from_base_agent(self):
        a = DeneubourgAgent(5, 10.0, 20.0, 90.0)
        assert a.id == 5
        assert a.x == 10.0
        assert a.y == 20.0
        assert a.heading_deg == 90.0
        assert a.carrying is None
