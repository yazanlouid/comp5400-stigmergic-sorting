"""Tests for evolution module — fitness, selection, mutation, GA loop."""

from __future__ import annotations

import numpy as np
import pytest

from src.rng import SeedBank
from src.controller import NeuralController
from src.evolution import (
    compute_composite_fitness,
    tournament_selection,
    gaussian_mutation,
    create_initial_population,
    FitnessComponents,
)


@pytest.fixture
def rng() -> np.random.RandomState:
    return np.random.RandomState(42)


@pytest.fixture
def minimal_config() -> dict:
    return {
        "seed": 42,
        "arena": {"width": 100, "height": 100},
        "pellets": {"total": 200, "colours": 2},
        "agents": {"count": 20, "sensor_radius": 5},
        "episode": {"max_ticks": 10000, "metrics_interval": 100},
        "evolution": {
            "pop_size": 5,
            "generations": 2,
            "tournament_size": 3,
            "mutation_sigma": 0.1,
            "eval_episodes": 1,
            "eval_max_ticks": 500,
            "fitness_alpha": 0.4,
            "fitness_beta": 0.3,
            "fitness_gamma": 0.2,
            "fitness_delta": 0.1,
            "activity_decay_gen": 25,
            "pickup_bias": 0.5,
        },
    }


def test_population_init(rng):
    pop = create_initial_population(30, rng, pickup_bias=0.5)
    assert len(pop) == 30
    for genome in pop:
        assert len(genome) == NeuralController.TOTAL_PARAMS


def test_composite_fitness_perfect():
    class FakeArena:
        def get_all_pellet_positions(self):
            return [(10, 10) for _ in range(10)]

        def get_all_pellet_colours(self):
            return ["red"] * 10

    fc = compute_composite_fitness(
        FakeArena(),
        initial_clusters=10,
        purity_history=[1.0, 1.0, 1.0],
        pickup_count=400,
        drop_count=400,
        locality_score=1.0,
        generation=0,
        alpha=0.4,
        beta=0.3,
        gamma=0.2,
        delta=0.1,
        epsilon=0.0,
    )
    assert 0 <= fc.purity_terminal <= 1
    assert 0 <= fc.purity_integrated <= 1
    assert 0 <= fc.consolidation <= 1
    assert 0 <= fc.activity <= 1


def test_tournament_selection(rng):
    pop = [rng.randn(149) for _ in range(10)]
    fitnesses = list(range(10))
    winner = tournament_selection(pop, fitnesses, 3, rng)
    assert len(winner) == 149
    assert winner is not pop[-1]


def test_gaussian_mutation(rng):
    parent = np.zeros(149)
    child = gaussian_mutation(parent, 0.1, rng)
    assert not np.array_equal(parent, child)
    assert np.all(np.abs(child) < 0.3)


def test_activity_decay(rng):
    class FakeArena:
        def get_all_pellet_positions(self):
            return [(10, 10) for _ in range(10)]

        def get_all_pellet_colours(self):
            return ["red"] * 10

    fc0 = compute_composite_fitness(
        FakeArena(),
        10,
        [0.5],
        100,
        100,
        0.0,
        generation=0,
        alpha=0,
        beta=0,
        gamma=0,
        delta=0.1,
        epsilon=0,
    )
    fc25 = compute_composite_fitness(
        FakeArena(),
        10,
        [0.5],
        100,
        100,
        0.0,
        generation=25,
        alpha=0,
        beta=0,
        gamma=0,
        delta=0.1,
        epsilon=0,
    )
    assert fc0.total > fc25.total


def test_create_initial_population_shape(rng):
    pop = create_initial_population(5, rng)
    assert len(pop) == 5
    assert all(len(g) == 149 for g in pop)
