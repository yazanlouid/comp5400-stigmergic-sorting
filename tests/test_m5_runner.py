"""Tests for experiments/m5_runner.py — config, genome persistence, sim loops."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import numpy as np
import pytest

from experiments.m5_runner import (
    load_config,
    config_with_seed,
    evolve_and_save,
    load_saved_genome,
    run_baseline_seed,
    run_evolved_seed,
    _run_sim_loop,
)
from src.controller import NeuralController
from src.agents import EvolvedAgent, BaseAgent
from src.baseline import DeneubourgAgent
from src.arena import Arena
from src.rng import SeedBank


@pytest.fixture
def baseline_config_path() -> str:
    return os.path.join("experiments", "configs", "baseline.yaml")


@pytest.fixture
def small_config() -> dict[str, Any]:
    """Tiny config for fast tests."""
    return {
        "seed": 42,
        "arena": {"width": 50, "height": 50},
        "pellets": {"total": 20, "colours": 2},
        "agents": {"count": 5, "sensor_radius": 5},
        "episode": {"max_ticks": 50, "metrics_interval": 10},
        "deneubourg": {"k1": 0.1, "k2": 0.3},
    }


@pytest.fixture
def mock_genome() -> np.ndarray:
    return np.random.RandomState(42).randn(NeuralController.TOTAL_PARAMS)


# -----------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------


class TestLoadConfig:
    def test_loads_baseline_yaml(self, baseline_config_path: str) -> None:
        cfg = load_config(baseline_config_path)
        assert cfg["seed"] == 42
        assert cfg["arena"]["width"] == 100
        assert cfg["pellets"]["colours"] == 2

    def test_returns_dict(self, baseline_config_path: str) -> None:
        assert isinstance(load_config(baseline_config_path), dict)


class TestConfigWithSeed:
    def test_overrides_seed(self, small_config: dict[str, Any]) -> None:
        cfg = config_with_seed(small_config, 999)
        assert cfg["seed"] == 999

    def test_deep_copy(self, small_config: dict[str, Any]) -> None:
        original = small_config
        new = config_with_seed(original, 777)
        original["seed"] = 0
        assert new["seed"] == 777
        assert original["seed"] == 0


# -----------------------------------------------------------------------
# Genome persistence
# -----------------------------------------------------------------------


class TestLoadSavedGenome:
    def test_roundtrip(self, mock_genome: np.ndarray, tmp_path) -> None:
        path = str(tmp_path / "genome.json")
        payload = {
            "seed": 42,
            "genome": mock_genome.tolist(),
            "best_fitness_final": 0.85,
            "best_fitness_initial": 0.1,
            "generations": 50,
            "wall_time_s": 12.3,
        }
        with open(path, "w") as f:
            json.dump(payload, f)

        loaded = load_saved_genome(path)
        assert loaded.shape == (NeuralController.TOTAL_PARAMS,)
        assert np.allclose(loaded, mock_genome)

    def test_returns_float64(self, tmp_path) -> None:
        path = str(tmp_path / "g.json")
        with open(path, "w") as f:
            json.dump({"genome": [0.0] * NeuralController.TOTAL_PARAMS}, f)
        assert load_saved_genome(path).dtype == np.float64


class TestEvolveAndSave:
    def test_saves_and_loads_payload(self, tmp_path) -> None:
        """Test save/load roundtrip without running actual evolution."""
        path = str(tmp_path / "saved_genome.json")
        genome = np.random.RandomState(1).randn(NeuralController.TOTAL_PARAMS)
        payload = {
            "seed": 1,
            "genome": genome.tolist(),
            "best_fitness_final": 0.9,
            "best_fitness_initial": 0.2,
            "generations": 10,
            "wall_time_s": 5.0,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

        loaded = load_saved_genome(path)
        assert np.allclose(loaded, genome)
        assert loaded.shape == (NeuralController.TOTAL_PARAMS,)


# -----------------------------------------------------------------------
# Simulation loops
# -----------------------------------------------------------------------


class TestRunSimLoop:
    def test_returns_expected_keys(self, small_config: dict[str, Any]) -> None:
        seed_bank = SeedBank(42)
        arena = Arena(
            small_config["arena"]["width"],
            small_config["arena"]["height"],
            small_config["pellets"]["total"],
            small_config["pellets"]["colours"],
            seed_bank,
        )
        arena.place_pellets()

        agents: list[BaseAgent] = [
            DeneubourgAgent(i, 10.0, 10.0, 0.0, k1=0.1, k2=0.3) for i in range(3)
        ]

        result = _run_sim_loop(
            agents,
            arena,
            sensor_radius=5,
            max_ticks=20,
            metrics_interval=5,
            agent_rng=seed_bank.get_rng("agent_actions"),
        )

        expected_keys = {
            "final_purity",
            "final_cluster_count",
            "tick_count",
            "purity_history",
            "cluster_history",
            "total_pickups",
            "total_drops",
        }
        assert set(result.keys()) == expected_keys
        assert result["tick_count"] == 20
        assert isinstance(result["purity_history"], list)
        assert isinstance(result["cluster_history"], list)
        assert len(result["purity_history"]) > 0


class TestRunBaselineSeed:
    def test_returns_expected_keys(self, small_config: dict[str, Any]) -> None:
        result = run_baseline_seed(42, small_config)

        expected_keys = {
            "final_purity",
            "final_cluster_count",
            "tick_count",
            "purity_history",
            "cluster_history",
            "total_pickups",
            "total_drops",
        }
        assert set(result.keys()) == expected_keys
        assert result["tick_count"] == small_config["episode"]["max_ticks"]

    def test_deterministic(self, small_config: dict[str, Any]) -> None:
        r1 = run_baseline_seed(42, small_config)
        r2 = run_baseline_seed(42, small_config)
        assert r1["final_purity"] == r2["final_purity"]
        assert r1["total_pickups"] == r2["total_pickups"]


class TestRunEvolvedSeed:
    def test_returns_expected_keys(
        self, small_config: dict[str, Any], mock_genome: np.ndarray
    ) -> None:
        result = run_evolved_seed(42, small_config, mock_genome)

        expected_keys = {
            "final_purity",
            "final_cluster_count",
            "tick_count",
            "purity_history",
            "cluster_history",
            "total_pickups",
            "total_drops",
        }
        assert set(result.keys()) == expected_keys
        assert result["tick_count"] == small_config["episode"]["max_ticks"]

    def test_deterministic(
        self, small_config: dict[str, Any], mock_genome: np.ndarray
    ) -> None:
        r1 = run_evolved_seed(42, small_config, mock_genome)
        r2 = run_evolved_seed(42, small_config, mock_genome)
        assert r1["final_purity"] == r2["final_purity"]
        assert r1["total_pickups"] == r2["total_pickups"]
