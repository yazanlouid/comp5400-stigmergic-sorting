"""M5 head-to-head experiment runner — Deneubourg baseline vs evolved neural controllers."""

from __future__ import annotations

import copy
import json
import os
import time
from typing import Any

import numpy as np
import yaml

from src.rng import SeedBank
from src.arena import Arena
from src.agents import BaseAgent, EvolvedAgent, Action
from src.baseline import DeneubourgAgent
from src.controller import NeuralController
from src.metrics import cluster_purity, cluster_count
from src.evolution import evolve

# Config helpers

def load_config(path: str) -> dict[str, Any]:
    """Load YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def config_with_seed(config: dict[str, Any], seed: int) -> dict[str, Any]:
    """Deep-copy config and override the seed value."""
    cfg = copy.deepcopy(config)
    cfg["seed"] = seed
    return cfg

# Genome persistence

def evolve_and_save(
    config_path: str,
    seed: int,
    output_path: str,
) -> dict[str, Any]:
    """Evolve once using GA, save best genome as JSON.

    Returns dict with evolution metadata and genome path.
    """
    config = load_config(config_path)
    config = config_with_seed(config, seed)

    print(f"  Evolving with seed={seed} (config={config_path}) ...")
    t0 = time.time()
    result = evolve(config, seed)
    wall = time.time() - t0

    best_genome = result["best_genome"]
    payload = {
        "seed": seed,
        "genome": best_genome.tolist(),
        "best_fitness_final": float(result["best_fitness_final"]),
        "best_fitness_initial": float(result["best_fitness_initial"]),
        "generations": len(result["best_fitness_per_gen"]),
        "wall_time_s": round(wall, 2),
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(
        f"  Evolution done in {wall:.1f}s. "
        f"Fitness {result['best_fitness_initial']:.4f} -> {result['best_fitness_final']:.4f}"
    )
    print(f"  Genome saved to {output_path}")

    return payload


def load_saved_genome(path: str) -> np.ndarray:
    """Load genome from JSON file. Shape: (149,)"""
    with open(path, "r") as f:
        data = json.load(f)
    return np.array(data["genome"], dtype=np.float64)

# Inline simulation loops

def _run_sim_loop(
    agents: list[BaseAgent],
    arena: Arena,
    sensor_radius: float,
    max_ticks: int,
    metrics_interval: int,
    agent_rng: np.random.RandomState,
) -> dict[str, Any]:
    """Core tick loop shared by baseline and evolved runs.

    Returns purity_history, cluster_history, tick_count, pickup/drop counts.
    """
    purity_history: list[float] = []
    cluster_history: list[int] = []
    total_pickups = 0
    total_drops = 0

    for tick in range(max_ticks):
        actions: list[tuple[BaseAgent, Action]] = []
        pre_carry: list[Any] = []

        # Phase 1: decide actions
        for agent in agents:
            pre_carry.append(agent.carrying)
            action = agent.decide_action(arena, sensor_radius, agent_rng)
            actions.append((agent, action))

        # Phase 2: apply actions
        for agent, action in actions:
            agent.apply_action(action, arena)

        # Track pickups / drops
        for idx, (agent, action) in enumerate(actions):
            was_carrying = pre_carry[idx] is not None
            is_carrying = agent.carrying is not None

            if action == Action.PICKUP and not was_carrying and is_carrying:
                total_pickups += 1
            elif action == Action.DROP and was_carrying and not is_carrying:
                total_drops += 1

        # Periodic metrics
        if tick % metrics_interval == 0:
            positions = arena.get_all_pellet_positions()
            colours = arena.get_all_pellet_colours()
            purity = cluster_purity(positions, colours)
            clusters = cluster_count(positions)
            purity_history.append(purity)
            cluster_history.append(clusters)

    # Final metrics
    positions = arena.get_all_pellet_positions()
    colours = arena.get_all_pellet_colours()
    final_purity = cluster_purity(positions, colours)
    final_clusters = cluster_count(positions)

    return {
        "final_purity": final_purity,
        "final_cluster_count": final_clusters,
        "tick_count": max_ticks,
        "purity_history": purity_history,
        "cluster_history": cluster_history,
        "total_pickups": total_pickups,
        "total_drops": total_drops,
    }


def run_baseline_seed(
    seed: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run Deneubourg baseline for one seed.

    Creates arena, places pellets, wraps agents in DeneubourgAgent,
    runs the simulation loop, returns metrics.
    """
    cfg = config_with_seed(config, seed)
    seed_bank = SeedBank(seed)

    arena = Arena(
        cfg["arena"]["width"],
        cfg["arena"]["height"],
        cfg["pellets"]["total"],
        cfg["pellets"]["colours"],
        seed_bank,
    )
    arena.place_pellets()

    # Create base agents then wrap in DeneubourgAgent
    from src.agents import create_agents

    base_agents = create_agents(cfg["agents"]["count"], arena, seed_bank)

    k1 = cfg.get("deneubourg", {}).get("k1", 0.1)
    k2 = cfg.get("deneubourg", {}).get("k2", 0.3)
    agents: list[BaseAgent] = [
        DeneubourgAgent(a.id, a.x, a.y, a.heading_deg, k1=k1, k2=k2)
        for a in base_agents
    ]

    sensor_radius = cfg["agents"].get("sensor_radius", 5)
    max_ticks = cfg["episode"]["max_ticks"]
    metrics_interval = cfg["episode"].get("metrics_interval", 100)
    agent_rng = seed_bank.get_rng("agent_actions")

    return _run_sim_loop(
        agents, arena, sensor_radius, max_ticks, metrics_interval, agent_rng
    )


def run_evolved_seed(
    seed: int,
    config: dict[str, Any],
    genome: np.ndarray,
) -> dict[str, Any]:
    """Run evolved controller for one seed.

    Creates arena, places pellets, creates EvolvedAgent instances
    sharing ONE NeuralController built from the genome.
    """
    cfg = config_with_seed(config, seed)
    seed_bank = SeedBank(seed)

    arena = Arena(
        cfg["arena"]["width"],
        cfg["arena"]["height"],
        cfg["pellets"]["total"],
        cfg["pellets"]["colours"],
        seed_bank,
    )
    arena.place_pellets()

    # Shared controller from genome
    controller = NeuralController(genome)

    # Create evolved agents with shared controller
    pos_rng = seed_bank.get_rng("agents")
    agents: list[BaseAgent] = []
    for i in range(cfg["agents"]["count"]):
        x = pos_rng.uniform(0.0, arena.width)
        y = pos_rng.uniform(0.0, arena.height)
        heading = pos_rng.uniform(-180.0, 180.0)
        agents.append(EvolvedAgent(i, x, y, heading, controller))

    sensor_radius = cfg["agents"].get("sensor_radius", 5)
    max_ticks = cfg["episode"]["max_ticks"]
    metrics_interval = cfg["episode"].get("metrics_interval", 100)
    agent_rng = seed_bank.get_rng("agent_actions")

    return _run_sim_loop(
        agents, arena, sensor_radius, max_ticks, metrics_interval, agent_rng
    )

# M5 orchestrator

SEEDS: list[int] = [42, 123, 256, 7, 999]
RESULTS_DIR: str = os.path.join("experiments", "results")
GENOME_PATH: str = os.path.join(RESULTS_DIR, "m5_best_genome.json")
AGGREGATED_PATH: str = os.path.join(RESULTS_DIR, "m5_aggregated.json")
BASELINE_CONFIG: str = os.path.join("experiments", "configs", "baseline.yaml")
EVOLUTION_CONFIG: str = os.path.join("experiments", "configs", "fitness_default.yaml")


def run_m5_experiment() -> dict[str, Any]:
    """Full M5 head-to-head: evolve once, test across seeds, aggregate.

    1. Load or evolve genome.
    2. Run baseline and evolved for each seed in SEEDS.
    3. Save aggregated results to m5_aggregated.json.

    Returns dict with baseline_results, evolved_results, genome_path.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Step 1: get genome ---
    if os.path.exists(GENOME_PATH):
        print(f"  Loading existing genome from {GENOME_PATH}")
        genome = load_saved_genome(GENOME_PATH)
    else:
        evolve_and_save(EVOLUTION_CONFIG, seed=0, output_path=GENOME_PATH)
        genome = load_saved_genome(GENOME_PATH)

    # Load configs
    baseline_cfg = load_config(BASELINE_CONFIG)
    evolved_cfg = load_config(EVOLUTION_CONFIG)

    # --- Step 2: run seeds ---
    baseline_results: dict[int, dict[str, Any]] = {}
    evolved_results: dict[int, dict[str, Any]] = {}

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")

        print(f"  Running baseline (seed={seed}) ...")
        t0 = time.time()
        baseline_results[seed] = run_baseline_seed(seed, baseline_cfg)
        print(
            f"    purity={baseline_results[seed]['final_purity']:.4f} "
            f"clusters={baseline_results[seed]['final_cluster_count']} "
            f"time={time.time() - t0:.1f}s"
        )

        print(f"  Running evolved (seed={seed}) ...")
        t0 = time.time()
        evolved_results[seed] = run_evolved_seed(seed, evolved_cfg, genome)
        print(
            f"    purity={evolved_results[seed]['final_purity']:.4f} "
            f"clusters={evolved_results[seed]['final_cluster_count']} "
            f"time={time.time() - t0:.1f}s"
        )

    # --- Step 3: aggregate ---
    def _summarize(results: dict[int, dict[str, Any]]) -> dict[str, Any]:
        purities = [results[s]["final_purity"] for s in SEEDS]
        clusters = [results[s]["final_cluster_count"] for s in SEEDS]
        pickups = [results[s]["total_pickups"] for s in SEEDS]
        drops = [results[s]["total_drops"] for s in SEEDS]
        return {
            "mean_purity": float(np.mean(purities)),
            "std_purity": float(np.std(purities)),
            "mean_clusters": float(np.mean(clusters)),
            "std_clusters": float(np.std(clusters)),
            "mean_pickups": float(np.mean(pickups)),
            "mean_drops": float(np.mean(drops)),
            "per_seed_purity": {str(s): purities[i] for i, s in enumerate(SEEDS)},
            "per_seed_clusters": {str(s): clusters[i] for i, s in enumerate(SEEDS)},
        }

    summary = {
        "seeds": SEEDS,
        "genome_path": GENOME_PATH,
        "baseline": _summarize(baseline_results),
        "evolved": _summarize(evolved_results),
        "baseline_results": {str(s): baseline_results[s] for s in SEEDS},
        "evolved_results": {str(s): evolved_results[s] for s in SEEDS},
    }

    with open(AGGREGATED_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"M5 RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"  Baseline: purity={summary['baseline']['mean_purity']:.4f} "
        f"± {summary['baseline']['std_purity']:.4f}, "
        f"clusters={summary['baseline']['mean_clusters']:.1f}"
    )
    print(
        f"  Evolved:  purity={summary['evolved']['mean_purity']:.4f} "
        f"± {summary['evolved']['std_purity']:.4f}, "
        f"clusters={summary['evolved']['mean_clusters']:.1f}"
    )
    print(
        f"  Delta:    {summary['evolved']['mean_purity'] - summary['baseline']['mean_purity']:+.4f}"
    )
    print(f"  Saved to {AGGREGATED_PATH}")

    return {
        "baseline_results": baseline_results,
        "evolved_results": evolved_results,
        "genome_path": GENOME_PATH,
    }

# CLI entry

def main() -> None:
    """Run M5 experiment from command line."""
    print("=" * 60)
    print("M5: Deneubourg Baseline vs Evolved Neural Controller")
    print("=" * 60)
    run_m5_experiment()


if __name__ == "__main__":
    main()
