"""Genetic algorithm for evolving NeuralController weight vectors.

Composite fitness:
  F = α·purity_terminal + β·purity_integrated + γ·consolidation + δ·activity

Each genome is evaluated over k episodes with different seeds.
Tournament selection, Gaussian mutation, elitism. No crossover.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

import numpy as np

from .rng import SeedBank
from .arena import Arena
from .agents import EvolvedAgent, Action
from .metrics import cluster_purity, cluster_count
from .controller import NeuralController


@dataclass
class FitnessComponents:
    """Breakdown of composite fitness components."""

    purity_terminal: float
    purity_integrated: float
    consolidation: float
    activity: float
    locality: float
    total: float


def compute_composite_fitness(
    arena: Arena,
    initial_clusters: int,
    purity_history: List[float],
    pickup_count: int,
    drop_count: int,
    locality_score: float,
    generation: int,
    alpha: float = 0.20,
    beta: float = 0.15,
    gamma: float = 0.15,
    delta: float = 0.20,
    epsilon: float = 0.30,
    activity_decay_gen: int = 25,
    max_activity: int = 400,
) -> FitnessComponents:
    positions = arena.get_all_pellet_positions()
    colours = arena.get_all_pellet_colours()

    purity_term = cluster_purity(positions, colours)
    integrated_term = float(np.mean(purity_history)) if purity_history else 0.0

    final_clusters = cluster_count(positions)
    consol_term = (
        max(0.0, 1.0 - final_clusters / initial_clusters)
        if initial_clusters > 0
        else 0.0
    )

    activity_term = min(1.0, (pickup_count + drop_count) / max_activity)
    delta_eff = delta * max(0.0, 1.0 - generation / activity_decay_gen)

    locality_term = min(1.0, locality_score)

    total = (
        alpha * purity_term
        + beta * integrated_term
        + gamma * consol_term
        + delta_eff * activity_term
        + epsilon * locality_term
    )

    return FitnessComponents(
        purity_terminal=purity_term,
        purity_integrated=integrated_term,
        consolidation=consol_term,
        activity=activity_term,
        locality=locality_term,
        total=total,
    )


def _run_episode(
    controller: NeuralController,
    arena: Arena,
    agent_count: int,
    sensor_radius: float,
    max_ticks: int,
    metrics_interval: int,
    seed: int,
) -> dict:
    """Run a single evaluation episode inline (no file I/O)."""
    seed_bank = SeedBank(seed)
    agent_rng = seed_bank.get_rng("agent_actions")
    pos_rng = seed_bank.get_rng("agents")

    agents: list[EvolvedAgent] = []
    for i in range(agent_count):
        x = pos_rng.uniform(0.0, arena.width)
        y = pos_rng.uniform(0.0, arena.height)
        heading = pos_rng.uniform(-180.0, 180.0)
        agents.append(EvolvedAgent(i, x, y, heading, controller))

    initial_clusters = cluster_count(arena.get_all_pellet_positions())
    purity_history: list[float] = []
    pickup_count = 0
    drop_count = 0
    locality_count = 0.0
    total_drops = 0

    for tick in range(max_ticks):
        actions: list[tuple[EvolvedAgent, Action]] = []
        pre_carry: list = []

        for agent in agents:
            pre_carry.append(agent.carrying)
            action = agent.decide_action(arena, sensor_radius, agent_rng)
            actions.append((agent, action))

        for agent, action in actions:
            agent.apply_action(action, arena)

        for idx, (agent, action) in enumerate(actions):
            was_carrying = pre_carry[idx] is not None
            is_carrying = agent.carrying is not None

            if action == Action.PICKUP and not was_carrying and is_carrying:
                pickup_count += 1
            elif action == Action.DROP and was_carrying and not is_carrying:
                drop_count += 1
                total_drops += 1
                dropped_colour = pre_carry[idx].colour if pre_carry[idx] else None
                if dropped_colour:
                    nearby = arena.get_pellets_in_radius(
                        agent.x, agent.y, sensor_radius
                    )
                    similar_nearby = sum(
                        1 for p in nearby if p.colour == dropped_colour
                    )
                    if nearby:
                        locality_count += similar_nearby / len(nearby)

        if tick % metrics_interval == 0:
            purity = cluster_purity(
                arena.get_all_pellet_positions(),
                arena.get_all_pellet_colours(),
            )
            purity_history.append(purity)

    locality_score = locality_count / max(1, total_drops)

    return {
        "arena": arena,
        "initial_clusters": initial_clusters,
        "purity_history": purity_history,
        "pickup_count": pickup_count,
        "drop_count": drop_count,
        "locality_score": locality_score,
    }


def evaluate_genome(
    weights: np.ndarray,
    config: dict,
    master_seed: int,
    generation: int = 0,
) -> float:
    """Evaluate a genome over k episodes, return mean fitness."""
    evo = config.get("evolution", {})
    eval_episodes = evo.get("eval_episodes", 3)
    agent_count = config["agents"]["count"]
    sensor_radius = config["agents"].get("sensor_radius", 5)
    eval_max_ticks = evo.get("eval_max_ticks", 5000)
    metrics_interval = config["episode"].get("metrics_interval", 100)

    alpha = evo.get("fitness_alpha", 0.20)
    beta = evo.get("fitness_beta", 0.15)
    gamma_val = evo.get("fitness_gamma", 0.15)
    delta = evo.get("fitness_delta", 0.20)
    epsilon = evo.get("fitness_epsilon", 0.30)
    activity_decay_gen = evo.get("activity_decay_gen", 25)

    controller = NeuralController(weights)
    fitnesses: list[float] = []

    for ep in range(eval_episodes):
        ep_seed = master_seed + ep * 1000
        ep_seed_bank = SeedBank(ep_seed)
        arena = Arena(
            config["arena"]["width"],
            config["arena"]["height"],
            config["pellets"]["total"],
            config["pellets"]["colours"],
            ep_seed_bank,
        )
        arena.place_pellets()

        result = _run_episode(
            controller,
            arena,
            agent_count,
            sensor_radius,
            eval_max_ticks,
            metrics_interval,
            ep_seed,
        )

        fitness = compute_composite_fitness(
            result["arena"],
            result["initial_clusters"],
            result["purity_history"],
            result["pickup_count"],
            result["drop_count"],
            result["locality_score"],
            generation=generation,
            alpha=alpha,
            beta=beta,
            gamma=gamma_val,
            delta=delta,
            epsilon=epsilon,
            activity_decay_gen=activity_decay_gen,
        )
        fitnesses.append(fitness.total)

    return float(np.mean(fitnesses))


def tournament_selection(
    population: list,
    fitnesses: list[float],
    tournament_size: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Standard tournament selection. Returns copy of winner."""
    indices = rng.choice(len(population), size=tournament_size, replace=False)
    winner_idx = indices[np.argmax([fitnesses[i] for i in indices])]
    return population[winner_idx].copy()


def gaussian_mutation(
    parent: np.ndarray,
    sigma: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Per-gene Gaussian mutation."""
    return parent + rng.normal(0, sigma, size=parent.shape)


def create_initial_population(
    pop_size: int,
    rng: np.random.RandomState,
    pickup_bias: float = 0.0,
    move_bias: float = 0.5,
) -> list[np.ndarray]:
    population: list[np.ndarray] = []
    for _ in range(pop_size):
        ctrl = NeuralController.random_init(
            rng, pickup_bias=pickup_bias, move_bias=move_bias
        )
        population.append(np.array(ctrl.to_list(), dtype=np.float64))
    return population


def evolve(
    config: dict,
    seed: int,
) -> dict:
    """Run the full GA evolution loop.

    Returns dict with best_fitness_per_gen, avg_fitness_per_gen,
    best_genome, best_fitness_initial, best_fitness_final, generation_times.
    """
    evo = config.get("evolution", {})
    pop_size = evo.get("pop_size", 30)
    generations = evo.get("generations", 50)
    tournament_size = evo.get("tournament_size", 3)
    mutation_sigma = evo.get("mutation_sigma", 0.1)
    pickup_bias = evo.get("pickup_bias", 0.5)
    move_bias = evo.get("move_bias", 0.5)

    ga_seed_bank = SeedBank(seed)
    ga_rng = ga_seed_bank.get_rng("ga")

    population = create_initial_population(
        pop_size, ga_rng, pickup_bias=pickup_bias, move_bias=move_bias
    )

    fitnesses: list[float] = []
    for genome in population:
        f = evaluate_genome(genome, config, seed, generation=0)
        fitnesses.append(f)

    best_fitness_per_gen: list[float] = []
    avg_fitness_per_gen: list[float] = []
    generation_times: list[float] = []

    best_idx = int(np.argmax(fitnesses))
    best_genome = population[best_idx].copy()
    best_fitness_initial = fitnesses[best_idx]

    for gen in range(generations):
        gen_start = time.perf_counter()

        new_population: list[np.ndarray] = []
        new_fitnesses: list[float] = []

        current_best = int(np.argmax(fitnesses))
        new_population.append(population[current_best].copy())
        new_fitnesses.append(fitnesses[current_best])

        for _ in range(pop_size - 1):
            parent = tournament_selection(
                population, fitnesses, tournament_size, ga_rng
            )
            child = gaussian_mutation(parent, mutation_sigma, ga_rng)
            new_population.append(child)

        for i, genome in enumerate(new_population):
            if i == 0:
                continue
            f = evaluate_genome(genome, config, seed + gen, generation=gen + 1)
            new_fitnesses.append(f)

        population = new_population
        fitnesses = new_fitnesses

        gen_best = max(fitnesses)
        gen_avg = float(np.mean(fitnesses))
        best_fitness_per_gen.append(gen_best)
        avg_fitness_per_gen.append(gen_avg)

        if gen_best > float(best_fitness_per_gen[-1] if best_fitness_per_gen else 0):
            overall_best = int(np.argmax(fitnesses))
            best_genome = population[overall_best].copy()

        generation_times.append(time.perf_counter() - gen_start)

        if (gen + 1) % 10 == 0 or gen == 0:
            print(
                f"  Gen {gen + 1:3d}: "
                f"best={gen_best:.4f} avg={gen_avg:.4f} "
                f"time={generation_times[-1]:.1f}s"
            )

    final_best_idx = int(np.argmax(fitnesses))
    best_genome = population[final_best_idx].copy()
    best_fitness_final = fitnesses[final_best_idx]

    return {
        "best_fitness_per_gen": best_fitness_per_gen,
        "avg_fitness_per_gen": avg_fitness_per_gen,
        "best_genome": best_genome,
        "best_fitness_initial": best_fitness_initial,
        "best_fitness_final": best_fitness_final,
        "generation_times": generation_times,
    }
