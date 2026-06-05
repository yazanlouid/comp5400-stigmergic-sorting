import time
from dataclasses import dataclass

import numpy as np

from .rng import SeedBank
from .arena import Arena
from .agents import EvolvedAgent, Action
from .metrics import cluster_count,cluster_purity, cluster_sizes
from .controller import NeuralController


@dataclass
class FitnessComponents:
    """Breakdown of fitness components.

    Stores the main component values used for reporting and selection.
    """

    initial_clusters: int
    final_clusters: int
    cluster_reduction: int
    pickup_count: int
    pickup_quality: float
    drop_count: int
    drop_quality: float
    total: float
    move_count: int

def compute_composite_fitness(
    arena: Arena,
    initial_clusters: int,
    pickup_count: int,
    drop_count: int,
    pickup_quality: float,
    drop_quality: float,
    move_count: int
) -> FitnessComponents:
    """Compute the composite fitness score used to evaluate an evolved controller.

    The score combines cluster count, average cluster size, purity, drop quality,
    and an interaction penalty. Higher values indicate stronger sorting behaviour
    under the selected evaluation metrics.

    Returns:
        FitnessComponents object containing the component values and total score.
    """
    positions = arena.get_all_pellet_positions()
    final_clusters = cluster_count(positions)

    cluster_reduction = final_clusters - initial_clusters

    pickup_quality_avg = pickup_quality / max(1, pickup_count)
    drop_quality_avg = drop_quality / max(1, drop_count)

    interaction_count = pickup_count + drop_count


    positions = arena.get_all_pellet_positions()
    colours = arena.get_all_pellet_colours()

    purity = cluster_purity(positions, colours)
    sizes = cluster_sizes(positions)
    avg_cluster_size = float(np.mean(sizes)) if sizes else 0.0

    total = (
        50.0 * final_clusters
        + 100.0 * avg_cluster_size
        + 800.0 * purity
        + 2.0 * drop_quality
        - 0.2 * interaction_count
    )

    return FitnessComponents(
        initial_clusters=initial_clusters,
        final_clusters=final_clusters,
        cluster_reduction=cluster_reduction,
        pickup_count=pickup_count,
        drop_count=drop_count,
        pickup_quality=pickup_quality,
        drop_quality=drop_quality,
        move_count=move_count,
        total=float(total),
    )


def _run_episode(
    controller: NeuralController,
    arena: Arena,
    agent_count: int,
    sensor_radius: float,
    max_ticks: int,
    seed: int,
) -> dict:
    """Run a single evaluation episode inline.

    This runs the actual simulated agents, but returns only the final arena
    and basic action counts. Fitness is calculated after the episode.
    """
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

    move_count = 0
    pickup_count = 0
    drop_count = 0
    pickup_quality = 0.0
    drop_quality = 0.0

    locality_count = 0.0
    total_drops = 0.0


    for _tick in range(max_ticks):
        actions: list[tuple[EvolvedAgent, Action]] = []
        pre_carry: list = []
        pre_nearby: list = []

        # Phase 1: record state BEFORE action, then decide action.
        for agent in agents:
            pre_carry.append(agent.carrying)

            pre_nearby.append(
                arena.get_pellets_in_radius(agent.x, agent.y, sensor_radius)
            )

            action = agent.decide_action(arena, sensor_radius, agent_rng)
            actions.append((agent, action))

        # Phase 2: apply actions.
        for agent, action in actions:
            agent.apply_action(action, arena)

        # Phase 3: count successful pickup/drop and quality.
        for idx, (agent, action) in enumerate(actions):
            before_carry = pre_carry[idx]
            after_carry = agent.carrying

            was_carrying = before_carry is not None
            is_carrying = after_carry is not None

            # Count MOVE command.
            if action == Action.MOVE:
                move_count += 1

            # Successful PICKUP only
            if action == Action.PICKUP and not was_carrying and is_carrying:
                pickup_count += 1

                picked_pellet = after_carry
                picked_colour = picked_pellet.colour

                nearby_before = pre_nearby[idx]

                # Exclude the pellet that was just picked.
                nearby_other = [
                    p for p in nearby_before
                    if p.id != picked_pellet.id
                ]

                if nearby_other:
                    same_colour_count = sum(
                        1 for p in nearby_other
                        if p.colour == picked_colour
                    )

                    same_colour_fraction = same_colour_count / len(nearby_other)

                    # Good pickup = low similarity / sparse area.
                    low_similarity_score = 1.0 - same_colour_fraction
                    sparse_score = 1.0 - min(1.0, len(nearby_other) / 8.0)

                    pickup_quality += (
                        0.5 * low_similarity_score
                        + 0.5 * sparse_score
                    )
                else:
                    # Isolated pickup is considered good.
                    pickup_quality += 1.0

            # Successful DROP only
            elif action == Action.DROP and was_carrying and not is_carrying:
                drop_count += 1
                total_drops += 1

                dropped_pellet = before_carry
                dropped_colour = dropped_pellet.colour

                nearby_after = arena.get_pellets_in_radius(
                    agent.x,
                    agent.y,
                    sensor_radius,
                )

                # Exclude the pellet that was just dropped.
                nearby_other = [
                    p for p in nearby_after
                    if p.id != dropped_pellet.id
                ]

                if nearby_other:
                    same_colour_count = sum(
                        1 for p in nearby_other
                        if p.colour == dropped_colour
                    )

                    same_colour_fraction = same_colour_count / len(nearby_other)

                    # Locality score.
                    locality_count += same_colour_fraction

                    # Good drop = dense same-colour area.
                    density_score = min(1.0, len(nearby_other) / 8.0)

                    drop_quality += same_colour_fraction * density_score

    return {
        "arena": arena,
        "initial_clusters": initial_clusters,
        "pickup_count": pickup_count,
        "drop_count": drop_count,
        "pickup_quality": pickup_quality,
        "drop_quality": drop_quality,
        "move_count": move_count
    }


def evaluate_genome(
    weights: np.ndarray,
    config: dict,
    master_seed: int,
    generation: int = 0,
) -> float:
    """Evaluate one genome over k episodes and return mean fitness.

    In this version:
        fitness = mean final cluster_count across evaluation episodes.

    Important:
        The same master_seed should be used across generations if you want fair
        comparison. The evolve() function below does that.
    """
    evo = config.get("evolution", {})

    eval_episodes = evo.get("eval_episodes", 3)
    agent_count = config["agents"]["count"]
    sensor_radius = config["agents"].get("sensor_radius", 5)
    eval_max_ticks = evo.get("eval_max_ticks", 5000)

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
            controller=controller,
            arena=arena,
            agent_count=agent_count,
            sensor_radius=sensor_radius,
            max_ticks=eval_max_ticks,
            seed=ep_seed,
        )


        fitness = compute_composite_fitness(
            result["arena"],
            result["initial_clusters"],
            result["pickup_count"],
            result["drop_count"],
            result["pickup_quality"],
            result["drop_quality"],
            result["move_count"]
        )
        fitnesses.append(fitness.total)

    return float(np.mean(fitnesses))


def tournament_selection(
    population: list[np.ndarray],
    fitnesses: list[float],
    tournament_size: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Standard tournament selection.

    Randomly sample tournament_size genomes and return a copy of the one with
    highest fitness.
    """
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
    """Create initial random population of NeuralController genomes."""
    population: list[np.ndarray] = []

    for _ in range(pop_size):
        ctrl = NeuralController.random_init(
            rng,
            pickup_bias=pickup_bias,
            move_bias=move_bias,
        )
        population.append(np.array(ctrl.to_list(), dtype=np.float64))

    return population


def evolve(
    config: dict,
    seed: int,
) -> dict:
    """Run the full genetic algorithm evolution loop.

    The algorithm evolves neural-controller genomes using tournament selection,
    elitism, and Gaussian mutation. Candidate genomes are evaluated through
    simulation episodes and scored using the composite fitness function.

    Returns:
        Dictionary containing the best genome, initial and final best fitness,
        and per-generation fitness history.
    """
    evo = config.get("evolution", {})

    pop_size = evo.get("pop_size", 30)
    generations = evo.get("generations", 50)
    tournament_size = evo.get("tournament_size", 3)
    mutation_sigma = evo.get("mutation_sigma", 0.1)
    pickup_bias = evo.get("pickup_bias", 0.5)
    move_bias = evo.get("move_bias", 0.5)

    if tournament_size > pop_size:
        raise ValueError(
            f"tournament_size ({tournament_size}) cannot be larger than "
            f"pop_size ({pop_size})."
        )

    ga_seed_bank = SeedBank(seed)
    ga_rng = ga_seed_bank.get_rng("ga")

    population = create_initial_population(
        pop_size,
        ga_rng,
        pickup_bias=pickup_bias,
        move_bias=move_bias,
    )

    # Initial evaluation
    fitnesses: list[float] = []
    for genome in population:
        f = evaluate_genome(genome, config, seed, generation=0)
        fitnesses.append(f)

    best_fitness_per_gen: list[float] = []
    avg_fitness_per_gen: list[float] = []
    generation_times: list[float] = []

    best_idx = int(np.argmax(fitnesses))
    best_genome = population[best_idx].copy()
    best_fitness_initial = float(fitnesses[best_idx])
    overall_best_fitness = best_fitness_initial

    print(
        f"  Initial: best={best_fitness_initial:.4f} "
        f"avg={float(np.mean(fitnesses)):.4f}"
    )

    # Evolution loop
    for gen in range(generations):
        gen_start = time.perf_counter()

        new_population: list[np.ndarray] = []
        new_fitnesses: list[float] = []

        # Elitism: keep the current best genome.
        current_best_idx = int(np.argmax(fitnesses))
        elite = population[current_best_idx].copy()
        elite_fitness = float(fitnesses[current_best_idx])

        new_population.append(elite)
        new_fitnesses.append(elite_fitness)

        # Create mutated children.
        for _ in range(pop_size - 1):
            parent = tournament_selection(
                population,
                fitnesses,
                tournament_size,
                ga_rng,
            )
            child = gaussian_mutation(parent, mutation_sigma, ga_rng)
            new_population.append(child)

        # Evaluate children.
        #
        # NOTE:
        # We keep master_seed as `seed`, not `seed + gen`.
        # This makes every generation use the same evaluation seeds, so changes
        # in fitness are less likely to be caused by lucky/easy seeds.
        for i, genome in enumerate(new_population):
            if i == 0:
                # Elite already has a valid fitness because evaluation seeds are fixed.
                continue

            f = evaluate_genome(genome, config, seed, generation=gen + 1)
            new_fitnesses.append(f)

        population = new_population
        fitnesses = new_fitnesses

        gen_best = float(max(fitnesses))
        gen_avg = float(np.mean(fitnesses))

        best_fitness_per_gen.append(gen_best)
        avg_fitness_per_gen.append(gen_avg)

        # Correct global best tracking.
        if gen_best > overall_best_fitness:
            overall_best_idx = int(np.argmax(fitnesses))
            best_genome = population[overall_best_idx].copy()
            overall_best_fitness = gen_best

        generation_times.append(time.perf_counter() - gen_start)

        print(
            f"  Gen {gen + 1:3d}: "
            f"best={gen_best:.4f} avg={gen_avg:.4f} "
            f"overall_best={overall_best_fitness:.4f} "
            f"time={generation_times[-1]:.1f}s"
        )

    best_fitness_final = overall_best_fitness

    return {
        "best_fitness_per_gen": best_fitness_per_gen,
        "avg_fitness_per_gen": avg_fitness_per_gen,
        "best_genome": best_genome,
        "best_fitness_initial": best_fitness_initial,
        "best_fitness_final": best_fitness_final,
        "generation_times": generation_times,
    }
