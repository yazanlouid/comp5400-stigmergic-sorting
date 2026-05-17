"""Experiment runner — YAML config loading, rich CSV logging, CLI entry point."""

from __future__ import annotations

import csv
import os

import yaml
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .arena import Arena, Pellet
    from .agents import BaseAgent

from .rng import SeedBank
from .arena import Arena
from .agents import BaseAgent, Action, create_agents
from .metrics import cluster_purity, cluster_count
from .logger import ConfigHeader, MetricsLogger, EventLogger, RunSummary


def load_config(path: str) -> dict:
    """Load YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _save_viz_snapshot(
    arena: Arena,
    agents: list[BaseAgent],
    tick: int,
    output_dir: str,
    prefix: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{prefix}_tick{tick}.txt")
    with open(filepath, "w") as f:
        f.write(f"# Snapshot at tick {tick}\n")
        f.write(f"# Pellets: {len(arena.pellets)}\n")
        f.write(f"# Agents: {len(agents)}\n")
        f.write("type,x,y,colour\n")
        for p in arena.pellets:
            f.write(f"pellet,{p.x:.2f},{p.y:.2f},{p.colour}\n")
        for a in agents:
            c = a.carrying.colour if a.carrying else "none"
            f.write(f"agent,{a.x:.2f},{a.y:.2f},{c}\n")


def _early_stop_check(
    purity_history: list[float],
    window: int,
    delta: float,
) -> bool:
    if len(purity_history) < window:
        return False
    recent = purity_history[-window:]
    return (max(recent) - min(recent)) < delta


def run_experiment(config_path: str) -> dict:
    """Run a single experiment from a YAML config.

    Produces:
    - {seed}_{type}.csv — per-agent per-tick log (with config header)
    - {seed}_{type}_metrics.csv — time-series of aggregate metrics
    - {seed}_{type}_events.csv — pickup/drop event log
    - {seed}_{type}_run.json — config + final metrics sidecar

    Returns dict with: csv_path, metrics_path, events_path,
                       run_json, final_purity, final_cluster_count, tick_count
    """
    config = load_config(config_path)

    seed = config["seed"]
    agent_type = config.get("agent_type", "random")
    sensor_radius = config["agents"].get("sensor_radius", 5)
    max_ticks = config["episode"]["max_ticks"]
    early_stop_delta = config["episode"].get("early_stop_delta", 0.01)
    early_stop_window = config["episode"].get("early_stop_window", 2000)
    metrics_interval = config["episode"].get("metrics_interval", 100)

    seed_bank = SeedBank(seed)
    arena = Arena(
        config["arena"]["width"],
        config["arena"]["height"],
        config["pellets"]["total"],
        config["pellets"]["colours"],
        seed_bank,
    )
    arena.place_pellets()

    agents = create_agents(config["agents"]["count"], arena, seed_bank)

    if agent_type == "deneubourg":
        from .baseline import DeneubourgAgent

        k1 = config.get("deneubourg", {}).get("k1", 0.1)
        k2 = config.get("deneubourg", {}).get("k2", 0.3)
        agents = [
            DeneubourgAgent(a.id, a.x, a.y, a.heading_deg, k1=k1, k2=k2) for a in agents
        ]

    agent_rng = seed_bank.get_rng("agent_actions")

    results_dir = os.path.join("experiments", "results")
    os.makedirs(results_dir, exist_ok=True)
    prefix = f"{seed}_{agent_type}"

    csv_path = os.path.join(results_dir, f"{prefix}.csv")
    figures_dir = os.path.join("docs", "figures")
    snapshot_ticks = {0, max_ticks // 2, max_ticks}

    metrics_log = MetricsLogger(
        results_dir, prefix, interval=metrics_interval, config=config
    )
    event_log = EventLogger(results_dir, prefix)
    run_summary = RunSummary(results_dir, prefix)

    do_early_stop = early_stop_window > 0 and early_stop_delta > 0
    purity_history: list[float] = []

    tick_count = 0
    interval_pickups = 0
    interval_drops = 0

    with open(csv_path, "w", newline="") as csvfile:
        ConfigHeader.write(csvfile, config)
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "tick",
                "agent_id",
                "x",
                "y",
                "carrying",
                "action",
            ]
        )

        for tick in range(max_ticks):
            if tick in snapshot_ticks:
                _save_viz_snapshot(arena, agents, tick, figures_dir, prefix)

            actions: list[tuple[BaseAgent, Action]] = []
            pre_carry: list[Pellet | None] = []

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
                    p = agent.carrying
                    event_log.pickup(tick, agent.id, p.id, p.colour, agent.x, agent.y)
                    interval_pickups += 1

                elif action == Action.DROP and was_carrying and not is_carrying:
                    dropped_pellet = pre_carry[idx]
                    event_log.drop(
                        tick,
                        agent.id,
                        dropped_pellet.id,
                        dropped_pellet.colour,
                        agent.x,
                        agent.y,
                    )
                    interval_drops += 1

                writer.writerow(
                    [
                        tick,
                        agent.id,
                        f"{agent.x:.4f}",
                        f"{agent.y:.4f}",
                        1 if is_carrying else 0,
                        Action.names()[action.value],
                    ]
                )

            tick_count = tick + 1

            if tick % metrics_interval == 0:
                positions = arena.get_all_pellet_positions()
                colours = arena.get_all_pellet_colours()
                purity = cluster_purity(positions, colours)
                clusters = cluster_count(positions)
                carried = sum(1 for a in agents if a.carrying is not None)

                metrics_log.record(
                    tick=tick,
                    purity=purity,
                    cluster_count=clusters,
                    pellets_on_ground=len(arena.pellets),
                    pellets_carried=carried,
                    pickups_this_interval=interval_pickups,
                    drops_this_interval=interval_drops,
                )
                interval_pickups = 0
                interval_drops = 0

                purity_history.append(purity)
                if do_early_stop and _early_stop_check(
                    purity_history, early_stop_window, early_stop_delta
                ):
                    break

    if max_ticks not in snapshot_ticks or tick_count < max_ticks:
        _save_viz_snapshot(arena, agents, tick_count - 1, figures_dir, prefix)

    positions = arena.get_all_pellet_positions()
    colours = arena.get_all_pellet_colours()
    final_purity = cluster_purity(positions, colours)
    final_cluster_count = cluster_count(positions)

    results = {
        "csv_path": csv_path,
        "final_purity": final_purity,
        "final_cluster_count": final_cluster_count,
        "tick_count": tick_count,
    }
    run_summary.save(config, results)

    metrics_log.close()
    event_log.close()

    return {
        "csv_path": csv_path,
        "metrics_path": metrics_log.path,
        "events_path": event_log.path,
        "run_json": run_summary.path,
        "final_purity": final_purity,
        "final_cluster_count": final_cluster_count,
        "tick_count": tick_count,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a stigmergic sorting experiment from a YAML config."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    result = run_experiment(args.config)
    print(f"Done. Ticks: {result['tick_count']}")
    print(f"  Agent CSV:  {result['csv_path']}")
    print(f"  Metrics:    {result['metrics_path']}")
    print(f"  Events:     {result['events_path']}")
    print(f"  Run JSON:   {result['run_json']}")
    print(f"  Final purity:   {result['final_purity']:.4f}")
    print(f"  Final clusters: {result['final_cluster_count']}")


if __name__ == "__main__":
    main()
