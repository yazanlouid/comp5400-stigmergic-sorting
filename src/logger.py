"""Rich, query-friendly logging for stigmergic sorting experiments.

Produces three artefacts alongside the agent CSV:
1. metrics.csv — time-series of aggregate metrics (tick, purity, clusters, etc.)
2. events.csv — discrete pickup/drop events with pellet provenance
3. run.json — config + final metrics sidecar for reproducibility
"""

from __future__ import annotations

import csv
import json
import os
import time

import numpy as np


class ConfigHeader:
    """Embed config as comment lines in CSV files for self-describing output."""

    @staticmethod
    def write(f, config: dict) -> None:
        """Write config as #key=value comment lines."""
        f.write(f"#seed={config.get('seed', 'none')}\n")
        f.write(f"#agent_type={config.get('agent_type', 'random')}\n")
        f.write(f"#arena={config.get('arena', {})}\n")
        f.write(f"#pellets={config.get('pellets', {})}\n")
        f.write(f"#agents={config.get('agents', {})}\n")
        f.write(f"#episode={config.get('episode', {})}\n")
        den = config.get("deneubourg", {})
        if den:
            f.write(f"#deneubourg={den}\n")


class MetricsLogger:
    """Time-series of aggregate metrics, persisted every *interval* ticks.

    Columns: tick, purity, cluster_count, pellets_on_ground, pellets_carried,
              pickup_count, drop_count, wall_time_s
    """

    def __init__(
        self,
        output_dir: str,
        prefix: str,
        interval: int = 100,
        config: dict | None = None,
    ) -> None:
        self.interval: int = interval
        self.path: str = os.path.join(output_dir, f"{prefix}_metrics.csv")
        self._handle = open(self.path, "w", newline="")
        self._writer = csv.writer(self._handle)
        ConfigHeader.write(self._handle, config or {})
        self._writer.writerow(
            [
                "tick",
                "purity",
                "cluster_count",
                "pellets_on_ground",
                "pellets_carried",
                "pickup_count",
                "drop_count",
                "wall_time_s",
            ]
        )
        self._start_time: float = time.perf_counter()
        self._pickup_total: int = 0
        self._drop_total: int = 0

    def record(
        self,
        tick: int,
        purity: float,
        cluster_count: int,
        pellets_on_ground: int,
        pellets_carried: int,
        pickups_this_interval: int,
        drops_this_interval: int,
    ) -> None:
        self._pickup_total += pickups_this_interval
        self._drop_total += drops_this_interval
        wall_time = time.perf_counter() - self._start_time
        self._writer.writerow(
            [
                tick,
                f"{purity:.4f}",
                cluster_count,
                pellets_on_ground,
                pellets_carried,
                self._pickup_total,
                self._drop_total,
                f"{wall_time:.3f}",
            ]
        )

    def close(self) -> None:
        self._handle.flush()
        self._handle.close()


class EventLogger:
    """Discrete pickup/drop events with pellet provenance.

    Columns: tick, agent_id, event_type, pellet_id, pellet_colour, x, y
    """

    def __init__(self, output_dir: str, prefix: str) -> None:
        self.path: str = os.path.join(output_dir, f"{prefix}_events.csv")
        self._handle = open(self.path, "w", newline="")
        self._writer = csv.writer(self._handle)
        self._writer.writerow(
            [
                "tick",
                "agent_id",
                "event_type",
                "pellet_id",
                "pellet_colour",
                "x",
                "y",
            ]
        )

    def pickup(
        self, tick: int, agent_id: int, pellet_id: int, colour: str, x: float, y: float
    ) -> None:
        self._writer.writerow(
            [
                tick,
                agent_id,
                "pickup",
                pellet_id,
                colour,
                f"{x:.2f}",
                f"{y:.2f}",
            ]
        )

    def drop(
        self, tick: int, agent_id: int, pellet_id: int, colour: str, x: float, y: float
    ) -> None:
        self._writer.writerow(
            [
                tick,
                agent_id,
                "drop",
                pellet_id,
                colour,
                f"{x:.2f}",
                f"{y:.2f}",
            ]
        )

    def close(self) -> None:
        self._handle.flush()
        self._handle.close()


class RunSummary:
    """JSON sidecar: config + final metrics + timing."""

    def __init__(self, output_dir: str, prefix: str) -> None:
        self.path: str = os.path.join(output_dir, f"{prefix}_run.json")
        self._start_time: float = time.perf_counter()

    def save(self, config: dict, results: dict) -> None:
        wall_time = time.perf_counter() - self._start_time
        summary = {
            "config": config,
            "results": results,
            "wall_time_s": round(wall_time, 3),
        }
        with open(self.path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
