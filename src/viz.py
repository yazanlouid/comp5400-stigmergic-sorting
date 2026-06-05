"""Matplotlib visualization for stigmergic sorting arena.

## DEBUG-GUIDE (for future AI agents working on this file)

ARCHITECTURE:
  - This module is HEADLESS (matplotlib Agg backend). It never opens windows.
  - All rendering produces PNG files saved to docs/figures/.
  - Two rendering paths exist:
    1. render_arena()       — legacy, no metadata overlay. Keep for backward compat.
    2. render_arena_metadata() — PRIMARY path. Adds metadata box + colour legend.

DATA FLOW:
  experiment.py (main loop)
    → at snapshot ticks (0, mid, end):
    → computes purity + cluster_count from metrics.py
    → builds metadata dict {seed, agent_type, generation, tick, purity, ...}
    → calls render_arena_metadata(arena, agents, tick, save_path, metadata)
    → saves PNG to docs/figures/{prefix}_tick{N}.png

KEY GOTCHAS:
  - Agg backend MUST be set BEFORE importing pyplot (line 7). If you see
    "cannot import matplotlib.backends.backend_agg" move the use("Agg") call up.
  - _PELLET_COLOUR_MAP is the SINGLE SOURCE OF TRUTH for pellet colours.
    gui.py PELLET_COLOURS and render_snapshots.py _COLOURS must match this.
  - Metadata box uses ax.transAxes coordinates (0-1), NOT data coordinates.
  - plt.close(fig) is called at end of every render function to prevent
    memory leaks during long experiments. NEVER remove it.

COLOUR LEGEND:
  Red dots   = pellets of colour "red"   (#e74c3c)
  Blue dots  = pellets of colour "blue"  (#3498db)
  Black dots = agents (white edge outline)
  Purity in metadata box is colour-coded: green>0.85, yellow>0.75, red otherwise.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend — MUST be before pyplot import
import matplotlib.pyplot as plt
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .arena import Arena
    from .agents import BaseAgent


# SINGLE SOURCE OF TRUTH for pellet colours — gui.py & render_snapshots.py must match
_PELLET_COLOUR_MAP: dict[str, str] = {
    "red": "#e74c3c",
    "blue": "#3498db",
    "green": "#2ecc71",
    "yellow": "#f1c40f",
    "orange": "#e67e22",
    "purple": "#9b59b6",
}


def _pellet_rgba(colour: str) -> str:
    """Return hex colour for a pellet colour name."""
    return _PELLET_COLOUR_MAP.get(colour, "#95a5a6")


def render_arena(
    arena: Arena,
    agents: list[BaseAgent],
    tick: int,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """Render the current arena state as a matplotlib figure.

    Args:
        arena: The arena containing pellets.
        agents: List of agents to overlay.
        tick: Current simulation tick (shown in title).
        save_path: If set, save figure to this path (PNG).
        show: If True, call plt.show() before closing.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # --- Pellets ---
    if arena.pellets:
        px = np.array([p.x for p in arena.pellets])
        py = np.array([p.y for p in arena.pellets])
        colours = [_pellet_rgba(p.colour) for p in arena.pellets]
        ax.scatter(px, py, s=20, c=colours, alpha=0.7, zorder=2)

    # --- Agents ---
    if agents:
        ax_x = np.array([a.x for a in agents])
        ax_y = np.array([a.y for a in agents])
        ax.scatter(
            ax_x,
            ax_y,
            s=50,
            facecolors="black",
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )

    # --- Axes ---
    ax.set_xlim(-5, arena.width + 5)
    ax.set_ylim(-5, arena.height + 5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Tick {tick} | Pellets: {len(arena.pellets)} | Agents: {len(agents)}")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _purity_colour(purity: float) -> str:
    """Return colour for purity value: green >0.85, yellow >0.75, red otherwise."""
    if purity > 0.85:
        return "#2ecc71"
    if purity > 0.75:
        return "#f1c40f"
    return "#e74c3c"


def render_arena_metadata(
    arena: Arena,
    agents: list[BaseAgent],
    tick: int,
    save_path: str,
    metadata: dict[str, str | float | int],
) -> None:
    """Render arena scatter plot with metadata overlay text box.

    Args:
        arena: Arena containing pellets.
        agents: List of agents to overlay.
        tick: Current simulation tick.
        save_path: Path to save the figure (PNG).
        metadata: Dict with keys: seed, agent_type, generation, tick,
                  purity, cluster_count, pellets_on_ground, pellets_carried,
                  wall_time_s.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # --- Pellets ---
    if arena.pellets:
        px = np.array([p.x for p in arena.pellets])
        py = np.array([p.y for p in arena.pellets])
        colours = [_pellet_rgba(p.colour) for p in arena.pellets]
        ax.scatter(px, py, s=20, c=colours, alpha=0.7, zorder=2)

    # --- Agents ---
    if agents:
        ax_x = np.array([a.x for a in agents])
        ax_y = np.array([a.y for a in agents])
        ax.scatter(
            ax_x,
            ax_y,
            s=50,
            facecolors="black",
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )

    # --- Axes ---
    ax.set_xlim(-5, arena.width + 5)
    ax.set_ylim(-5, arena.height + 5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Tick {tick} | Pellets: {len(arena.pellets)} | Agents: {len(agents)}")

    # --- Metadata overlay ---
    purity = float(metadata.get("purity", 0.0))
    purity_col = _purity_colour(purity)
    line_height = 0.018
    x0, y0 = 0.02, 0.97

    items = [
        ("Seed:", metadata.get("seed", "?")),
        ("Agent:", metadata.get("agent_type", "?")),
        ("Gen:", metadata.get("generation", "?")),
        ("Tick:", metadata.get("tick", "?")),
        ("Purity:", f"{purity:.4f}"),
        ("Clusters:", metadata.get("cluster_count", "?")),
        ("Ground:", metadata.get("pellets_on_ground", "?")),
        ("Carried:", metadata.get("pellets_carried", "?")),
        ("Wall:", f"{metadata.get('wall_time_s', 0):.1f}s"),
    ]

    text_lines = [f"{k} {v}" for k, v in items]
    textstr = "\n".join(text_lines)
    props = dict(
        boxstyle="round,pad=0.4",
        facecolor="white",
        alpha=0.85,
        edgecolor="gray",
        linewidth=0.8,
    )
    ax.text(
        x0,
        y0,
        textstr,
        transform=ax.transAxes,
        fontsize=7,
        family="monospace",
        verticalalignment="top",
        bbox=props,
        zorder=4,
    )

    for i, (key, val) in enumerate(items):
        y = y0 - i * line_height
        if key == "Purity:":
            ax.text(
                x0,
                y,
                f"{key} {val}",
                transform=ax.transAxes,
                fontsize=7,
                family="monospace",
                verticalalignment="top",
                color=purity_col,
                fontweight="bold",
                zorder=5,
            )

    # --- Colour legend ---
    legend_items = []
    for colour_name in sorted(set(p.colour for p in arena.pellets)):
        legend_items.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=_pellet_rgba(colour_name),
                markersize=6,
                markeredgecolor="gray",
                markeredgewidth=0.5,
                label=colour_name.capitalize() + " pellets",
            )
        )
    legend_items.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="black",
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label="Agents",
        )
    )
    ax.legend(
        handles=legend_items,
        loc="upper right",
        fontsize=7,
        framealpha=0.85,
        facecolor="white",
        edgecolor="gray",
    )

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_timeline(
    purity_history: list[float],
    cluster_count_history: list[float],
    save_path: str,
) -> None:
    """Render purity and cluster-count timelines as two subplots.

    Args:
        purity_history: Purity value per tick.
        cluster_count_history: Cluster count per tick.
        save_path: Path to save the figure (PNG).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ticks = np.arange(len(purity_history))

    ax1.plot(ticks, purity_history, color="#2ecc71", linewidth=1)
    ax1.set_ylabel("Cluster Purity")
    ax1.set_title("Sorting Progress Over Time")
    ax1.grid(True, alpha=0.3)

    ax2.plot(ticks, cluster_count_history, color="#e74c3c", linewidth=1)
    ax2.set_xlabel("Tick")
    ax2.set_ylabel("Cluster Count")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
