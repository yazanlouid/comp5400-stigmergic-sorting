"""Matplotlib visualization for stigmergic sorting arena."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .arena import Arena
    from .agents import BaseAgent


# Colour mapping for pellet scatter plot
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
