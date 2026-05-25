"""Pygame live GUI for stigmergic sorting simulation.

## DEBUG-GUIDE (for future AI agents)

PURPOSE:
  Real-time Pygame visualization of the stigmergic sorting simulation.
  Runs a LIVE simulation (not a replay) with interactive controls.

DATA FLOW:
  YAML config → StigmergyGUI.__init__() → _setup_sim()
    → creates Arena, agents, SeedBank from config
    → run() main loop: _run_tick() → _draw_arena() → _draw_hud() → display.flip()

COLOUR LEGEND:
  Embeds a small legend in the bottom-right corner of the arena showing
  each pellet colour present + agents (white triangles).

CONTROLS:
  Space: Pause/Resume  |  R: Reset  |  Q: Quit
  S: Save snapshot PNG |  +/-: Speed (1x-10x)

PELLET_COLOURS dict must match src/viz.py _PELLET_COLOUR_MAP exactly.
  viz.py uses hex strings, gui.py uses RGB tuples — same colours.

USAGE:
  python -m experiments.run_gui --config experiments/configs/baseline.yaml

GOTCHAS:
  - pygame.display.set_mode() blocks until window is created.
  - _w2s() converts world coords to screen pixel coords (accounts for HUD bar).
  - _run_tick() uses two-phase commit: decide all actions first, apply after.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pygame
import yaml

# Add project root to path so src imports work
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.arena import Arena
from src.agents import BaseAgent, Action, create_agents
from src.baseline import DeneubourgAgent
from src.metrics import cluster_purity, cluster_count
from src.rng import SeedBank

# --- Constants ---
WINDOW_W = 800
WINDOW_H = 800
HUD_H = 40
ARENA_PX = WINDOW_W - 20  # 10px margin each side
PELLET_R = 3
AGENT_SIZE = 8
BG_COLOUR = (26, 26, 46)
HUD_BG = (30, 30, 50, 200)
HUD_TEXT = (240, 240, 240)
GRID_COLOUR = (40, 40, 60)
AGENT_COLOUR = (255, 255, 255)
AGENT_EDGE = (180, 180, 200)

PELLET_COLOURS = {
    "red": (231, 76, 60),
    "blue": (52, 152, 219),
    "green": (46, 204, 113),
    "yellow": (241, 196, 15),
    "orange": (230, 126, 34),
    "purple": (155, 89, 182),
}


def _w2s(x: float, y: float, arena_w: float, arena_h: float) -> tuple[int, int]:
    """Convert world coords to screen pixel coords."""
    margin = 10
    sx = margin + int((x / arena_w) * ARENA_PX)
    sy = HUD_H + margin + int((y / arena_h) * ARENA_PX)
    return (sx, sy)


class StigmergyGUI:
    """Real-time Pygame visualization of stigmergic sorting."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.paused = False
        self.speed = 1
        self.tick = 0
        self.total_pickups = 0
        self.total_drops = 0

        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption(
            f"Stigmergy | {config.get('agent_type', 'random')} | "
            f"Seed {config.get('seed', '?')}"
        )
        self.font = pygame.font.SysFont("consolas", 13)
        self.clock = pygame.time.Clock()

        self._setup_sim()

    def _setup_sim(self) -> None:
        """Initialise arena, agents, RNG from config."""
        cfg = self.config
        seed = cfg["seed"]
        self.seed_bank = SeedBank(seed)

        self.arena = Arena(
            cfg["arena"]["width"],
            cfg["arena"]["height"],
            cfg["pellets"]["total"],
            cfg["pellets"]["colours"],
            self.seed_bank,
        )
        self.arena.place_pellets()

        agent_type = cfg.get("agent_type", "random")
        self.sensor_radius = cfg["agents"].get("sensor_radius", 5)
        agents = create_agents(cfg["agents"]["count"], self.arena, self.seed_bank)

        if agent_type == "deneubourg":
            k1 = cfg.get("deneubourg", {}).get("k1", 0.1)
            k2 = cfg.get("deneubourg", {}).get("k2", 0.3)
            self.agents = [
                DeneubourgAgent(a.id, a.x, a.y, a.heading_deg, k1=k1, k2=k2)
                for a in agents
            ]
        else:
            self.agents = agents

        self.agent_rng = self.seed_bank.get_rng("agent_actions")
        self.tick = 0
        self.total_pickups = 0
        self.total_drops = 0

    def _reset(self) -> None:
        """Reset simulation to initial state."""
        self._setup_sim()

    def _run_tick(self) -> None:
        """Execute one simulation tick."""
        actions: list[tuple[BaseAgent, Action]] = []
        pre_carry: list = []

        for agent in self.agents:
            pre_carry.append(agent.carrying)
            action = agent.decide_action(self.arena, self.sensor_radius, self.agent_rng)
            actions.append((agent, action))

        for agent, action in actions:
            agent.apply_action(action, self.arena)

        for idx, (agent, action) in enumerate(actions):
            was_carrying = pre_carry[idx] is not None
            is_carrying = agent.carrying is not None

            if action == Action.PICKUP and not was_carrying and is_carrying:
                self.total_pickups += 1
            elif action == Action.DROP and was_carrying and not is_carrying:
                self.total_drops += 1

        self.tick += 1

    def _draw_hud(self) -> None:
        """Draw HUD bar at top of window."""
        hud_surf = pygame.Surface((WINDOW_W, HUD_H))
        hud_surf.set_alpha(200)
        hud_surf.fill((30, 30, 50))
        self.screen.blit(hud_surf, (0, 0))

        positions = self.arena.get_all_pellet_positions()
        colours = self.arena.get_all_pellet_colours()
        purity = cluster_purity(positions, colours) if positions else 0.0
        clusters = cluster_count(positions) if positions else 0
        carried = sum(1 for a in self.agents if a.carrying is not None)

        parts = [
            f"Purity: {purity:.3f}",
            f"Clusters: {clusters}",
            f"Tick: {self.tick}",
            f"Ground: {len(self.arena.pellets)}",
            f"Carried: {carried}",
            f"Pickups: {self.total_pickups}",
            f"Drops: {self.total_drops}",
            f"Speed: {self.speed}x",
            "Space:Pause  R:Reset  S:Snapshot  +/-:Speed  Q:Quit",
        ]

        x = 8
        for text in parts:
            surf = self.font.render(text, True, HUD_TEXT)
            self.screen.blit(surf, (x, 8))
            x += surf.get_width() + 12

    def _draw_arena(self) -> None:
        """Draw arena grid, pellets, and agents."""
        self.screen.fill(BG_COLOUR)

        aw = self.arena.width
        ah = self.arena.height

        # Grid lines every 10 units
        for i in range(0, int(aw) + 1, 10):
            x1, y1 = _w2s(i, 0, aw, ah)
            x2, y2 = _w2s(i, ah, aw, ah)
            pygame.draw.line(self.screen, GRID_COLOUR, (x1, y1), (x2, y2), 1)
        for j in range(0, int(ah) + 1, 10):
            x1, y1 = _w2s(0, j, aw, ah)
            x2, y2 = _w2s(aw, j, aw, ah)
            pygame.draw.line(self.screen, GRID_COLOUR, (x1, y1), (x2, y2), 1)

        # Arena border
        tl = _w2s(0, 0, aw, ah)
        br = _w2s(aw, ah, aw, ah)
        pygame.draw.rect(
            self.screen,
            (80, 80, 100),
            (tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]),
            2,
        )

        # Pellets
        for p in self.arena.pellets:
            sx, sy = _w2s(p.x, p.y, aw, ah)
            col = PELLET_COLOURS.get(p.colour, (149, 165, 166))
            pygame.draw.circle(self.screen, col, (sx, sy), PELLET_R)

        # Agents
        for a in self.agents:
            sx, sy = _w2s(a.x, a.y, aw, ah)
            angle = math.radians(a.heading_deg)
            # Triangle pointing in heading direction
            cx = AGENT_SIZE * math.cos(angle)
            cy = AGENT_SIZE * math.sin(angle)
            p1 = (sx + cx, sy + cy)
            # Perpendicular for base
            px = -AGENT_SIZE * 0.6 * math.sin(angle)
            py = AGENT_SIZE * 0.6 * math.cos(angle)
            p2 = (sx - cx * 0.5 + px, sy - cy * 0.5 + py)
            p3 = (sx - cx * 0.5 - px, sy - cy * 0.5 - py)

            pygame.draw.polygon(self.screen, AGENT_COLOUR, [p1, p2, p3])
            pygame.draw.polygon(self.screen, AGENT_EDGE, [p1, p2, p3], 1)

            # Carried pellet on agent
            if a.carrying is not None:
                col = PELLET_COLOURS.get(a.carrying.colour, (149, 165, 166))
                pygame.draw.circle(self.screen, col, (sx, sy), 2)

        # --- Colour legend (bottom-right) ---
        self._draw_legend()

    def _draw_legend(self) -> None:
        """Draw colour legend in bottom-right corner."""
        colours_present = sorted(set(p.colour for p in self.arena.pellets))
        has_agents = len(self.agents) > 0
        entries = colours_present + (["__agent__"] if has_agents else [])
        if not entries:
            return

        legend_w = 140
        legend_h = 20 + len(entries) * 18
        margin = 10
        bx = WINDOW_W - legend_w - margin
        by = WINDOW_H - legend_h - margin

        # Background
        leg_surf = pygame.Surface((legend_w, legend_h))
        leg_surf.set_alpha(200)
        leg_surf.fill((20, 20, 40))
        self.screen.blit(leg_surf, (bx, by))
        pygame.draw.rect(self.screen, (60, 60, 80), (bx, by, legend_w, legend_h), 1)

        y = by + 16
        for entry in entries:
            if entry == "__agent__":
                col = AGENT_COLOUR
                label = "Agents"
                pygame.draw.polygon(
                    self.screen, col, [(bx + 14, y - 4), (bx + 22, y), (bx + 14, y + 4)]
                )
            else:
                col = PELLET_COLOURS.get(entry, (149, 165, 166))
                label = entry.capitalize()
                pygame.draw.circle(self.screen, col, (bx + 14, y), 5)

            surf = self.font.render(label, True, HUD_TEXT)
            self.screen.blit(surf, (bx + 28, y - 7))
            y += 18

    def _save_snapshot(self) -> None:
        """Save current frame as PNG."""
        path = os.path.join("docs", "figures", f"gui_snapshot_tick{self.tick}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pygame.image.save(self.screen, path)
        print(f"Snapshot saved: {path}")

    def _handle_events(self) -> bool:
        """Process events. Returns False to quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self._reset()
                elif event.key == pygame.K_s:
                    self._save_snapshot()
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.speed = min(10, self.speed + 1)
                elif event.key == pygame.K_MINUS:
                    self.speed = max(1, self.speed - 1)
        return True

    def run(self) -> None:
        """Main simulation loop."""
        max_ticks = self.config["episode"]["max_ticks"]

        while True:
            if not self._handle_events():
                break

            if not self.paused and self.tick < max_ticks:
                for _ in range(self.speed):
                    if self.tick >= max_ticks:
                        break
                    self._run_tick()

            self._draw_arena()
            self._draw_hud()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Live Pygame GUI for stigmergic sorting."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    gui = StigmergyGUI(config)
    gui.run()


if __name__ == "__main__":
    main()
