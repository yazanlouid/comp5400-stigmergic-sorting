import json
import math
import os
import sys

import numpy as np
import pygame
import yaml
from sklearn.cluster import DBSCAN


# Add project root to path so src imports work when run from experiments/.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.arena import Arena
from src.agents import BaseAgent, Action, create_agents, EvolvedAgent
from src.baseline import DeneubourgAgent
from src.metrics import cluster_purity, cluster_count
from src.rng import SeedBank


# Constants
WINDOW_W = 800
WINDOW_H = 600
HUD_H = 40

# Keep the arena fully inside the window.
ARENA_PX = min(WINDOW_W - 20, WINDOW_H - HUD_H - 20)

PELLET_R = 3
AGENT_SIZE = 8
BG_COLOUR = (26, 26, 46)
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

CLUSTER_OUTLINE_COLOURS = [
    (255, 255, 255),
    (255, 220, 120),
    (120, 220, 255),
    (180, 255, 180),
    (255, 160, 220),
    (210, 180, 255),
    (255, 180, 120),
]


def _w2s(x: float, y: float, arena_w: float, arena_h: float) -> tuple[int, int]:
    """Convert world coordinates to screen pixel coordinates."""
    margin = 10
    sx = margin + int((x / arena_w) * ARENA_PX)
    sy = HUD_H + margin + int((y / arena_h) * ARENA_PX)
    return sx, sy


class StigmergyGUI:
    """Real-time Pygame visualisation of stigmergic sorting."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.paused = False
        self.speed = 1
        self.tick = 0
        self.total_pickups = 0
        self.total_drops = 0

        # Used only for agent_type: evolved.
        # The controller is loaded/evolved once, then reused on reset.
        self.evolved_controller = None
        self.evolution_result: dict | None = None
        self.genome_source: str = "none"

        # Cluster overlay settings.
        # These defaults match metrics.cluster_count() / cluster_purity().
        gui_cfg = self.config.get("gui", {})
        self.show_clusters = bool(gui_cfg.get("show_clusters", True))
        self.cluster_eps = float(gui_cfg.get("cluster_eps", 2.0))
        self.cluster_min_samples = int(gui_cfg.get("cluster_min_samples", 2))

        self.debug_action_counts = bool(gui_cfg.get("debug_action_counts", False))
        self.action_counts = {name: 0 for name in Action.names()}

        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption(
            f"Stigmergy | {config.get('agent_type', 'random')} | "
            f"Seed {config.get('seed', '?')}"
        )
        self.font = pygame.font.SysFont("consolas", 13)
        self.small_font = pygame.font.SysFont("consolas", 11)
        self.clock = pygame.time.Clock()

        self._setup_sim()

    # Genome helpers

    def _default_genome_path(self) -> str:
        """Default path used when save/load path is not supplied."""
        return "experiments/results/gui_best_genome.json"

    def _load_genome_from_json(self, genome_path: str):
        """Load a saved genome JSON and return a NeuralController."""
        from src.controller import NeuralController

        if not os.path.exists(genome_path):
            raise FileNotFoundError(
                f"Saved genome file not found: {genome_path}\n"
                "Either set evolution.use_saved_genome: false to evolve fresh, "
                "or provide a valid evolution.load_genome_path."
            )

        print("=" * 60)
        print(f"Loading saved genome from: {genome_path}")
        print("=" * 60)

        with open(genome_path, "r") as f:
            payload = json.load(f)

        if "genome" not in payload:
            raise KeyError(
                f"Genome JSON does not contain key 'genome': {genome_path}"
            )

        genome = np.array(payload["genome"], dtype=np.float64)

        # NeuralController will also validate shape == (149,).
        controller = NeuralController(genome)

        self.evolution_result = payload
        self.genome_source = f"loaded: {genome_path}"

        print("Loaded genome successfully.")
        print(f"Genome length: {genome.shape[0]}")
        if "best_fitness_final" in payload:
            print(f"Saved final fitness: {float(payload['best_fitness_final']):.4f}")
        print("=" * 60)

        return controller

    def _save_genome_to_json(self, genome: np.ndarray, save_path: str) -> None:
        """Save best genome and GA metadata to JSON."""
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        result = self.evolution_result or {}
        payload = {
            "seed": self.config["seed"],
            "genome": genome.tolist(),
            "best_fitness_initial": float(result.get("best_fitness_initial", 0.0)),
            "best_fitness_final": float(result.get("best_fitness_final", 0.0)),
            "generations": len(result.get("best_fitness_per_gen", [])),
            "best_fitness_per_gen": [
                float(x) for x in result.get("best_fitness_per_gen", [])
            ],
            "avg_fitness_per_gen": [
                float(x) for x in result.get("avg_fitness_per_gen", [])
            ],
            "generation_times": [
                float(x) for x in result.get("generation_times", [])
            ],
        }

        with open(save_path, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"Saved best genome to: {save_path}")

    def _evolve_fresh_controller(self):
        """Run GA once and return the final evolved NeuralController."""
        from src.controller import NeuralController
        from src.evolution import evolve

        seed = self.config["seed"]
        evo_cfg = self.config.get("evolution", {})
        save_path = evo_cfg.get("save_genome_path", self._default_genome_path())

        print("=" * 60)
        print("Starting fresh GA evolution for GUI...")
        print("The pygame window may not respond until evolution finishes.")
        print("This is expected because real-time GA display is disabled here.")
        print("=" * 60)

        self.evolution_result = evolve(self.config, seed)
        best_genome = self.evolution_result["best_genome"]

        self._save_genome_to_json(best_genome, save_path)

        controller = NeuralController(best_genome)
        self.genome_source = f"fresh evolution, saved: {save_path}"

        initial = float(self.evolution_result.get("best_fitness_initial", 0.0))
        final = float(self.evolution_result.get("best_fitness_final", 0.0))
        generations = len(self.evolution_result.get("best_fitness_per_gen", []))

        print("=" * 60)
        print("GA evolution finished.")
        print(f"Generations: {generations}")
        print(f"Initial best fitness: {initial:.4f}")
        print(f"Final best fitness:   {final:.4f}")
        print("Starting GUI simulation with final evolved controller.")
        print("=" * 60)

        return controller

    def _get_evolved_controller(self):
        """Either load a saved genome or run fresh GA, based on YAML config."""
        if self.evolved_controller is not None:
            return self.evolved_controller

        evo_cfg = self.config.get("evolution", {})
        use_saved = bool(evo_cfg.get("use_saved_genome", False))

        if use_saved:
            genome_path = evo_cfg.get(
                "load_genome_path",
                evo_cfg.get("save_genome_path", self._default_genome_path()),
            )
            self.evolved_controller = self._load_genome_from_json(genome_path)
        else:
            self.evolved_controller = self._evolve_fresh_controller()

        return self.evolved_controller

    # Simulation setup

    def _setup_sim(self) -> None:
        """Initialise arena, agents, and RNG from config."""
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

        # Create initial random positions/headings once, then wrap them into
        # the selected agent type.
        base_agents = create_agents(cfg["agents"]["count"], self.arena, self.seed_bank)

        if agent_type == "deneubourg":
            k1 = cfg.get("deneubourg", {}).get("k1", 0.1)
            k2 = cfg.get("deneubourg", {}).get("k2", 0.3)
            self.agents: list[BaseAgent] = [
                DeneubourgAgent(a.id, a.x, a.y, a.heading_deg, k1=k1, k2=k2)
                for a in base_agents
            ]

        elif agent_type == "evolved":
            controller = self._get_evolved_controller()
            self.agents = [
                EvolvedAgent(a.id, a.x, a.y, a.heading_deg, controller)
                for a in base_agents
            ]

        else:
            # random/default controller
            self.agents = base_agents

        self.agent_rng = self.seed_bank.get_rng("agent_actions")
        self.tick = 0
        self.total_pickups = 0
        self.total_drops = 0
        self.action_counts = {name: 0 for name in Action.names()}

        if self.agents:
            print(f"Actual agent class: {type(self.agents[0]).__name__}")
            if agent_type == "evolved":
                print(f"Genome source: {self.genome_source}")

    def _reset(self) -> None:
        """Reset simulation state.

        For evolved mode, this reuses the same loaded/evolved controller.
        It does not re-run GA every time R is pressed.
        """
        self._setup_sim()

    # Tick/update logic

    def _run_tick(self) -> None:
        """Execute one simulation tick using two-phase commit."""
        actions: list[tuple[BaseAgent, Action]] = []
        pre_carry: list = []

        # Phase 1: every agent decides using the same world state.
        for agent in self.agents:
            pre_carry.append(agent.carrying)
            action = agent.decide_action(self.arena, self.sensor_radius, self.agent_rng)
            actions.append((agent, action))

            if self.debug_action_counts:
                self.action_counts[Action.names()[action.value]] += 1

        # Phase 2: apply actions after all decisions are made.
        for agent, action in actions:
            agent.apply_action(action, self.arena)

        # Track successful pickup/drop events.
        for idx, (agent, action) in enumerate(actions):
            was_carrying = pre_carry[idx] is not None
            is_carrying = agent.carrying is not None

            if action == Action.PICKUP and not was_carrying and is_carrying:
                self.total_pickups += 1
            elif action == Action.DROP and was_carrying and not is_carrying:
                self.total_drops += 1

        self.tick += 1

        if self.debug_action_counts and self.tick % 500 == 0:
            print(f"Tick {self.tick} action counts: {self.action_counts}")

    # Drawing

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
        agent_type = self.config.get("agent_type", "random")

        parts = [
            f"Agent: {agent_type}",
            f"Purity: {purity:.3f}",
            f"Clusters: {clusters}",
            f"Overlay: {'ON' if self.show_clusters else 'OFF'}",
            f"Tick: {self.tick}",
            f"Ground: {len(self.arena.pellets)}",
            f"Carried: {carried}",
            f"Pickups: {self.total_pickups}",
            f"Drops: {self.total_drops}",
            f"Speed: {self.speed}x",
        ]

        if agent_type == "evolved" and self.evolution_result is not None:
            final_fit = self.evolution_result.get("best_fitness_final", None)
            if final_fit is not None:
                parts.append(f"GA Fit: {float(final_fit):.3f}")

        parts.append("Space:Pause R:Reset C:Clusters S:Snapshot +/-:Speed Q:Quit")

        x = 8
        for text in parts:
            surf = self.font.render(text, True, HUD_TEXT)
            self.screen.blit(surf, (x, 8))
            x += surf.get_width() + 12
            if x > WINDOW_W - 20:
                break

    def _draw_cluster_overlay(self) -> None:
        """Draw DBSCAN cluster outlines using the same default logic as metrics.py.

        This is a visual guide only. DBSCAN clusters can be irregular/chained,
        so the drawn circle is an approximate highlight around each cluster.
        """
        positions = self.arena.get_all_pellet_positions()

        if len(positions) < self.cluster_min_samples:
            return

        X = np.array(positions, dtype=np.float64)
        labels = DBSCAN(
            eps=self.cluster_eps,
            min_samples=self.cluster_min_samples,
        ).fit(X).labels_

        aw = self.arena.width
        ah = self.arena.height

        cluster_labels = sorted(label for label in set(labels) if label != -1)

        for order, label in enumerate(cluster_labels):
            cluster_points = X[labels == label]

            if len(cluster_points) == 0:
                continue

            cx = float(np.mean(cluster_points[:, 0]))
            cy = float(np.mean(cluster_points[:, 1]))

            distances = np.sqrt(
                (cluster_points[:, 0] - cx) ** 2
                + (cluster_points[:, 1] - cy) ** 2
            )

            # Add a little padding so pellets sit inside the outline.
            radius_world = float(np.max(distances)) + 2.0

            sx, sy = _w2s(cx, cy, aw, ah)

            # Convert world units to screen pixels. For non-square arenas, use
            # the smaller scale so the circle stays conservative.
            scale_x = ARENA_PX / aw
            scale_y = ARENA_PX / ah
            radius_px = int(radius_world * min(scale_x, scale_y))
            radius_px = max(6, radius_px)

            colour = CLUSTER_OUTLINE_COLOURS[order % len(CLUSTER_OUTLINE_COLOURS)]

            pygame.draw.circle(
                self.screen,
                colour,
                (sx, sy),
                radius_px,
                2,
            )

            label_text = self.small_font.render(
                f"C{label} n={len(cluster_points)}",
                True,
                colour,
            )

            self.screen.blit(label_text, (sx + 5, sy - 12))

    def _draw_arena(self) -> None:
        """Draw arena grid, pellets, cluster overlay, agents, and legend."""
        self.screen.fill(BG_COLOUR)

        aw = self.arena.width
        ah = self.arena.height

        # Grid lines every 10 units.
        for i in range(0, int(aw) + 1, 10):
            x1, y1 = _w2s(i, 0, aw, ah)
            x2, y2 = _w2s(i, ah, aw, ah)
            pygame.draw.line(self.screen, GRID_COLOUR, (x1, y1), (x2, y2), 1)

        for j in range(0, int(ah) + 1, 10):
            x1, y1 = _w2s(0, j, aw, ah)
            x2, y2 = _w2s(aw, j, aw, ah)
            pygame.draw.line(self.screen, GRID_COLOUR, (x1, y1), (x2, y2), 1)

        # Arena border.
        tl = _w2s(0, 0, aw, ah)
        br = _w2s(aw, ah, aw, ah)
        pygame.draw.rect(
            self.screen,
            (80, 80, 100),
            (tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]),
            2,
        )

        # Pellets.
        for pellet in self.arena.pellets:
            sx, sy = _w2s(pellet.x, pellet.y, aw, ah)
            col = PELLET_COLOURS.get(pellet.colour, (149, 165, 166))
            pygame.draw.circle(self.screen, col, (sx, sy), PELLET_R)

        # DBSCAN cluster overlay.
        if self.show_clusters:
            self._draw_cluster_overlay()

        # Agents.
        for agent in self.agents:
            sx, sy = _w2s(agent.x, agent.y, aw, ah)
            angle = math.radians(agent.heading_deg)

            # Triangle pointing in heading direction.
            cx = AGENT_SIZE * math.cos(angle)
            cy = AGENT_SIZE * math.sin(angle)
            p1 = (sx + cx, sy + cy)

            # Perpendicular for base.
            px = -AGENT_SIZE * 0.6 * math.sin(angle)
            py = AGENT_SIZE * 0.6 * math.cos(angle)
            p2 = (sx - cx * 0.5 + px, sy - cy * 0.5 + py)
            p3 = (sx - cx * 0.5 - px, sy - cy * 0.5 - py)

            pygame.draw.polygon(self.screen, AGENT_COLOUR, [p1, p2, p3])
            pygame.draw.polygon(self.screen, AGENT_EDGE, [p1, p2, p3], 1)

            # Show carried pellet on top of agent.
            if agent.carrying is not None:
                col = PELLET_COLOURS.get(agent.carrying.colour, (149, 165, 166))
                pygame.draw.circle(self.screen, col, (sx, sy), 3)

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

        # Background.
        leg_surf = pygame.Surface((legend_w, legend_h))
        leg_surf.set_alpha(200)
        leg_surf.fill((20, 20, 40))
        self.screen.blit(leg_surf, (bx, by))
        pygame.draw.rect(self.screen, (60, 60, 80), (bx, by, legend_w, legend_h), 1)

        y = by + 16
        for entry in entries:
            if entry == "__agent__":
                label = "Agents"
                pygame.draw.polygon(
                    self.screen,
                    AGENT_COLOUR,
                    [(bx + 14, y - 4), (bx + 22, y), (bx + 14, y + 4)],
                )
            else:
                label = entry.capitalize()
                col = PELLET_COLOURS.get(entry, (149, 165, 166))
                pygame.draw.circle(self.screen, col, (bx + 14, y), 5)

            surf = self.font.render(label, True, HUD_TEXT)
            self.screen.blit(surf, (bx + 28, y - 7))
            y += 18

    # Input / lifecycle

    def _save_snapshot(self) -> None:
        """Save current frame as PNG."""
        path = os.path.join("docs", "figures", f"gui_snapshot_tick{self.tick}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pygame.image.save(self.screen, path)
        print(f"Snapshot saved: {path}")

    def _handle_events(self) -> bool:
        """Process pygame events. Return False to quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return False
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self._reset()
                elif event.key == pygame.K_c:
                    self.show_clusters = not self.show_clusters
                elif event.key == pygame.K_s:
                    self._save_snapshot()
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.speed = min(10, self.speed + 1)
                elif event.key == pygame.K_MINUS:
                    self.speed = max(1, self.speed - 1)

        return True

    def run(self) -> None:
        """Main pygame loop."""
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
