# Environment Specification

Locked: 2026-05-16. Supersedes all prior design discussions.

## Arena

| Parameter | Value | Rationale |
|---|---|---|
| Dimensions | 100 × 100 (continuous, float) | Large enough for meaningful cluster formation; small enough for clean visualization. Deneubourg's original 30×30 discrete was too cramped for 200 pellets. Continuous space with discretized sensing gives smooth movement + efficient spatial hashing. |
| Boundary | Hard walls at [0, 100] × [0, 100] | Agents clamp to boundary on movement. No wrapping — matches biological analogy (ants on a bounded surface). |
| Coordinate system | Origin (0, 0) = bottom-left, (100, 100) = top-right | Standard Cartesian for matplotlib compatibility. |

## Pellets

| Parameter | Value | Rationale |
|---|---|---|
| Total count | 200 | Canonical Deneubourg number. 1 pellet per 50 cells — sparse enough for non-trivial sorting, dense enough for visual clusters. |
| Colours | 2 (red, blue) | Baseline config. Extensions (M6) test 3+ colours. |
| Per colour | 100 each | Balanced classes. |
| Placement | Uniform random within arena | Deterministic via SeedBank. No overlap guarantee needed (continuous space, probability of exact overlap = 0). |
| Size (visual) | Radius 1.5 in render | Large enough to distinguish, small enough to show density. |

## Agents

| Parameter | Value | Rationale |
|---|---|---|
| Count | 20 | 1:10 agent-to-pellet ratio. Enough parallelism for emergent behaviour; few enough that inter-agent collisions don't dominate. |
| Initial position | Uniform random within arena | Deterministic via SeedBank. |
| Initial heading | Uniform random in [-180°, 180°] | No preferred direction at start. |
| Sensor radius | r = 5 cells | "Local" sensing — ~0.8% of arena area (π×5²/10000) but ~1-3 pellets on average given 200 pellets. r > 10 would let agents see too much and trivialize the task. |
| Sensor output | `{red_density: float, blue_density: float}` | Normalised count of pellets per colour within sensor radius. Used by both Deneubourg baseline (f computation) and evolved controller (NN input). |
| Carrying capacity | 1 pellet (binary: carrying/not carrying) | Matches biological model — one ant carries one brood item. |

## Action Space

| Action | Effect | Rationale |
|---|---|---|
| `MOVE` | Move forward 1 unit in current heading | Primary locomotion. |
| `TURN_LEFT` | Rotate heading by -30° | Directional control. |
| `TURN_RIGHT` | Rotate heading by +30° | Directional control. |
| `PICKUP` | Pick up pellet at current position (if not carrying, pellet present within distance 1.0) | Stigmergic interaction. |
| `DROP` | Drop carried pellet at current position (if carrying) | Stigmergic interaction. |

Discrete 5-action space chosen over continuous heading for: (1) simpler NN output (5-way softmax), (2) well-behaved GA search space, (3) cleaner probabilistic decision-making in Deneubourg baseline.

## Episode

| Parameter | Value | Rationale |
|---|---|---|
| Max ticks | 10,000 | Deneubourg showed convergence within this range for equivalent scales. 20 agents × 10,000 ticks = 200,000 total actions — enough for each pellet to be moved ~10 times. |
| Early stop | Plateau of 2,000 consecutive ticks with Δ purity < 0.01 | Saves compute on converged episodes. |
| Tick semantics | Synchronous update — all agents compute actions at t, all update to t+1 simultaneously | Avoids ordering artifacts. Matches discrete-time simulation standard. |

## RNG Seeding Policy

```
Master seed (int) → SeedBank → deterministic subsystem seeds
```

- `SeedBank(master_seed)` generates unique, deterministic seeds per subsystem name
- Subsystems: `pellets`, `agents`, `baseline`, `ga`
- **No global `np.random` calls anywhere.** Every RNG operation uses a named subsystem RNG from SeedBank.
- Same master seed → identical pellet placement, agent init, and all stochastic behaviour.
- Different master seeds → guaranteed non-overlapping RNG streams.

## Sensing Implementation

Grid-based spatial hashing for O(1) average pellet queries:
- Pellets bucketed into cells of size = sensor_radius
- Query: check agent's cell + 8 neighbours
- Adds/removes update the hash in O(1)
- Keeps sensing cost independent of total pellet count
