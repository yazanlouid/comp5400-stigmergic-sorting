# Running the GUI

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

**Dependencies:** numpy, scipy, matplotlib, PyYAML, scikit-learn, pytest, pygame

---

## Run the GUI

```bash
python -m experiments.run_gui --config experiments/configs/baseline.yaml
```

Replace `baseline.yaml` with any config in `experiments/configs/`.

---

## GUI Controls

| Key | Action |
|-----|--------|
| `Space` | Pause / Resume |
| `R` | Reset simulation |
| `Q` | Quit |
| `S` | Save snapshot → `docs/figures/gui_snapshot_tick{N}.png` |
| `+` / `-` | Speed up/down (1x–10x) |

---

## Run Headless Experiments

Single run (produces CSV + metrics + events + PNG snapshots):

```bash
python -m src.experiment --config experiments/configs/baseline.yaml
```

Output lands in `experiments/results/` as `{configname}_{seed}_{type}.*`.

---

## Run M5 Head-to-Head

Evolve once, then compare evolved vs Deneubourg across 5 seeds:

```bash
python -m experiments.m5_runner
```

Aggregated results → `experiments/results/m5_aggregated.json`

---

## Config Levers

All configs live in `experiments/configs/`. Edit YAML to tune:

### Environment
| Key | Default | What it does |
|-----|---------|--------------|
| `seed` | 42 | RNG seed (deterministic runs) |
| `arena.width` / `height` | 100 | Arena dimensions |
| `pellets.total` | 200 | Number of pellets |
| `pellets.colours` | 2 | Number of pellet colours (2–6) |
| `agents.count` | 20 | Number of agents |
| `agents.sensor_radius` | 5 | Local sensing radius |

### Episode
| Key | Default | What it does |
|-----|---------|--------------|
| `episode.max_ticks` | 10000 | Ticks per episode |
| `episode.early_stop_delta` | 0.01 | Purity convergence threshold |
| `episode.early_stop_window` | 2000 | Ticks to check convergence over |
| `episode.metrics_interval` | 100 | How often to sample metrics |

### Agent Type
| Key | Values | What it does |
|-----|--------|--------------|
| `agent_type` | `random`, `deneubourg`, `evolved` | Which controller to use |

### Deneubourg (when `agent_type: deneubourg`)
| Key | Default | What it does |
|-----|---------|--------------|
| `deneubourg.k1` | 0.1 | Pickup threshold constant |
| `deneubourg.k2` | 0.3 | Drop threshold constant |

### Evolution (when `agent_type: evolved`)
| Key | Default | What it does |
|-----|---------|--------------|
| `evolution.pop_size` | 30 | GA population size |
| `evolution.generations` | 50 | GA generations |
| `evolution.tournament_size` | 3 | Tournament selection size |
| `evolution.mutation_sigma` | 0.3 | Gaussian mutation std |
| `evolution.eval_episodes` | 3 | Episodes per genome eval |
| `evolution.eval_max_ticks` | 5000 | Ticks per eval episode |
| `evolution.pickup_bias` | 0.0 | NN pickup output bias (warm-start) |
| `evolution.move_bias` | 0.5 | NN move output bias |

### Fitness Weights (composite fitness)
| Key | Default | Component |
|-----|---------|-----------|
| `evolution.fitness_alpha` | 0.20 | Terminal purity |
| `evolution.fitness_beta` | 0.15 | Integrated purity |
| `evolution.fitness_gamma` | 0.15 | Cluster consolidation |
| `evolution.fitness_delta` | 0.20 | Pickup/drop activity (decays) |
| `evolution.fitness_epsilon` | 0.30 | Drop locality bonus |
| `evolution.activity_decay_gen` | 25 | Generation when activity weight reaches 0 |

---

## Available Configs

| Config | Agent Type | Use case |
|--------|-----------|----------|
| `sanity.yaml` | random | Quick sanity check (1000 ticks) |
| `baseline.yaml` | deneubourg | Deneubourg baseline run |
| `fitness_quick.yaml` | evolved | Fast evolution test (15 pop, 10 gen) |
| `fitness_default.yaml` | evolved | Full evolution run (30 pop, 50 gen) |
