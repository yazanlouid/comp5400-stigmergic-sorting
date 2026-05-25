"""Render .txt snapshot files as PNG scatter plots.

## DEBUG-GUIDE (for future AI agents)

PURPOSE:
  Takes pre-existing .txt snapshot files (CSV format from experiment.py) and
  renders them as PNG scatter plots. Used for:
  1. Re-rendering old snapshots after viz.py changes (legend, metadata format)
  2. Batch conversion: all 15 existing snapshots → PNGs in one command

DATA FLOW:
  docs/figures/*.txt (CSV: type,x,y,colour)
    → parse() extracts tick/pellets/agents/seed from # comments + CSV rows
    → render() creates scatter plot with colour legend
    → saves as docs/figures/{original_name}.png

COLOUR LEGEND:
  Each pellet colour present in the data gets a labelled dot in the legend.
  Agents shown as black dots with white edges.
  _COLOURS dict must match src/viz.py _PELLET_COLOUR_MAP exactly.

USAGE:
  python -m experiments.render_snapshots           # render all .txt files
  python -m experiments.render_snapshots file.txt   # render single file

GOTCHAS:
  - Agg backend MUST be set before pyplot import (line 4).
  - plt.close(fig) at end of render() prevents memory leaks.
  - .txt files have # comments with metadata (tick, pellets, agents) before CSV.
"""

import argparse, glob, os, re, sys
import matplotlib

matplotlib.use("Agg")  # Non-interactive — MUST be before pyplot
import matplotlib.pyplot as plt

# Must match src/viz.py _PELLET_COLOUR_MAP exactly
_COLOURS = {
    "red": "#e74c3c",
    "blue": "#3498db",
    "green": "#2ecc71",
    "yellow": "#f1c40f",
    "orange": "#e67e22",
    "purple": "#9b59b6",
}
FIG_DIR = os.path.join("docs", "figures")


def parse(path):
    with open(path) as f:
        lines = f.readlines()
    tick = pellets = agents = seed = None
    data = []
    hdr = False
    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            for p, k in [
                (r"tick\s+(\d+)", 0),
                (r"Pellets:\s*(\d+)", 1),
                (r"Agents:\s*(\d+)", 2),
            ]:
                m = re.search(p, line, re.I)
                if m:
                    val = int(m.group(1))
                    if k == 0:
                        tick = val
                    elif k == 1:
                        pellets = val
                    elif k == 2:
                        agents = val
            continue
        if not hdr:
            if line.startswith("type,"):
                hdr = True
            continue
        data.append(line)
    px, py, pc, ax, ay = [], [], [], [], []
    for row in data:
        p = row.split(",")
        if len(p) < 4:
            continue
        t, x, y, c = p[0].strip(), float(p[1]), float(p[2]), p[3].strip()
        if t == "pellet":
            px.append(x)
            py.append(y)
            pc.append(_COLOURS.get(c, "#95a5a6"))
        elif t == "agent":
            ax.append(x)
            ay.append(y)
    base = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"(\d+)", base)
    if m:
        seed = m.group(1)
    return tick, pellets, agents, seed, px, py, pc, ax, ay


def render(path, out):
    tick, pellets, agents, seed, px, py, pc, ax, ay = parse(path)
    fig, a = plt.subplots(figsize=(8, 8))
    if px:
        a.scatter(px, py, s=20, c=pc, alpha=0.7, zorder=2)
    if ax:
        a.scatter(
            ax,
            ay,
            s=50,
            facecolors="black",
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
    xlim = max(max(px) if px else 0, max(ax) if ax else 0)
    ylim = max(max(py) if py else 0, max(ay) if ay else 0)
    a.set_xlim(-5, xlim + 5)
    a.set_ylim(-5, ylim + 5)
    a.set_aspect("equal")
    a.grid(True, alpha=0.3)
    parts = []
    if seed:
        parts.append(f"Seed {seed}")
    if tick is not None:
        parts.append(f"Tick {tick}")
    if pellets is not None:
        parts.append(f"Pellets: {pellets}")
    if agents is not None:
        parts.append(f"Agents: {agents}")
    a.set_title(" | ".join(parts))

    # --- Colour legend ---
    legend_items = []
    for colour_name in sorted(set(c for c in pc if c in _COLOURS.values())):
        cname = [k for k, v in _COLOURS.items() if v == colour_name][0]
        legend_items.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=colour_name,
                markersize=6,
                markeredgecolor="gray",
                markeredgewidth=0.5,
                label=cname.capitalize() + " pellets",
            )
        )
    if ax:
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
    if legend_items:
        a.legend(
            handles=legend_items,
            loc="upper right",
            fontsize=7,
            framealpha=0.85,
            facecolor="white",
            edgecolor="gray",
            fancybox=True,
        )

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {os.path.basename(out)}")


def main():
    p = argparse.ArgumentParser(description="Render snapshot .txt files as PNG plots.")
    p.add_argument("files", nargs="*", default=[], help="Specific .txt files to render")
    args = p.parse_args()
    targets = (
        args.files if args.files else sorted(glob.glob(os.path.join(FIG_DIR, "*.txt")))
    )
    if not targets:
        print("No .txt snapshot files found")
        sys.exit(1)
    print(f"Rendering {len(targets)} snapshot(s)...")
    for f in targets:
        render(f, os.path.splitext(f)[0] + ".png")
    print(f"Done. {len(targets)} PNG(s) written to {FIG_DIR}/")


if __name__ == "__main__":
    main()
