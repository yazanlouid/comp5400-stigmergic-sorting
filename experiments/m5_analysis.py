"""M5 statistical analysis — Mann-Whitney U, boxplots, time-series comparison."""

from __future__ import annotations

import json
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

# Data loading

def load_aggregated(path: str) -> dict[str, Any]:
    """Load m5_aggregated.json and return parsed dict."""
    with open(path, "r") as f:
        return json.load(f)


def _extract_purities(results: dict[str, Any]) -> list[float]:
    """Extract final_purity values from results dict (seed-keyed)."""
    return [v["final_purity"] for v in results.values()]

# Statistics

def mann_whitney_with_effect_size(
    baseline_purities: list[float],
    evolved_purities: list[float],
) -> dict[str, Any]:
    """Mann-Whitney U test with rank-biserial effect size.

    Returns dict with u_statistic, p_value, rank_biserial,
    n_baseline, n_evolved, mean_baseline, std_baseline,
    mean_evolved, std_evolved.
    """
    b = np.array(baseline_purities)
    e = np.array(evolved_purities)

    u_stat, p_value = sp_stats.mannwhitneyu(b, e, alternative="two-sided")

    n1, n2 = len(b), len(e)
    rank_biserial = 1.0 - (2.0 * u_stat) / (n1 * n2) if (n1 * n2) > 0 else 0.0

    return {
        "u_statistic": float(u_stat),
        "p_value": float(p_value),
        "rank_biserial": float(rank_biserial),
        "n_baseline": n1,
        "n_evolved": n2,
        "mean_baseline": float(np.mean(b)),
        "std_baseline": float(np.std(b)),
        "median_baseline": float(np.median(b)),
        "mean_evolved": float(np.mean(e)),
        "std_evolved": float(np.std(e)),
        "median_evolved": float(np.median(e)),
    }

# Plots

def plot_boxplot(
    baseline_purities: list[float],
    evolved_purities: list[float],
    output_path: str,
) -> None:
    """Boxplot of final purity distributions with jittered points."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    data = [baseline_purities, evolved_purities]
    bp = ax.boxplot(data, patch_artist=True)
    ax.set_xticklabels(["Deneubourg", "Evolved"])

    colors = ["#3498db", "#e74c3c"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Final Cluster Purity")
    ax.set_title("M5: Baseline vs Evolved")
    ax.set_ylim(0.5, 1.0)

    # Jittered individual points
    np.random.seed(42)
    for i, group in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(group))
        ax.scatter(x, group, color=colors[i], alpha=0.7, zorder=3, s=30)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_timeseries_comparison(
    baseline_results: dict[str, Any],
    evolved_results: dict[str, Any],
    output_path: str,
) -> None:
    """Mean purity over time with ±1σ shading for each condition."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))

    for label, results, color in [
        ("Deneubourg", baseline_results, "#3498db"),
        ("Evolved", evolved_results, "#e74c3c"),
    ]:
        histories = [v["purity_history"] for v in results.values()]
        if not histories:
            continue

        # Align to shortest history
        min_len = min(len(h) for h in histories)
        aligned = np.array([h[:min_len] for h in histories])

        mean = np.mean(aligned, axis=0)
        std = np.std(aligned, axis=0)
        ticks = np.arange(min_len)

        ax.plot(ticks, mean, color=color, linewidth=1.5, label=label)
        ax.fill_between(ticks, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Metric Sample Index")
    ax.set_ylabel("Cluster Purity")
    ax.set_title("M5: Purity Over Time (mean ± 1σ)")
    ax.legend()
    ax.set_ylim(0.0, 1.05)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# Reporting

def print_results_table(stats_result: dict[str, Any]) -> str:
    """Return a LaTeX tabular row string for report.tex.

    Format: condition & n & mean & std & median \\
    """
    rows = []
    for label, prefix in [("Deneubourg", "baseline"), ("Evolved", "evolved")]:
        n = stats_result[f"n_{prefix}"]
        mean = stats_result[f"mean_{prefix}"]
        std = stats_result[f"std_{prefix}"]
        median = stats_result[f"median_{prefix}"]
        rows.append(f"  {label} & {n} & {mean:.4f} & {std:.4f} & {median:.4f} \\\\")

    p_val = stats_result["p_value"]
    rb = stats_result["rank_biserial"]
    rows.append(
        f"  \\textit{{Mann-Whitney U}} & -- & -- & $p$={p_val:.4f} & $r$={rb:.4f} \\\\"
    )

    return "\n".join(rows)

# Pipeline

def run_analysis(
    aggregated_path: str,
    output_dir: str = "experiments/results",
) -> dict[str, Any]:
    """Full analysis pipeline: load → stats → plots → save.

    Returns the stats dict.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join("docs", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Load
    data = load_aggregated(aggregated_path)
    baseline_results = data["baseline_results"]
    evolved_results = data["evolved_results"]

    baseline_purities = _extract_purities(baseline_results)
    evolved_purities = _extract_purities(evolved_results)

    print(
        f"Loaded {len(baseline_purities)} baseline, {len(evolved_purities)} evolved seeds"
    )

    # Stats
    stats_result = mann_whitney_with_effect_size(baseline_purities, evolved_purities)
    print(
        f"Mann-Whitney U={stats_result['u_statistic']}, p={stats_result['p_value']:.4f}"
    )
    print(f"Rank-biserial r={stats_result['rank_biserial']:.4f}")

    # Plots
    boxplot_path = os.path.join(fig_dir, "m5_boxplot.png")
    plot_boxplot(baseline_purities, evolved_purities, boxplot_path)
    print(f"Boxplot saved to {boxplot_path}")

    timeseries_path = os.path.join(fig_dir, "m5_timeseries.png")
    plot_timeseries_comparison(baseline_results, evolved_results, timeseries_path)
    print(f"Timeseries saved to {timeseries_path}")

    # Save stats JSON
    stats_path = os.path.join(output_dir, "m5_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats_result, f, indent=2)
    print(f"Stats saved to {stats_path}")

    # LaTeX table
    table = print_results_table(stats_result)
    print("\nLaTeX table rows:")
    print(table)

    return stats_result

# CLI

def main() -> None:
    """Run M5 analysis from command line."""
    aggregated = os.path.join("experiments", "results", "m5_aggregated.json")
    if not os.path.exists(aggregated):
        print(f"ERROR: {aggregated} not found. Run m5_runner.py first.")
        return

    print("=" * 60)
    print("M5 Statistical Analysis")
    print("=" * 60)
    run_analysis(aggregated)


if __name__ == "__main__":
    main()
