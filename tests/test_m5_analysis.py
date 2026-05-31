"""Tests for experiments/m5_analysis — stats, plots, pipeline."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import pytest

from experiments.m5_analysis import (
    load_aggregated,
    mann_whitney_with_effect_size,
    plot_boxplot,
    plot_timeseries_comparison,
    print_results_table,
    run_analysis,
)


@pytest.fixture
def synthetic_purities() -> tuple[list[float], list[float]]:
    return [0.72, 0.75, 0.71, 0.78, 0.74], [0.85, 0.88, 0.82, 0.90, 0.86]


@pytest.fixture
def synthetic_results() -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = {
        str(i): {"final_purity": 0.72 + i * 0.01, "purity_history": [0.5, 0.6, 0.7]}
        for i in range(5)
    }
    evolved = {
        str(i): {"final_purity": 0.85 + i * 0.01, "purity_history": [0.6, 0.75, 0.88]}
        for i in range(5)
    }
    return baseline, evolved


@pytest.fixture
def aggregated_json(tmp_path: Any) -> str:
    data = {
        "seeds": [42, 123, 256, 7, 999],
        "genome_path": "experiments/results/m5_best_genome.json",
        "baseline_results": {
            str(i): {"final_purity": 0.72 + i * 0.01, "purity_history": [0.5, 0.6, 0.7]}
            for i in range(5)
        },
        "evolved_results": {
            str(i): {
                "final_purity": 0.85 + i * 0.01,
                "purity_history": [0.6, 0.75, 0.88],
            }
            for i in range(5)
        },
    }
    path = str(tmp_path / "m5_aggregated.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return path


class TestLoadAggregated:
    def test_load_returns_dict(self, aggregated_json: str) -> None:
        data = load_aggregated(aggregated_json)
        assert isinstance(data, dict)

    def test_load_has_expected_keys(self, aggregated_json: str) -> None:
        data = load_aggregated(aggregated_json)
        assert "baseline_results" in data
        assert "evolved_results" in data
        assert "seeds" in data

    def test_load_correct_counts(self, aggregated_json: str) -> None:
        data = load_aggregated(aggregated_json)
        assert len(data["baseline_results"]) == 5
        assert len(data["evolved_results"]) == 5


class TestMannWhitney:
    def test_returns_required_keys(self, synthetic_purities) -> None:
        b, e = synthetic_purities
        result = mann_whitney_with_effect_size(b, e)
        for key in [
            "u_statistic",
            "p_value",
            "rank_biserial",
            "n_baseline",
            "n_evolved",
        ]:
            assert key in result

    def test_p_value_range(self, synthetic_purities) -> None:
        b, e = synthetic_purities
        result = mann_whitney_with_effect_size(b, e)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_rank_biserial_range(self, synthetic_purities) -> None:
        b, e = synthetic_purities
        result = mann_whitney_with_effect_size(b, e)
        assert -1.0 <= result["rank_biserial"] <= 1.0

    def test_sample_sizes(self, synthetic_purities) -> None:
        b, e = synthetic_purities
        result = mann_whitney_with_effect_size(b, e)
        assert result["n_baseline"] == 5
        assert result["n_evolved"] == 5

    def test_means_and_stds(self, synthetic_purities) -> None:
        b, e = synthetic_purities
        result = mann_whitney_with_effect_size(b, e)
        assert result["mean_evolved"] > result["mean_baseline"]
        assert result["std_baseline"] >= 0.0
        assert result["std_evolved"] >= 0.0

    def test_identical_lists(self) -> None:
        result = mann_whitney_with_effect_size([0.5, 0.5], [0.5, 0.5])
        assert result["rank_biserial"] == pytest.approx(0.0, abs=0.5)


class TestPlotBoxplot:
    def test_generates_png(self, synthetic_purities, tmp_path: Any) -> None:
        b, e = synthetic_purities
        out = str(tmp_path / "boxplot.png")
        plot_boxplot(b, e, out)
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 0

    def test_creates_parent_dir(self, synthetic_purities, tmp_path: Any) -> None:
        b, e = synthetic_purities
        out = str(tmp_path / "sub" / "boxplot.png")
        plot_boxplot(b, e, out)
        assert os.path.isfile(out)


class TestPlotTimeseries:
    def test_generates_png(self, synthetic_results, tmp_path: Any) -> None:
        b, e = synthetic_results
        out = str(tmp_path / "timeseries.png")
        plot_timeseries_comparison(b, e, out)
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 0

    def test_creates_parent_dir(self, synthetic_results, tmp_path: Any) -> None:
        b, e = synthetic_results
        out = str(tmp_path / "sub" / "ts.png")
        plot_timeseries_comparison(b, e, out)
        assert os.path.isfile(out)


class TestPrintResultsTable:
    def test_contains_labels(self, synthetic_purities) -> None:
        b, e = synthetic_purities
        stats = mann_whitney_with_effect_size(b, e)
        table = print_results_table(stats)
        assert "Deneubourg" in table
        assert "Evolved" in table

    def test_contains_mann_whitney(self, synthetic_purities) -> None:
        b, e = synthetic_purities
        stats = mann_whitney_with_effect_size(b, e)
        table = print_results_table(stats)
        assert "Mann-Whitney" in table

    def test_returns_string(self, synthetic_purities) -> None:
        b, e = synthetic_purities
        stats = mann_whitney_with_effect_size(b, e)
        table = print_results_table(stats)
        assert isinstance(table, str)
        assert len(table.splitlines()) == 3


class TestRunAnalysis:
    def test_full_pipeline(self, aggregated_json: str, tmp_path: Any) -> None:
        stats = run_analysis(aggregated_json, output_dir=str(tmp_path))
        assert isinstance(stats, dict)
        assert "p_value" in stats
        assert os.path.isfile(str(tmp_path / "m5_stats.json"))

    def test_creates_figure_dir(self, aggregated_json: str, tmp_path: Any) -> None:
        run_analysis(aggregated_json, output_dir=str(tmp_path))
        fig_dir = os.path.join("docs", "figures")
        assert os.path.isdir(fig_dir)
