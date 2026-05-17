import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics import (
    cluster_purity,
    cluster_count,
    generate_perfectly_sorted_config,
    generate_random_config,
    generate_intermediate_config,
)
import numpy as np


def test_perfectly_sorted_purity():
    positions, colours = generate_perfectly_sorted_config()
    purity = cluster_purity(positions, colours)
    assert purity > 0.95, f"Expected purity > 0.95, got {purity}"


def test_perfectly_sorted_clusters():
    positions, colours = generate_perfectly_sorted_config()
    count = cluster_count(positions)
    assert 1 <= count <= 3, f"Expected 1-3 clusters, got {count}"


def test_random_purity():
    rng = np.random.RandomState(123)
    positions, colours = generate_random_config(rng=rng)
    purity = cluster_purity(positions, colours)
    # Expected ~0.76 (DBSCAN chance-clusters of 2 pellets have expected purity 0.75)
    # Gate: within ±5% of expected (0.72 -- 0.80)
    assert 0.72 < purity < 0.80, f"Expected purity ~0.76 (±5%), got {purity}"


def test_random_cluster_count():
    rng = np.random.RandomState(123)
    positions, colours = generate_random_config(rng=rng)
    count = cluster_count(positions)
    assert count > 2, f"Expected many clusters for random, got {count}"


def test_intermediate_purity():
    positions, colours = generate_intermediate_config()
    purity = cluster_purity(positions, colours)
    assert 0.5 < purity < 0.95, f"Expected intermediate purity, got {purity}"


def test_empty_input():
    import pytest

    with pytest.raises(ValueError):
        cluster_count([])
    with pytest.raises(ValueError):
        cluster_purity([], [])
