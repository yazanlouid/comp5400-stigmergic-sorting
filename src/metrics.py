from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN


cluster_range = 2.0 #5

def cluster_sizes(
    positions: list[tuple[float, float]],
    eps: float = cluster_range,
    min_samples: int = 2,
) -> list[int]:
    """Return size of each DBSCAN cluster, excluding noise."""
    if not positions:
        return []

    X = np.array(positions, dtype=np.float64)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(X).labels_

    sizes: list[int] = []

    for label in set(labels):
        if label == -1:
            continue

        size = int(np.sum(labels == label))
        sizes.append(size)

    return sizes


def cluster_count(
    positions: list[tuple[float, float]],
    eps: float = cluster_range, # 5
    min_samples: int = 2,
) -> int:
    """Count pellet clusters using DBSCAN, excluding noise.

    If every pellet is classified as noise, each pellet counts as its own
    cluster (i.e. the function returns *len(positions)*).

    Args:
        positions: (x, y) coordinates of every pellet.
        eps: Maximum distance between two samples for one to be considered
             as in the neighbourhood of the other.
        min_samples: Number of samples in a neighbourhood for a point to be
                     considered as a core point.

    Returns:
        Number of non-noise clusters (or total pellet count when all are noise).
    """
    X = np.array(positions, dtype=np.float64)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(X).labels_

    unique_labels = set(labels)
    unique_labels.discard(-1)

    if not unique_labels:
        return len(positions)
    return len(unique_labels)


def cluster_purity(
    positions: list[tuple[float, float]],
    colours: list[str],
    eps: float = cluster_range,
    min_samples: int = 2,
) -> float:
    """Compute mean per-cluster colour purity using DBSCAN.

    For each non-noise cluster the purity is the fraction of pellets that
    share the most common colour in that cluster.  The overall score is the
    arithmetic mean across all non-noise clusters.

    Args:
        positions: (x, y) coordinates of every pellet.
        colours: Colour label for each pellet (same order as *positions*).
        eps: DBSCAN neighbourhood radius.
        min_samples: DBSCAN minimum samples for core point.

    Returns:
        Mean purity in [0.0, 1.0].  Returns 0.0 when no non-noise clusters
        exist.
    """
    X = np.array(positions, dtype=np.float64)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(X).labels_

    purities: list[float] = []
    for label in set(labels):
        if label == -1:
            continue
        mask = labels == label
        cluster_colours = [colours[i] for i in range(len(colours)) if mask[i]]
        counts: dict[str, int] = {}
        for c in cluster_colours:
            counts[c] = counts.get(c, 0) + 1
        purities.append(max(counts.values()) / len(cluster_colours))

    if not purities:
        return 0.0
    return float(np.mean(purities))


def generate_perfectly_sorted_config(
    n_per_colour: int = 100,
    arena_size: float = 100.0,
    rng: np.random.RandomState | None = None,
) -> tuple[list[tuple[float, float]], list[str]]:
    """Generate a fully sorted pellet configuration.

    Red pellets cluster around (25, 50), blue pellets around (75, 50),
    both drawn from a normal distribution (std=2.0) and clamped to the
    arena bounds.

    Args:
        n_per_colour: Number of pellets per colour.
        arena_size: Side length of the square arena.
        rng: Seeded RandomState for reproducibility.  Defaults to
             ``RandomState(42)`` when *None*.

    Returns:
        Tuple of (positions, colours).  Expected purity ≈ 1.0,
        cluster_count ≈ 2.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    red_x = rng.normal(25.0, 2.0, n_per_colour)
    red_y = rng.normal(50.0, 2.0, n_per_colour)
    blue_x = rng.normal(75.0, 2.0, n_per_colour)
    blue_y = rng.normal(50.0, 2.0, n_per_colour)

    positions: list[tuple[float, float]] = []
    colours: list[str] = []

    for x, y in zip(red_x, red_y):
        positions.append(
            (float(np.clip(x, 0, arena_size)), float(np.clip(y, 0, arena_size)))
        )
        colours.append("red")

    for x, y in zip(blue_x, blue_y):
        positions.append(
            (float(np.clip(x, 0, arena_size)), float(np.clip(y, 0, arena_size)))
        )
        colours.append("blue")

    return positions, colours


def generate_random_config(
    total: int = 200,
    num_colours: int = 2,
    arena_size: float = 100.0,
    rng: np.random.RandomState | None = None,
) -> tuple[list[tuple[float, float]], list[str]]:
    """Generate a uniformly random pellet configuration.

    Positions are drawn uniformly from the arena.  Colours are assigned
    evenly then shuffled.

    Args:
        total: Total number of pellets.
        num_colours: Number of distinct colours (currently only 2 is used).
        arena_size: Side length of the square arena.
        rng: Seeded RandomState for reproducibility.  Defaults to
             ``RandomState(42)`` when *None*.

    Returns:
        Tuple of (positions, colours).
    """
    if rng is None:
        rng = np.random.RandomState(42)

    xs = rng.uniform(0, arena_size, total)
    ys = rng.uniform(0, arena_size, total)
    positions = [(float(x), float(y)) for x, y in zip(xs, ys)]

    n_red = total // 2
    colours = ["red"] * n_red + ["blue"] * (total - n_red)
    rng.shuffle(colours)

    return positions, colours


def generate_intermediate_config(
    n_per_colour: int = 100,
    arena_size: float = 100.0,
    rng: np.random.RandomState | None = None,
) -> tuple[list[tuple[float, float]], list[str]]:
    """Generate a partially sorted pellet configuration.

    80 % of each colour is clustered near its target centre; the remaining
    20 % is scattered uniformly across the arena.

    Args:
        n_per_colour: Number of pellets per colour.
        arena_size: Side length of the square arena.
        rng: Seeded RandomState for reproducibility.  Defaults to
             ``RandomState(42)`` when *None*.

    Returns:
        Tuple of (positions, colours).
    """
    if rng is None:
        rng = np.random.RandomState(42)

    positions: list[tuple[float, float]] = []
    colours: list[str] = []

    for centre, colour in [(25.0, "red"), (75.0, "blue")]:
        n_clustered = int(n_per_colour * 0.8)
        n_scattered = n_per_colour - n_clustered

        cx = rng.normal(centre, 3.0, n_clustered)
        cy = rng.normal(50.0, 3.0, n_clustered)
        for x, y in zip(cx, cy):
            positions.append(
                (float(np.clip(x, 0, arena_size)), float(np.clip(y, 0, arena_size)))
            )
            colours.append(colour)

        sx = rng.uniform(0, arena_size, n_scattered)
        sy = rng.uniform(0, arena_size, n_scattered)
        for x, y in zip(sx, sy):
            positions.append(
                (float(np.clip(x, 0, arena_size)), float(np.clip(y, 0, arena_size)))
            )
            colours.append(colour)

    return positions, colours
