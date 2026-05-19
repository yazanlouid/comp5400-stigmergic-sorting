import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rng import SeedBank
from src.arena import Arena, Pellet, SpatialHash


def test_pellet_placement_count():
    sb = SeedBank(42)
    arena = Arena(100, 100, 200, 2, sb)
    arena.place_pellets()
    assert len(arena.pellets) == 200


def test_pellet_placement_deterministic():
    sb1 = SeedBank(42)
    sb2 = SeedBank(42)
    a1 = Arena(100, 100, 200, 2, sb1)
    a2 = Arena(100, 100, 200, 2, sb2)
    a1.place_pellets()
    a2.place_pellets()
    for p1, p2 in zip(a1.pellets, a2.pellets):
        assert p1.x == p2.x
        assert p1.y == p2.y
        assert p1.colour == p2.colour


def test_boundary_clamping():
    sb = SeedBank(42)
    arena = Arena(100, 100, 0, 2, sb)
    assert arena.clamp_position(-5, 50) == (0.0, 50.0)
    assert arena.clamp_position(105, 50) == (100.0, 50.0)
    assert arena.clamp_position(50, 50) == (50.0, 50.0)


def test_spatial_hash_query():
    sh = SpatialHash(cell_size=5.0)
    p = Pellet(id=0, x=50.0, y=50.0, colour="red")
    sh.add(p)
    results = sh.query(50.0, 50.0, 5.0)
    assert len(results) == 1
    assert results[0].id == 0


def test_pickup_drop():
    sb = SeedBank(42)
    arena = Arena(100, 100, 200, 2, sb)
    arena.place_pellets()
    initial_count = len(arena.pellets)
    pellet = arena.pickup_pellet(
        arena.pellets[0].x, arena.pellets[0].y, pickup_radius=5.0
    )
    assert pellet is not None
    assert len(arena.pellets) == initial_count - 1
    arena.drop_pellet(60.0, 60.0, pellet)
    assert len(arena.pellets) == initial_count
