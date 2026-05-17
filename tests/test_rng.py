import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rng import SeedBank


def test_seed_split_deterministic():
    sb1 = SeedBank(42)
    sb2 = SeedBank(42)
    assert sb1.get_seed("pellets") == sb2.get_seed("pellets")


def test_different_names_different_seeds():
    sb = SeedBank(42)
    s1 = sb.get_seed("pellets")
    s2 = sb.get_seed("agents")
    assert s1 != s2


def test_different_master_different_seeds():
    sb1 = SeedBank(42)
    sb2 = SeedBank(99)
    assert sb1.get_seed("pellets") != sb2.get_seed("pellets")
