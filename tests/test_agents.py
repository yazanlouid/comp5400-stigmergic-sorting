import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rng import SeedBank
from src.arena import Arena
from src.agents import BaseAgent, Action, create_agents


def test_create_agents_count():
    sb = SeedBank(42)
    arena = Arena(100, 100, 0, 2, sb)
    agents = create_agents(20, arena, sb)
    assert len(agents) == 20


def test_agent_ids_sequential():
    sb = SeedBank(42)
    arena = Arena(100, 100, 0, 2, sb)
    agents = create_agents(20, arena, sb)
    assert [a.id for a in agents] == list(range(20))


def test_action_move():
    sb = SeedBank(42)
    arena = Arena(100, 100, 0, 2, sb)
    agent = BaseAgent(0, 50.0, 50.0, 0.0)
    agent.apply_action(Action.MOVE, arena)
    assert abs(agent.x - 51.0) < 0.01
    assert abs(agent.y - 50.0) < 0.01


def test_action_turn():
    agent = BaseAgent(0, 50.0, 50.0, 0.0)
    sb = SeedBank(42)
    arena = Arena(100, 100, 0, 2, sb)
    agent.apply_action(Action.TURN_RIGHT, arena)
    assert agent.heading_deg == 30.0
    agent.apply_action(Action.TURN_LEFT, arena)
    assert agent.heading_deg == 0.0


def test_heading_normalization():
    agent = BaseAgent(0, 0, 0, 0.0)
    assert agent._normalize_heading(180) == -180
    assert agent._normalize_heading(181) == -179
    assert agent._normalize_heading(-181) == 179
    assert agent._normalize_heading(360) == 0.0


def test_random_walk():
    sb = SeedBank(42)
    arena = Arena(100, 100, 0, 2, sb)
    agent = BaseAgent(0, 50.0, 50.0, 0.0)
    rng = sb.get_rng("test")
    action = agent.decide_action(arena, 5.0, rng)
    assert action in [Action.MOVE, Action.TURN_LEFT, Action.TURN_RIGHT]
