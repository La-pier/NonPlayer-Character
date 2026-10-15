"""Behaviour-balance regression tests.

These guard the *shape* of the simulated behaviour (not exact numbers, which the
golden run pins): no single action should dominate, strangers shouldn't be
hostile by default, grudges shouldn't pile up, and the starting personalities
shouldn't collapse into one. Bands are deliberately wider than the tuning
targets (see README, "Behaviour balance") so ordinary changes don't trip them,
but a real regression (e.g. an action becoming the default) does.
"""
from collections import Counter

import pytest

from classes import RelationshipLayer
from simulation import SimConfig, run

SEEDS = (101, 102, 103, 104)          # not the seeds used for tuning
TICKS = 150


@pytest.fixture(scope="module")
def worlds():
    return [run(SimConfig(seed=s, npcs=12), TICKS) for s in SEEDS]


def test_no_action_dominates_or_vanishes(worlds):
    acts = Counter(r["action"] for w in worlds for r in w.log.of_type("action"))
    total = sum(acts.values())
    shares = {a: n / total for a, n in acts.items()}
    assert set(shares) == {"avoid", "pursue_goal", "rest", "seek_food", "socialize"}
    assert all(0.05 <= s <= 0.40 for s in shares.values()), shares


def test_first_meetings_are_mostly_friendly(worlds):
    kinds = Counter(r["kind"] for w in worlds for r in w.log.of_type("interaction"))
    argue = kinds["argue"] / sum(kinds.values())
    assert 0.08 <= argue <= 0.28, kinds


def test_grudges_do_not_pile_up(worlds):
    enemies = [len(a.npc.social.with_kind(RelationshipLayer.ENEMY))
               for w in worlds for a in w.agents.values()]
    assert sum(enemies) / len(enemies) <= 1.6


def test_avoidance_is_not_the_default_response_to_people(worlds):
    acts = Counter(r["action"] for w in worlds for r in w.log.of_type("action"))
    assert acts["avoid"] / sum(acts.values()) <= 0.25


def test_personalities_do_not_collapse(worlds):
    final = Counter(a.npc.individual.personality for w in worlds
                    for a in w.agents.values())
    assert len(final) >= 5
    assert max(final.values()) / sum(final.values()) <= 0.40


def test_balance_test_can_fail():
    """Sanity: a world where goal pursuit never relaxes would breach the band."""
    import autonomy
    from autonomy import Action, Agent, Needs, Perception
    from classes import NPC, GoalLayer
    npc = NPC("x"); npc.activate(); npc.individual.goals.append(GoalLayer.POWER)
    a = Agent(npc, Needs(hunger_rate=0, fatigue_rate=0, loneliness_rate=0,
                         purpose_rate=1.0))       # purpose always maxed
    acts = [a.step(Perception()) for _ in range(50)]
    assert acts.count(Action.PURSUE_GOAL) / 50 > 0.40
