import random

import pytest

from autonomy import Action, Agent, Needs, Perception
from classes import (
    NPC, EmotionLayer, GoalLayer, NPCState, Personality, RelationshipLayer,
)


def agent(personality=Personality.CAUTIOUS, seed=0, **needs) -> Agent:
    npc = NPC("a")
    npc.activate()
    npc.individual.personality = personality
    return Agent(npc, Needs(**needs), random.Random(seed))


def test_inactive_npc_does_nothing():
    a = Agent(NPC("a"))
    assert a.step() is None
    assert a.needs.hunger == Needs().hunger


def test_needs_rise_and_clamp():
    n = Needs(hunger=0.99)
    n.tick()
    assert n.hunger == 1.0 and n.fatigue > 0.2


def test_hungry_with_food_seeks_food():
    a = agent(hunger=0.9, fatigue=0.1, loneliness=0.1)
    assert a.decide(Perception(food_available=True)) == Action.SEEK_FOOD


def test_exhausted_rests():
    a = agent(hunger=0.1, fatigue=0.95, loneliness=0.1)
    assert a.decide(Perception()) == Action.REST


def test_threat_beats_hunger():
    a = agent(hunger=0.8)
    assert a.decide(Perception(threat=0.9, food_available=True)) == Action.AVOID


def test_personality_changes_choice_in_same_situation():
    p = Perception(threat=0.4)
    # A moderate competing need (fatigue 0.5) that a bold NPC puts ahead of a
    # moderate threat, but a cautious one does not.
    cautious = agent(Personality.CAUTIOUS, hunger=0.1, fatigue=0.5, loneliness=0.1)
    bold = agent(Personality.BOLD, hunger=0.1, fatigue=0.5, loneliness=0.1)
    assert cautious.decide(p) == Action.AVOID
    assert bold.decide(p) == Action.REST


def test_lonely_extrovert_socializes_only_when_someone_nearby():
    a = agent(Personality.EXTROVERTED, hunger=0.1, fatigue=0.1, loneliness=0.9)
    assert a.decide(Perception(nearby=["b"])) == Action.SOCIALIZE
    assert Action.SOCIALIZE not in {a.decide(Perception()) for _ in range(20)}


def test_nearby_enemy_counts_as_threat():
    a = agent(hunger=0.1, fatigue=0.1, loneliness=0.1)
    a.npc.social.set_relationship("b", RelationshipLayer.ENEMY)
    assert a.decide(Perception(nearby=["b"])) == Action.AVOID


def test_goal_pursued_when_needs_are_low():
    a = agent(Personality.AMBITIOUS, hunger=0.0, fatigue=0.0, loneliness=0.0,
              hunger_rate=0, fatigue_rate=0, loneliness_rate=0)
    a.npc.individual.goals.append(GoalLayer.POWER)
    assert a.decide(Perception()) == Action.PURSUE_GOAL


def test_act_reduces_need_and_updates_layers():
    a = agent(hunger=0.9)
    p = Perception(food_available=True)
    before = a.needs.hunger
    action = a.step(p)
    assert action == Action.SEEK_FOOD
    assert a.needs.hunger < before
    assert a.npc.individual.emotion == EmotionLayer.RELIEVED
    assert any("food" in m.content for m in a.npc.individual.memories)


def test_socializing_turns_stranger_into_acquaintance():
    a = agent(Personality.EXTROVERTED, hunger=0.0, fatigue=0.0, loneliness=0.95)
    a.step(Perception(nearby=["b"]))
    assert a.npc.social.relationship_with("b") == RelationshipLayer.ACQUAINTANCE


def test_seeded_runs_are_reproducible():
    def run(seed):
        a = agent(seed=seed)
        return [a.step(Perception(food_available=True, nearby=["b"]))
                for _ in range(30)]
    assert run(1) == run(1)


def test_long_run_keeps_needs_bounded_and_npc_alive():
    a = agent()
    for i in range(500):
        a.step(Perception(food_available=i % 3 == 0, nearby=["b"] if i % 2 else []))
    n = a.needs
    assert all(0.0 <= v <= 1.0 for v in (n.hunger, n.fatigue, n.loneliness))
    assert a.npc.state == NPCState.ACTIVE
    assert len(a.npc.individual.recall()) <= 500


# --- purpose: goal pursuit is a drive, not the default idle action -----------------------------

def test_purpose_rises_each_tick_and_clamps():
    n = Needs(purpose=0.95)
    n.tick()
    assert n.purpose == 1.0
    n = Needs()
    n.tick()
    assert n.purpose == pytest.approx(0.2 + n.purpose_rate)


def test_no_goals_means_no_pursuit_urge():
    a = agent(hunger=0.0, fatigue=0.0, loneliness=0.0, purpose=1.0)
    assert a.npc.individual.goals == []
    assert a.score(Perception())[Action.PURSUE_GOAL] == 0.0


def test_pursuing_a_goal_relieves_purpose_so_it_is_not_endless():
    from autonomy import PURPOSE_RELIEF
    a = agent(Personality.CAUTIOUS, hunger=0.0, fatigue=0.0, loneliness=0.0,
              purpose=0.9, hunger_rate=0, fatigue_rate=0, loneliness_rate=0,
              purpose_rate=0.05)
    a.npc.individual.goals.append(GoalLayer.KNOWLEDGE)
    assert a.step(Perception()) == Action.PURSUE_GOAL
    assert a.needs.purpose == pytest.approx(0.9 + 0.05 - PURPOSE_RELIEF)
    # the urge itself is what competes with other actions, and it just dropped
    assert a.score(Perception())[Action.PURSUE_GOAL] == pytest.approx(a.needs.purpose)
    assert a.needs.purpose < 0.95 - PURPOSE_RELIEF + 1e-9


def test_goal_pursuit_share_is_bounded_when_idle():
    """With every other need satisfied the agent still doesn't pursue goals
    every tick (the old constant score made that the default)."""
    a = agent(Personality.CAUTIOUS, hunger=0.0, fatigue=0.0, loneliness=0.0,
              hunger_rate=0, fatigue_rate=0, loneliness_rate=0)
    a.npc.individual.goals.append(GoalLayer.KNOWLEDGE)
    acts = [a.step(Perception()) for _ in range(100)]
    share = acts.count(Action.PURSUE_GOAL) / len(acts)
    assert 0.05 < share < 0.6
