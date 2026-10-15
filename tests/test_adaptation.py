import random

import pytest

import adaptation
from adaptation import (
    BASELINE, TraitProfile, learn, reflect, retrieve, sentiment, valence,
)
from autonomy import Action, Agent, Needs, Perception
from classes import (
    NPC, BeliefLayer, EmotionLayer, GoalLayer, IndividualIntelligence,
    Personality, ReflectionLayer,
)


def mind(**kw) -> IndividualIntelligence:
    return IndividualIntelligence(**kw)


def agent(personality=Personality.CAUTIOUS, seed=0, **needs) -> Agent:
    npc = NPC("a")
    npc.activate()
    npc.individual.personality = personality
    return Agent(npc, Needs(**needs), random.Random(seed))


# --- retrieval ---------------------------------------------------------------

def test_event_tags_are_stored_on_memories():
    npc = NPC("a")
    npc.activate()
    npc.tick({"events": [{"content": "x", "tags": ["danger", "npc:b"]}]})
    assert npc.individual.memories[0].tags == ("danger", "npc:b")


def test_retrieve_filters_by_tag_and_ranks_salience():
    m = mind()
    m.remember("low", salience=0.1, tags=["npc:b"])
    m.remember("high", salience=0.9, tags=["npc:b"])
    m.remember("other", salience=1.0, tags=["npc:c"])
    got = retrieve(m, ["npc:b"], k=5)
    assert [x.content for x in got] == ["high", "low"]


def test_retrieve_prefers_recent_when_salience_equal():
    m = mind()
    for i in range(5):
        m.remember(f"m{i}", salience=0.5, tags=["t"])
    assert retrieve(m, ["t"], k=1)[0].content == "m4"


def test_retrieve_without_query_returns_top_k_overall():
    m = mind()
    for i in range(10):
        m.remember(f"m{i}", salience=0.5)
    assert len(retrieve(m, k=3)) == 3
    assert retrieve(mind(), ["x"]) == []


def test_sentiment_reflects_emotions():
    m = mind()
    m.remember("nice", EmotionLayer.HAPPY, 0.5, tags=["npc:b"])
    m.remember("scary", EmotionLayer.FEARFUL, 0.5, tags=["npc:c"])
    assert sentiment(m, ["npc:b"]) > 0 > sentiment(m, ["npc:c"])
    assert sentiment(m, ["npc:zzz"]) == 0.0
    assert valence(m.memories[0]) == 1.0


# --- traits ------------------------------------------------------------------

def test_reinforce_raises_trait_and_lowers_opposite():
    p = TraitProfile()
    p.reinforce(Personality.BOLD, 0.2)
    gain = 0.2 * (1 - BASELINE)                    # diminishing returns
    assert p.strengths[Personality.BOLD] == pytest.approx(BASELINE + gain)
    assert p.strengths[Personality.CAUTIOUS] == pytest.approx(BASELINE - gain / 2)


def test_reinforcement_has_diminishing_returns():
    p = TraitProfile()
    gains = []
    for _ in range(30):
        before = p.strengths[Personality.BOLD]
        p.reinforce(Personality.BOLD, 0.2)
        gains.append(p.strengths[Personality.BOLD] - before)
    assert all(a > b > 0 for a, b in zip(gains, gains[1:]))   # strictly shrinking
    assert p.strengths[Personality.BOLD] < 1.0                # never saturates


def test_learning_rate_scales_how_fast_traits_move(monkeypatch):
    def after(rate):
        monkeypatch.setattr(adaptation, "LEARNING_RATE", rate)
        p = TraitProfile()
        learn(p, "socialize", EmotionLayer.HAPPY)
        return p.strengths[Personality.EXTROVERTED] - BASELINE
    assert after(1.0) == pytest.approx(10 * after(0.1))


def test_strengths_stay_in_range():
    p = TraitProfile()
    for _ in range(100):
        p.reinforce(Personality.BOLD, 0.1)
    assert all(0.0 <= v <= 1.0 for v in p.strengths.values())


def test_drift_pulls_toward_baseline():
    p = TraitProfile.from_personality(Personality.SHY)
    before = p.strengths[Personality.SHY]
    for _ in range(50):
        p.tick()
    assert BASELINE < p.strengths[Personality.SHY] < before


def test_dominant_has_hysteresis():
    p = TraitProfile.from_personality(Personality.CAUTIOUS)   # 0.7
    p.strengths[Personality.BOLD] = 0.75
    assert p.dominant(Personality.CAUTIOUS) == Personality.CAUTIOUS
    p.strengths[Personality.BOLD] = 0.9
    assert p.dominant(Personality.CAUTIOUS) == Personality.BOLD


def test_learn_applies_lessons():
    p = TraitProfile()
    learn(p, "socialize", EmotionLayer.HAPPY)
    assert p.strengths[Personality.EXTROVERTED] > BASELINE
    learn(p, "pursue_goal", None)
    assert p.strengths[Personality.AMBITIOUS] > BASELINE
    before = dict(p.strengths)
    learn(p, "rest", None)          # no lesson
    assert p.strengths == before


def test_personality_can_change_through_experience(monkeypatch):
    # Personality is slow by default; speed learning up so 80 ticks is enough.
    monkeypatch.setattr(adaptation, "LEARNING_RATE", 1.0)
    a = agent(Personality.SHY, hunger=0.0, fatigue=0.0, loneliness=0.9,
              hunger_rate=0, fatigue_rate=0)
    for _ in range(80):
        a.step(Perception(nearby=["b"]))
    assert a.npc.individual.personality == Personality.EXTROVERTED


# --- reflection --------------------------------------------------------------

def feed(m, emotion, n, tags=()):
    for _ in range(n):
        m.remember("e", emotion, 0.5, tags=tags)


def test_reflect_needs_enough_memories():
    m = mind()
    feed(m, EmotionLayer.FEARFUL, 3)
    assert reflect(m) == {}


def test_bad_experience_adopts_pessimism_and_peace():
    m = mind(beliefs={BeliefLayer.OPTIMISTIC})
    feed(m, EmotionLayer.FEARFUL, 8)
    out = reflect(m)
    assert BeliefLayer.PESSIMISTIC in m.beliefs
    assert BeliefLayer.OPTIMISTIC not in m.beliefs
    assert out["revised_belief"] == BeliefLayer.OPTIMISTIC
    assert GoalLayer.PEACE in m.goals
    assert m.reflection == ReflectionLayer.THOROUGH


def test_good_experience_revises_pessimism():
    m = mind(beliefs={BeliefLayer.PESSIMISTIC})
    feed(m, EmotionLayer.HAPPY, 10)
    out = reflect(m)
    assert m.beliefs == {BeliefLayer.OPTIMISTIC}
    assert out["revised_belief"] == BeliefLayer.PESSIMISTIC


def test_social_joy_forms_connection_goal_once():
    m = mind()
    feed(m, EmotionLayer.HAPPY, 6, tags=("npc:b",))
    reflect(m)
    reflect(m)
    assert m.goals.count(GoalLayer.CONNECTION) == 1


def test_mixed_experience_changes_no_beliefs():
    m = mind()
    feed(m, EmotionLayer.HAPPY, 3)
    feed(m, EmotionLayer.SAD, 3)
    assert reflect(m) == {}
    assert m.beliefs == set() and m.reflection == ReflectionLayer.SHALLOW


# --- integration -------------------------------------------------------------

def test_good_history_with_an_npc_raises_socialize_score():
    a = agent(hunger=0.0, fatigue=0.0, loneliness=0.5)
    base = a.score(Perception(nearby=["b"]))[Action.SOCIALIZE]
    a.npc.individual.remember("fun", EmotionLayer.HAPPY, 0.8, tags=["npc:b"])
    assert a.score(Perception(nearby=["b"]))[Action.SOCIALIZE] > base
    a.npc.individual.remember("fight", EmotionLayer.ANGRY, 0.9, tags=["npc:c"])
    assert a.score(Perception(nearby=["c"]))[Action.SOCIALIZE] < base


def test_pessimistic_belief_increases_avoidance():
    a = agent()
    p = Perception(threat=0.5)
    base = a.score(p)[Action.AVOID]
    a.npc.individual.beliefs.add(BeliefLayer.PESSIMISTIC)
    assert a.score(p)[Action.AVOID] > base


def test_agent_reflects_on_schedule():
    a = agent(hunger=0.0, fatigue=0.0, loneliness=0.9, hunger_rate=0, fatigue_rate=0)
    a.reflect_every = 5
    for _ in range(12):
        a.step(Perception(nearby=["b"]))
    assert BeliefLayer.OPTIMISTIC in a.npc.individual.beliefs


def test_long_adaptive_run_stays_sane():
    a = agent()
    for i in range(400):
        a.step(Perception(threat=0.7 if i % 7 == 0 else 0.0,
                          food_available=i % 3 == 0,
                          nearby=["b"] if i % 2 else []))
    assert all(0.0 <= v <= 1.0 for v in a.traits.strengths.values())
    assert a.npc.individual.personality in Personality
