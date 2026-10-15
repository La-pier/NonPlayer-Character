import pytest

from classes import (
    ALLOWED_TRANSITIONS, NPC, AllianceLayer, EmergentIntelligence, EmotionLayer,
    FactionLayer, GoalLayer, InstitutionLayer, InvalidTransition, LawLayer,
    MemoryLayer, NPCState, Personality, RelationshipLayer,
    SocialTransformationLayer,
)


def make_active(name="a") -> NPC:
    npc = NPC(name)
    npc.activate()
    return npc


# --- lifecycle -------------------------------------------------------------

def test_new_npc_is_initial_and_activates():
    npc = NPC("a")
    assert npc.state == NPCState.INITIAL
    npc.activate()
    assert npc.state == NPCState.ACTIVE
    assert npc.history[-1][1:3] == (NPCState.INITIAL, NPCState.ACTIVE)


@pytest.mark.parametrize("src", list(NPCState))
@pytest.mark.parametrize("dst", list(NPCState))
def test_transition_table_is_enforced(src, dst):
    npc = NPC("a")
    npc.state = src
    if dst in ALLOWED_TRANSITIONS[src]:
        npc.transition(dst)
        assert npc.state == dst
    else:
        with pytest.raises(InvalidTransition):
            npc.transition(dst)
        assert npc.state == src


def test_die_passes_through_death_pending():
    npc = make_active()
    npc.die("fell")
    states = [(h[1], h[2]) for h in npc.history]
    assert (NPCState.ACTIVE, NPCState.DEATH_PENDING) in states
    assert npc.state == NPCState.DEATH


def test_die_from_initial_is_invalid():
    with pytest.raises(InvalidTransition):
        NPC("a").die()


def test_rescue_returns_to_active():
    npc = make_active()
    npc.mark_dying()
    npc.rescue()
    assert npc.state == NPCState.ACTIVE


def test_resurrect_clears_short_term_and_remembers():
    npc = make_active()
    npc.individual.remember("trivia", salience=0.1)
    npc.individual.remember("lesson", salience=0.9, kind=MemoryLayer.LONG_TERM)
    npc.die()
    npc.resurrect()
    assert npc.state == NPCState.RESURRECTED
    assert npc.individual.recall(MemoryLayer.SHORT_TERM) == []
    contents = [m.content for m in npc.individual.memories]
    assert "lesson" in contents and "Returned from death" in contents


def test_is_alive_semantics():
    npc = NPC("a")
    assert npc.is_alive  # INITIAL counts as alive (not yet dead)
    npc.activate()
    assert npc.is_alive
    npc.mark_dying()
    assert not npc.is_alive
    npc.die()
    assert not npc.is_alive


def test_tick_only_in_live_states():
    npc = NPC("a")
    assert npc.tick({"events": [{"content": "x"}]}) == {}
    npc.activate()
    out = npc.tick({})
    assert set(out) == {"individual", "social", "collective"}
    npc.mark_dying()
    assert npc.tick({}) == {}


# --- individual layer ------------------------------------------------------

def test_personality_goals_adopted_once():
    npc = make_active()
    npc.individual.personality = Personality.AMBITIOUS
    first = npc.tick({})["individual"]
    assert GoalLayer.POWER in first["new_goals"]
    second = npc.tick({})["individual"]
    assert "new_goals" not in second


def test_salient_memory_promoted_to_long_term():
    npc = make_active()
    npc.tick({"events": [{"content": "big", "salience": 0.9, "emotion": "awed"}]})
    mem = npc.individual.memories[0]
    assert mem.kind == MemoryLayer.LONG_TERM
    assert mem.emotion == EmotionLayer.AWED
    assert npc.individual.emotion == EmotionLayer.AWED


def test_short_term_overflow_forgets_least_salient():
    ind = make_active().individual
    ind.short_term_limit = 3
    for i in range(5):
        ind.remember(f"m{i}", salience=0.1 + i * 0.05)
    assert ind._consolidate() == 2
    assert sorted(m.content for m in ind.memories) == ["m2", "m3", "m4"]


def test_overflow_does_not_drop_equal_duplicates_beyond_overflow():
    # Two identical entries must be treated as distinct memories.
    ind = make_active().individual
    ind.short_term_limit = 3
    a = ind.remember("same", salience=0.2)
    b = ind.remember("same", salience=0.2)
    b.timestamp = a.timestamp
    ind.remember("x", salience=0.5)
    ind.remember("y", salience=0.5)
    assert ind._consolidate() == 1
    assert len(ind.recall(MemoryLayer.SHORT_TERM)) == 3


def test_salience_is_clamped():
    ind = make_active().individual
    assert ind.remember("a", salience=5).salience == 1.0
    assert ind.remember("b", salience=-1).salience == 0.0


# --- social / collective ---------------------------------------------------

def test_relationships_and_hostility_ratio():
    soc = make_active().social
    assert soc.relationship_with("zz") == RelationshipLayer.STRANGER
    assert soc.hostility_ratio() == 0.0
    soc.update({"interactions": [{"with": "b", "kind": "enemy"},
                                 {"with": "c", "kind": "friend"}]})
    assert soc.relationship_with("c") == RelationshipLayer.FRIEND
    assert soc.hostility_ratio() == 0.5


def test_faction_and_alliance_updates():
    col = make_active().collective
    changes = col.update({"join_factions": ["military", FactionLayer.MILITARY],
                          "alliances": {"north": "alliance"}})
    assert col.factions == {FactionLayer.MILITARY}
    assert changes["joined"] == [FactionLayer.MILITARY]
    assert col.alliances["north"] == AllianceLayer.ALLIANCE


def test_snapshot_is_independent():
    ind = make_active().individual
    snap = ind.snapshot()
    ind.remember("later")
    assert len(snap.memories) == 0


# --- emergent --------------------------------------------------------------

def world_with(hostilities, institution=None):
    world = EmergentIntelligence()
    for i, h in enumerate(hostilities):
        npc = make_active(f"n{i}")
        for j in range(10):
            kind = RelationshipLayer.ENEMY if j < h else RelationshipLayer.FRIEND
            npc.social.set_relationship(f"o{j}", kind)
        npc.collective.institution = institution
        world.add(npc)
    return world


def test_observe_empty_world_is_stagnant():
    assert EmergentIntelligence().observe()["transformation"] == \
        SocialTransformationLayer.STAGNANT


@pytest.mark.parametrize("enemies,expected", [
    (0, SocialTransformationLayer.STAGNANT),
    (2, SocialTransformationLayer.GRADUAL),
    (4, SocialTransformationLayer.RAPID),
    (6, SocialTransformationLayer.REVOLUTIONARY),
])
def test_observe_transformation_thresholds(enemies, expected):
    assert world_with([enemies]).observe()["transformation"] == expected


def test_observe_derives_law_from_dominant_institution():
    world = world_with([0, 0], InstitutionLayer.DEMOCRACY)
    world.population[0].collective.institution = InstitutionLayer.THEOCRACY
    world.population.append(make_active("c"))
    world.population[-1].collective.institution = InstitutionLayer.DEMOCRACY
    assert world.observe()["law"] == LawLayer.CONSTITUTION


def test_observe_ignores_dead_npcs():
    world = world_with([10, 0])
    world.population[0].die()
    assert world.observe()["hostility"] == 0.0
