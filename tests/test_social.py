import random

import pytest

from autonomy import Action, Agent, Needs, Perception
from classes import (
    NPC, AllianceLayer, EmotionLayer, InstitutionLayer, LawLayer, LeadershipLayer,
    NPCState, Personality, RelationshipLayer, SocialIntelligence,
)
from social import (
    CommunityTracker, Interaction, InteractionKind, World, goodwill, interact,
)


def make(name="a", personality=Personality.CAUTIOUS, seed=0) -> Agent:
    npc = NPC(name)
    npc.activate()
    npc.individual.personality = personality
    return Agent(npc, Needs(), random.Random(seed))


def bond(x: Agent, y: Agent, t: float = 0.8) -> None:
    x.npc.social.adjust_trust(y.npc.npc_id, t)
    y.npc.social.adjust_trust(x.npc.npc_id, t)


def group(n, **kw):
    agents = [make(f"g{i}", seed=i, **kw) for i in range(n)]
    return agents, {a.npc.npc_id: a for a in agents}


# --- trust -------------------------------------------------------------------

@pytest.mark.parametrize("delta,expected", [
    (0.9, RelationshipLayer.FRIEND), (0.3, RelationshipLayer.ACQUAINTANCE),
    (0.0, RelationshipLayer.NEUTRAL), (-0.2, RelationshipLayer.NEUTRAL),
    (-0.6, RelationshipLayer.ENEMY),
])
def test_trust_maps_to_relationship(delta, expected):
    soc = SocialIntelligence()
    soc.adjust_trust("x", delta)
    assert soc.relationship_with("x") == expected


def test_trust_is_clamped_and_accumulates():
    soc = SocialIntelligence()
    for _ in range(20):
        soc.adjust_trust("x", 0.3)
    assert soc.trust_of("x") == 1.0
    soc.adjust_trust("x", -5)
    assert soc.trust_of("x") == -1.0 and soc.hostility_ratio() == 1.0


def test_loved_one_label_survives_while_trust_positive():
    soc = SocialIntelligence()
    soc.set_relationship("x", RelationshipLayer.LOVED_ONE)
    soc.adjust_trust("x", 0.9)
    assert soc.relationship_with("x") == RelationshipLayer.LOVED_ONE
    soc.adjust_trust("x", -1.5)
    assert soc.relationship_with("x") == RelationshipLayer.ENEMY


def test_unknown_npc_has_zero_trust():
    assert SocialIntelligence().trust_of("nobody") == 0.0


# --- interaction ---------------------------------------------------------------

class FixedRng(random.Random):
    """random() returns scripted values, then falls back to the base rng."""
    def __init__(self, *vals):
        super().__init__(0)
        self.vals = list(vals)

    def random(self):
        return self.vals.pop(0) if self.vals else super().random()


def test_goodwill_rises_with_trust_and_extroversion():
    a, b = make("a"), make("b")
    base = goodwill(a.npc, b.npc)
    bond(a, b, 0.8)
    assert goodwill(a.npc, b.npc) > base
    c, d = make("c", Personality.EXTROVERTED), make("d", Personality.EXTROVERTED)
    assert goodwill(c.npc, d.npc) > base
    assert 0.1 <= goodwill(a.npc, b.npc) <= 0.95


def test_rival_ambitions_lower_goodwill():
    amb = goodwill(make("a", Personality.AMBITIOUS).npc,
                   make("b", Personality.AMBITIOUS).npc)
    assert amb < goodwill(make("c").npc, make("d").npc)


def test_cooperation_builds_trust_memories_and_eases_loneliness():
    a, b = make("a"), make("b")
    b.needs.loneliness = 0.8
    out = interact(a, b, FixedRng(0.0, 0.9))   # good outcome, no gossip
    assert out.kind == InteractionKind.COOPERATE
    assert a.npc.social.trust_of(b.npc.npc_id) == 0.15
    assert b.npc.social.trust_of(a.npc.npc_id) == 0.15
    assert b.needs.loneliness == pytest.approx(0.6)
    mem = a.npc.individual.memories[-1]
    assert mem.emotion == EmotionLayer.HAPPY and f"npc:{b.npc.npc_id}" in mem.tags


def test_argument_costs_trust_and_creates_anger():
    a, b = make("a"), make("b")
    out = interact(a, b, FixedRng(0.99))
    assert out.kind == InteractionKind.ARGUE
    assert a.npc.social.trust_of(b.npc.npc_id) == -0.25
    assert a.npc.individual.emotion == EmotionLayer.ANGRY
    assert b.npc.individual.memories[-1].salience == 0.7


def test_repeated_arguments_make_enemies():
    a, b = make("a"), make("b")
    for _ in range(3):
        interact(a, b, FixedRng(0.99))
    assert a.npc.social.relationship_with(b.npc.npc_id) == RelationshipLayer.ENEMY


def test_gossip_spreads_reputation_scaled_by_trust():
    a, b, c = make("a"), make("b"), make("c")
    a.npc.social.adjust_trust(c.npc.npc_id, 0.9)      # a likes c
    b.npc.social.adjust_trust(a.npc.npc_id, 1.0)      # b trusts a fully
    out = interact(a, b, FixedRng(0.0, 0.0))          # good outcome, gossip
    assert out.kind == InteractionKind.GOSSIP
    assert b.npc.social.trust_of(c.npc.npc_id) == pytest.approx(0.25 * 0.9)


def test_gossip_from_untrusted_source_does_nothing():
    a, b, c = make("a"), make("b"), make("c")
    a.npc.social.adjust_trust(c.npc.npc_id, 0.9)
    interact(a, b, FixedRng(0.0, 0.0))                # b has no trust in a yet
    assert b.npc.social.trust_of(c.npc.npc_id) == 0.0


# --- communities -----------------------------------------------------------------

def test_mutual_trust_triangle_forms_community():
    agents, by_id = group(3)
    for i in range(3):
        for j in range(i + 1, 3):
            bond(agents[i], agents[j])
    t = CommunityTracker()
    out = t.update(by_id)
    assert out["formed"] == ["Community-1"]
    c = t.communities["Community-1"]
    assert c.members == set(by_id)
    for a in agents:
        assert a.npc.collective.alliances["Community-1"] == AllianceLayer.ALLIANCE
        assert a.npc.collective.leadership is not None
        assert a.npc.collective.institution == InstitutionLayer.DEMOCRACY


def test_two_circles_joined_by_a_bridge_stay_separate():
    agents, by_id = group(6)
    for lo, hi in ((0, 3), (3, 6)):
        for i in range(lo, hi):
            for j in range(i + 1, hi):
                bond(agents[i], agents[j])
    bond(agents[2], agents[3])                          # a lone bridge friendship
    t = CommunityTracker()
    t.update(by_id)
    sets = sorted(sorted(c.members) for c in t.communities.values())
    ids = [a.npc.npc_id for a in agents]
    assert sets == sorted([sorted(ids[:3]), sorted(ids[3:])])


def test_a_chain_of_friends_is_not_a_community():
    agents, by_id = group(5)
    for i in range(4):
        bond(agents[i], agents[i + 1])
    assert CommunityTracker().update(by_id)["formed"] == []


def test_one_way_trust_or_small_groups_are_not_communities():
    agents, by_id = group(3)
    bond(agents[0], agents[1])                         # pair only
    agents[1].npc.social.adjust_trust(agents[2].npc.npc_id, 0.9)   # one-way
    assert CommunityTracker().update(by_id)["formed"] == []


def test_leader_is_the_most_trusted():
    agents, by_id = group(3)
    for i in range(3):
        for j in range(i + 1, 3):
            bond(agents[i], agents[j], 0.6)
    for other in (agents[0], agents[1]):
        other.npc.social.adjust_trust(agents[2].npc.npc_id, 0.3)
    t = CommunityTracker()
    t.update(by_id)
    assert t.communities["Community-1"].leader == agents[2].npc.npc_id


def test_ambitious_leader_makes_autocracy_and_monarchy():
    agents, by_id = group(3)
    for i in range(3):
        for j in range(i + 1, 3):
            bond(agents[i], agents[j], 0.6)
    agents[0].npc.individual.personality = Personality.AMBITIOUS
    for other in agents[1:]:
        other.npc.social.adjust_trust(agents[0].npc.npc_id, 0.3)
    t = CommunityTracker()
    t.update(by_id)
    c = t.communities["Community-1"]
    assert (c.leadership, c.institution) == (LeadershipLayer.AUTOCRATIC,
                                             InstitutionLayer.MONARCHY)


def test_high_cohesion_is_consensual():
    agents, by_id = group(3)
    for i in range(3):
        for j in range(i + 1, 3):
            bond(agents[i], agents[j], 0.9)
    t = CommunityTracker()
    t.update(by_id)
    assert t.communities["Community-1"].leadership == LeadershipLayer.CONSENSUAL


def test_community_identity_persists_and_survives_member_churn():
    agents, by_id = group(4)
    for i in range(3):
        for j in range(i + 1, 3):
            bond(agents[i], agents[j])
    t = CommunityTracker()
    t.update(by_id)
    for i in range(3):                       # a fourth joins
        bond(agents[i], agents[3])
    out = t.update(by_id)
    assert out == {"formed": [], "dissolved": []}
    assert t.communities["Community-1"].members == set(by_id)


def test_community_dissolves_and_fields_are_cleared():
    agents, by_id = group(3)
    for i in range(3):
        for j in range(i + 1, 3):
            bond(agents[i], agents[j])
    t = CommunityTracker()
    t.update(by_id)
    for a in agents:
        for b in agents:
            if a is not b:
                a.npc.social.adjust_trust(b.npc.npc_id, -1.5)
    out = t.update(by_id)
    assert out["dissolved"] == ["Community-1"]
    for a in agents:
        col = a.npc.collective
        assert col.alliances == {} and col.leadership is None and col.institution is None


def test_dead_members_leave_the_community():
    agents, by_id = group(4)
    for i in range(4):
        for j in range(i + 1, 4):
            bond(agents[i], agents[j])
    t = CommunityTracker()
    t.update(by_id)
    agents[3].npc.die()
    t.update(by_id)
    assert agents[3].npc.npc_id not in t.communities["Community-1"].members


def test_rival_communities_become_hostile():
    agents, by_id = group(6)
    for lo, hi in ((0, 3), (3, 6)):
        for i in range(lo, hi):
            for j in range(i + 1, hi):
                bond(agents[i], agents[j])
    for a in agents[:3]:
        for b in agents[3:]:
            a.npc.social.adjust_trust(b.npc.npc_id, -0.6)
            b.npc.social.adjust_trust(a.npc.npc_id, -0.6)
    t = CommunityTracker()
    t.update(by_id)
    assert t.rivals() == [("Community-1", "Community-2")]
    assert agents[0].npc.collective.alliances["Community-2"] == AllianceLayer.HOSTILE
    assert agents[5].npc.collective.alliances["Community-1"] == AllianceLayer.HOSTILE


# --- world -------------------------------------------------------------------------

def build_world(seed=3, n=12) -> World:
    w = World(random.Random(seed))
    for i in range(n):
        npc = NPC(f"n{i}", npc_id=f"id{i:02d}")   # fixed ids => reproducible runs
        npc.individual.personality = (Personality.EXTROVERTED if i % 2
                                      else Personality.OPTIMISTIC)
        w.add(Agent(npc, Needs(), random.Random(i)))
    return w


def test_world_activates_added_npcs():
    w = World()
    w.add(Agent(NPC("a")))
    assert next(iter(w.agents.values())).npc.state == NPCState.ACTIVE


def test_world_is_reproducible():
    def run():
        w = build_world()
        for _ in range(100):
            w.step()
        return ([(i.kind, i.a, i.b) for i in w.interactions],
                {k: sorted(c.members) for k, c in w.tracker.communities.items()})
    assert run() == run()


def test_friendly_world_grows_communities_and_law():
    w = build_world()
    last = {}
    for _ in range(300):
        s = w.step()
        last = s.get("emergent", last)
    assert w.tracker.communities
    assert {i.kind for i in w.interactions} >= {InteractionKind.COOPERATE,
                                                InteractionKind.ARGUE}
    assert last["law"] == LawLayer.CONSTITUTION


def test_world_invariants_over_long_run():
    w = build_world(seed=11)
    for _ in range(400):
        w.step()
    for a in w.agents.values():
        soc = a.npc.social
        assert all(-1.0 <= t <= 1.0 for t in soc.trust.values())
        assert a.npc.npc_id not in soc.trust            # nobody befriends themselves
        assert all(0.0 <= v <= 1.0 for v in (a.needs.hunger, a.needs.fatigue,
                                             a.needs.loneliness))


def test_pairs_interact_at_most_once_per_tick():
    w = build_world(seed=5)
    for _ in range(50):
        s = w.step()
        pairs = [frozenset((i.a, i.b)) for i in s["interactions"]]
        assert len(pairs) == len(set(pairs))


def test_dead_npcs_are_ignored_by_the_world():
    w = build_world()
    victim = next(iter(w.agents.values()))
    victim.npc.die()
    for _ in range(50):
        s = w.step()
        assert all(victim.npc.npc_id not in (i.a, i.b) for i in s["interactions"])
