"""NPC-to-NPC interaction and emergent communities.

NPCs meet, cooperate / argue / gossip, and build or lose trust. Groups of
mutually trusting NPCs become communities with a leader and an institution,
which the world-level EmergentIntelligence turns into law. Rule-based; the
numbers are placeholder heuristics to tune.
"""
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from autonomy import Action, Agent, Perception
from classes import (
    NPC, AllianceLayer, EmergentIntelligence, EmotionLayer, InstitutionLayer,
    LeadershipLayer, NPCState, Personality,
)


class InteractionKind(Enum):
    COOPERATE = "cooperate"
    GOSSIP = "gossip"
    ARGUE = "argue"


@dataclass
class Interaction:
    kind: InteractionKind
    a: str
    b: str


# -- interaction ---------------------------------------------------------------

PERSONALITY_MOOD = {
    Personality.EXTROVERTED: 0.08, Personality.OPTIMISTIC: 0.05,
    Personality.PESSIMISTIC: -0.05, Personality.BOLD: -0.03,
}


def goodwill(a: NPC, b: NPC) -> float:
    """Chance an encounter goes well, from existing trust and temperaments."""
    p = 0.5 + 0.35 * (a.social.trust_of(b.npc_id) + b.social.trust_of(a.npc_id)) / 2
    for npc in (a, b):
        p += PERSONALITY_MOOD.get(npc.individual.personality, 0.0)
    if (a.individual.personality == Personality.AMBITIOUS
            and b.individual.personality == Personality.AMBITIOUS):
        p -= 0.1   # rivals for the same things
    return max(0.1, min(0.95, p))


def _record(npc: NPC, other: NPC, text: str, emotion: EmotionLayer,
            salience: float) -> None:
    ind = npc.individual
    ind.emotion = emotion
    ind.remember(text, emotion, salience, tags=("social", f"npc:{other.npc_id}"))


def interact(a: Agent, b: Agent, rng: random.Random) -> Interaction:
    """One encounter between two agents; updates trust, memory and needs."""
    na, nb = a.npc, b.npc
    if rng.random() < goodwill(na, nb):
        gossip = _gossip_topics(na, nb)
        if gossip and rng.random() < 0.3:
            kind = InteractionKind.GOSSIP
            _share_reputations(na, nb, gossip)
            _record(na, nb, "gossiped with", EmotionLayer.HAPPY, 0.4)
            _record(nb, na, "gossiped with", EmotionLayer.HAPPY, 0.4)
        else:
            kind = InteractionKind.COOPERATE
            _record(na, nb, "worked together with", EmotionLayer.HAPPY, 0.5)
            _record(nb, na, "worked together with", EmotionLayer.HAPPY, 0.5)
        na.social.adjust_trust(nb.npc_id, 0.15)
        nb.social.adjust_trust(na.npc_id, 0.15)
        b.needs.loneliness = max(0.0, b.needs.loneliness - 0.2)
    else:
        kind = InteractionKind.ARGUE
        _record(na, nb, "argued with", EmotionLayer.ANGRY, 0.7)
        _record(nb, na, "argued with", EmotionLayer.ANGRY, 0.7)
        na.social.adjust_trust(nb.npc_id, -0.25)
        nb.social.adjust_trust(na.npc_id, -0.25)
    return Interaction(kind, na.npc_id, nb.npc_id)


def _gossip_topics(teller: NPC, listener: NPC) -> List[str]:
    """Third parties the teller has a strong opinion of (up to two)."""
    opinions = [(abs(t), i) for i, t in sorted(teller.social.trust.items())
                if i not in (listener.npc_id, teller.npc_id) and abs(t) >= 0.4]
    opinions.sort(key=lambda x: (-x[0], x[1]))
    return [i for _, i in opinions[:2]]


def _share_reputations(teller: NPC, listener: NPC, topics: List[str]) -> None:
    """Listener's view of a third party moves toward the teller's, scaled by
    how much the listener trusts the teller."""
    weight = max(0.0, listener.social.trust_of(teller.npc_id))
    for third in topics:
        listener.social.adjust_trust(third, 0.25 * teller.social.trust_of(third) * weight)


# -- communities -----------------------------------------------------------------

@dataclass
class Community:
    name: str
    members: Set[str]
    leader: str
    leadership: LeadershipLayer
    institution: InstitutionLayer
    cohesion: float


class CommunityTracker:
    """Finds groups of mutually trusting NPCs and records them on each member.

    A community is a connected group (>= min_size) of mutual trust >= bond
    links that are each backed by a shared friend. Identity persists while
    membership overlaps.
    """

    PREFIX = "Community-"

    def __init__(self, min_size: int = 3, bond: float = 0.5,
                 rival_trust: float = -0.2) -> None:
        self.min_size = min_size
        self.bond = bond
        self.rival_trust = rival_trust
        self.communities: Dict[str, Community] = {}
        self._counter = 0
        self._assigned: Set[str] = set()   # npc_ids whose leadership/institution we set
        self._rival_pairs: List[Tuple[str, str]] = []

    # -- detection
    def _components(self, agents: Dict[str, Agent]) -> List[Set[str]]:
        """Connected groups of triangle-supported mutual-trust links.

        A link (mutual trust >= bond) only counts if the two also share a
        mutual friend, repeated until stable (a 3-truss). That keeps tight
        circles of friends together but cuts bridges and loose chains.
        """
        alive = sorted(i for i, ag in agents.items() if ag.npc.is_alive)

        def linked(x: str, y: str) -> bool:
            return (agents[x].npc.social.trust_of(y) >= self.bond
                    and agents[y].npc.social.trust_of(x) >= self.bond)

        edges = {i: {j for j in alive if j != i and linked(i, j)} for i in alive}
        changed = True
        while changed:
            changed = False
            for i in alive:
                keep = {j for j in edges[i] if edges[i] & edges[j]}
                if keep != edges[i]:
                    edges[i], changed = keep, True
            # drop one-sided edges left over from asymmetric pruning
            for i in alive:
                keep = {j for j in edges[i] if i in edges[j]}
                if keep != edges[i]:
                    edges[i], changed = keep, True

        seen: Set[str] = set()
        groups: List[Set[str]] = []
        for start in alive:
            if start in seen or not edges[start]:
                continue
            group, stack = set(), [start]
            while stack:
                cur = stack.pop()
                if cur not in group:
                    group.add(cur)
                    stack.extend(sorted(edges[cur] - group))
            seen |= group
            if len(group) >= self.min_size:
                groups.append(group)
        return groups

    @staticmethod
    def _cohesion(members: Set[str], agents: Dict[str, Agent]) -> float:
        ids = sorted(members)
        pairs = [(x, y) for x in ids for y in ids if x != y]
        return sum(agents[x].npc.social.trust_of(y) for x, y in pairs) / len(pairs)

    def _describe(self, name: str, members: Set[str],
                  agents: Dict[str, Agent]) -> Community:
        ids = sorted(members)
        incoming = {m: sum(agents[o].npc.social.trust_of(m) for o in ids if o != m)
                    for m in ids}
        leader = max(ids, key=lambda m: (incoming[m], m))
        cohesion = self._cohesion(members, agents)
        if agents[leader].npc.individual.personality == Personality.AMBITIOUS:
            leadership, inst = LeadershipLayer.AUTOCRATIC, InstitutionLayer.MONARCHY
        elif cohesion >= 0.7:
            leadership, inst = LeadershipLayer.CONSENSUAL, InstitutionLayer.DEMOCRACY
        else:
            leadership, inst = LeadershipLayer.DEMOCRATIC, InstitutionLayer.DEMOCRACY
        return Community(name, set(members), leader, leadership, inst, cohesion)

    # -- update
    def update(self, agents: Dict[str, Agent]) -> Dict[str, List[str]]:
        old = self.communities
        unmatched = dict(old)
        fresh: Dict[str, Community] = {}
        formed: List[str] = []
        for group in self._components(agents):
            best, best_j = None, 0.0
            for name, c in sorted(unmatched.items()):
                j = len(group & c.members) / len(group | c.members)
                if j > best_j:
                    best, best_j = name, j
            if best is not None and best_j >= 0.5:
                name = best
                del unmatched[name]
            else:
                self._counter += 1
                name = f"{self.PREFIX}{self._counter}"
                formed.append(name)
            fresh[name] = self._describe(name, group, agents)
        self.communities = fresh
        self._apply(agents)
        return {"formed": formed, "dissolved": sorted(unmatched)}

    def rivals(self) -> List[Tuple[str, str]]:
        return list(self._rival_pairs)

    def _apply(self, agents: Dict[str, Agent]) -> None:
        # Rebuild every community-derived field from scratch: idempotent.
        for ag in agents.values():
            col = ag.npc.collective
            for key in [k for k in col.alliances if k.startswith(self.PREFIX)]:
                del col.alliances[key]
            if ag.npc.npc_id in self._assigned:
                col.leadership = None
                col.institution = None
        self._assigned = set()

        for c in self.communities.values():
            for m in c.members:
                col = agents[m].npc.collective
                col.alliances[c.name] = AllianceLayer.ALLIANCE
                col.leadership = c.leadership
                col.institution = c.institution
                self._assigned.add(m)

        self._rival_pairs = []
        names = sorted(self.communities)
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                cx, cy = self.communities[x], self.communities[y]
                cross = [agents[p].npc.social.trust_of(q)
                         for p in cx.members for q in cy.members]
                cross += [agents[q].npc.social.trust_of(p)
                          for p in cx.members for q in cy.members]
                if sum(cross) / len(cross) < self.rival_trust:
                    self._rival_pairs.append((x, y))
                    for m in cx.members:
                        agents[m].npc.collective.alliances[y] = AllianceLayer.HOSTILE
                    for m in cy.members:
                        agents[m].npc.collective.alliances[x] = AllianceLayer.HOSTILE


# -- world ---------------------------------------------------------------------------

@dataclass
class World:
    """A population of agents that meet, interact and form communities.

    Runs are reproducible given a seeded `rng`, seeded agent rngs AND explicit
    `npc_id`s: ordering decisions sort by id, and default ids are random UUIDs.
    """
    rng: random.Random = field(default_factory=random.Random)
    meet_base: float = 0.15       # chance to run into any given NPC
    meet_friend: float = 0.3      # extra chance for each point of trust
    max_nearby: int = 3
    food_rate: float = 0.4
    threat_rate: float = 0.05
    observe_every: int = 10
    agents: Dict[str, Agent] = field(default_factory=dict)
    emergent: EmergentIntelligence = field(default_factory=EmergentIntelligence)
    tracker: CommunityTracker = field(default_factory=CommunityTracker)
    tick: int = 0
    interactions: List[Interaction] = field(default_factory=list)

    def add(self, agent: Agent) -> None:
        if agent.npc.state == NPCState.INITIAL:
            agent.npc.activate()
        self.agents[agent.npc.npc_id] = agent
        self.emergent.add(agent.npc)

    def _perceive(self, me: Agent) -> Perception:
        soc = me.npc.social
        others = [o for o in sorted(self.agents)
                  if o != me.npc.npc_id and self.agents[o].npc.is_alive]
        near = [o for o in others
                if self.rng.random() < self.meet_base
                + self.meet_friend * max(0.0, soc.trust_of(o))]
        self.rng.shuffle(near)
        threat = (0.3 + 0.5 * self.rng.random()
                  if self.rng.random() < self.threat_rate else 0.0)
        return Perception(threat=threat, nearby=near[:self.max_nearby],
                          food_available=self.rng.random() < self.food_rate)

    def step(self) -> Dict[str, object]:
        self.tick += 1
        order = sorted(self.agents)
        self.rng.shuffle(order)
        done: Set[frozenset] = set()
        happened: List[Interaction] = []
        for npc_id in order:
            agent = self.agents[npc_id]
            action = agent.step(self._perceive(agent))
            partner = agent.last_partner
            if action != Action.SOCIALIZE or partner is None:
                continue
            other = self.agents.get(partner)
            pair = frozenset((npc_id, partner))
            if other is None or not other.npc.is_alive or pair in done:
                continue
            done.add(pair)
            happened.append(interact(agent, other, self.rng))
        self.interactions.extend(happened)
        events = self.tracker.update(self.agents)
        summary: Dict[str, object] = {"interactions": happened, **events}
        if self.tick % self.observe_every == 0:
            summary["emergent"] = self.emergent.observe()
        return summary
