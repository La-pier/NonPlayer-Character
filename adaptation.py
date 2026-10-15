"""Learning for NPCs: evolving traits, memory retrieval, and reflection.

Like autonomy.py this is rule-based; the constants are placeholder heuristics.
"""
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set

from classes import (
    BeliefLayer, EmotionLayer, GoalLayer, IndividualIntelligence, MemoryEntry,
    Personality, ReflectionLayer,
)

# -- emotional valence -------------------------------------------------------

VALENCE: Dict[EmotionLayer, float] = {
    EmotionLayer.HAPPY: 1.0, EmotionLayer.RELIEVED: 0.5, EmotionLayer.AWED: 0.5,
    EmotionLayer.SURPRISED: 0.0, EmotionLayer.CONFUSED: -0.2,
    EmotionLayer.SAD: -0.5, EmotionLayer.FEARFUL: -1.0, EmotionLayer.ANGRY: -1.0,
}


def valence(m: MemoryEntry) -> float:
    return VALENCE.get(m.emotion, 0.0) if m.emotion is not None else 0.0


# -- memory retrieval --------------------------------------------------------

def retrieve(ind: IndividualIntelligence, tags: Iterable[str] = (), k: int = 5,
             recency_decay: float = 0.9) -> List[MemoryEntry]:
    """Top-k memories ranked by salience, recency and tag overlap.

    Recency is by position (memories are appended in order), not wall-clock,
    so results are deterministic inside a fast simulation loop.
    """
    query: Set[str] = set(tags)
    n = len(ind.memories)
    scored = []
    for i, m in enumerate(ind.memories):
        recency = recency_decay ** (n - 1 - i)
        overlap = len(query & set(m.tags)) / len(query) if query else 0.0
        if query and overlap == 0.0:
            continue   # with a query, only matching memories are candidates
        scored.append((0.4 * m.salience + 0.3 * recency + 0.3 * overlap, i, m))
    scored.sort(key=lambda t: (-t[0], -t[1]))
    return [m for _, _, m in scored[:k]]


def sentiment(ind: IndividualIntelligence, tags: Iterable[str], k: int = 5) -> float:
    """Average valence of the retrieved memories matching tags (0 if none)."""
    found = retrieve(ind, tags, k)
    return sum(valence(m) for m in found) / len(found) if found else 0.0


# -- evolving personality ----------------------------------------------------

OPPOSITES: Dict[Personality, Personality] = {
    Personality.BOLD: Personality.CAUTIOUS, Personality.CAUTIOUS: Personality.BOLD,
    Personality.EXTROVERTED: Personality.INTROVERTED,
    Personality.INTROVERTED: Personality.EXTROVERTED,
    Personality.OPTIMISTIC: Personality.PESSIMISTIC,
    Personality.PESSIMISTIC: Personality.OPTIMISTIC,
}


@dataclass
class TraitProfile:
    """Strength (0..1) of every personality trait; they shift with experience.

    `Personality` stays the NPC's dominant trait; this holds the whole mix.
    """
    strengths: Dict[Personality, float] = field(
        default_factory=lambda: {p: BASELINE for p in Personality})
    switch_margin: float = 0.1   # hysteresis: a new trait must beat the old by this
    drift: float = 0.01          # pull toward BASELINE each tick

    @classmethod
    def from_personality(cls, p: Personality) -> "TraitProfile":
        prof = cls()
        prof.strengths[p] = 0.7
        return prof

    def reinforce(self, trait: Personality, amount: float) -> None:
        self.strengths[trait] = _clamp(self.strengths[trait] + amount)
        opp = OPPOSITES.get(trait)
        if opp is not None:
            self.strengths[opp] = _clamp(self.strengths[opp] - amount / 2)

    def tick(self) -> None:
        for p, v in self.strengths.items():
            self.strengths[p] = v + (BASELINE - v) * self.drift

    def dominant(self, current: Personality) -> Personality:
        best = max(self.strengths, key=lambda p: (self.strengths[p], p.value))
        if self.strengths[best] > self.strengths[current] + self.switch_margin:
            return best
        return current


BASELINE = 0.3


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


# What an (action, felt emotion) outcome teaches. Keys are action values so this
# module doesn't import autonomy (which imports it).
LESSONS: Dict[tuple, List[tuple]] = {
    ("avoid", EmotionLayer.FEARFUL): [(Personality.CAUTIOUS, 0.03)],
    ("socialize", EmotionLayer.HAPPY): [(Personality.EXTROVERTED, 0.03)],
    ("seek_food", EmotionLayer.RELIEVED): [(Personality.OPTIMISTIC, 0.02)],
    ("seek_food", EmotionLayer.SAD): [(Personality.PESSIMISTIC, 0.03)],
    ("pursue_goal", None): [(Personality.AMBITIOUS, 0.02)],
}


def learn(profile: TraitProfile, action: str, emotion: Optional[EmotionLayer]) -> None:
    """Apply the lesson for (action, emotion), else the emotion-agnostic one."""
    lessons = LESSONS.get((action, emotion)) or LESSONS.get((action, None), [])
    for trait, amount in lessons:
        profile.reinforce(trait, amount)


# -- reflection --------------------------------------------------------------

def reflect(ind: IndividualIntelligence, window: int = 20,
            min_memories: int = 5) -> Dict[str, object]:
    """Interpret recent experience; revise beliefs and form intentions.

    Mostly-bad recent memories push toward a pessimistic belief and a wish for
    peace; mostly-good ones push toward optimism and drop pessimism (a revision).
    Returns what changed.
    """
    recent = ind.memories[-window:]
    changes: Dict[str, object] = {}
    if len(recent) < min_memories:
        return changes
    avg = sum(valence(m) for m in recent) / len(recent)
    social_joy = sum(1 for m in recent
                     if valence(m) > 0 and any(t.startswith("npc:") for t in m.tags))

    if avg <= -0.3:
        ind.reflection = ReflectionLayer.THOROUGH
        if BeliefLayer.PESSIMISTIC not in ind.beliefs:
            ind.beliefs.add(BeliefLayer.PESSIMISTIC)
            changes["adopted_belief"] = BeliefLayer.PESSIMISTIC
        if BeliefLayer.OPTIMISTIC in ind.beliefs:
            ind.beliefs.discard(BeliefLayer.OPTIMISTIC)
            changes["revised_belief"] = BeliefLayer.OPTIMISTIC
        if GoalLayer.PEACE not in ind.goals:
            ind.goals.append(GoalLayer.PEACE)
            changes["new_goal"] = GoalLayer.PEACE
    elif avg >= 0.3:
        ind.reflection = ReflectionLayer.REFLECTIVE
        if BeliefLayer.OPTIMISTIC not in ind.beliefs:
            ind.beliefs.add(BeliefLayer.OPTIMISTIC)
            changes["adopted_belief"] = BeliefLayer.OPTIMISTIC
        if BeliefLayer.PESSIMISTIC in ind.beliefs:
            ind.beliefs.discard(BeliefLayer.PESSIMISTIC)
            changes["revised_belief"] = BeliefLayer.PESSIMISTIC
    else:
        ind.reflection = ReflectionLayer.SHALLOW

    if social_joy >= 3 and GoalLayer.CONNECTION not in ind.goals:
        ind.goals.append(GoalLayer.CONNECTION)
        changes.setdefault("new_goal", GoalLayer.CONNECTION)
    return changes
