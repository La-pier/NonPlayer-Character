"""Rule-based autonomy for NPCs: needs -> perceive -> decide -> act -> remember.

Deliberately simple: a weighted scoring function over a handful of actions.
The weights are placeholder heuristics to tune for your game.
"""
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from adaptation import TraitProfile, learn, reflect, sentiment
from classes import (
    NPC, TICKING_STATES, BeliefLayer, EmotionLayer, GoalLayer, Personality,
    RelationshipLayer,
)


class Action(Enum):
    REST = "rest"
    SEEK_FOOD = "seek_food"
    SOCIALIZE = "socialize"
    AVOID = "avoid"
    PURSUE_GOAL = "pursue_goal"


@dataclass
class Needs:
    """Urgencies in 0..1; higher means more pressing. They rise every tick."""
    hunger: float = 0.2
    fatigue: float = 0.2
    loneliness: float = 0.2
    hunger_rate: float = 0.05
    fatigue_rate: float = 0.04
    loneliness_rate: float = 0.03

    def tick(self) -> None:
        self.hunger = _clamp(self.hunger + self.hunger_rate)
        self.fatigue = _clamp(self.fatigue + self.fatigue_rate)
        self.loneliness = _clamp(self.loneliness + self.loneliness_rate)


@dataclass
class StepRecord:
    """Everything notable that happened to one agent in one tick."""
    tick: int
    action: Action
    partner: Optional[str]
    what: str                       # the event text that went into memory
    threat: float
    nearby: List[str]
    needs: Dict[str, float]         # after acting
    emotion: EmotionLayer
    personality_before: Personality
    personality_after: Personality
    changes: Dict[str, Any]         # layer updates: new goals, forgotten memories...
    reflection: Dict[str, Any]      # belief / goal revisions, if it reflected


@dataclass
class Perception:
    """What an NPC can sense this tick."""
    threat: float = 0.0                 # 0 (calm) .. 1 (mortal danger)
    nearby: List[str] = field(default_factory=list)   # npc_ids in range
    food_available: bool = False


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


PERSONALITY_BIAS: Dict[Personality, Dict[Action, float]] = {
    Personality.BOLD: {Action.AVOID: -0.3},
    Personality.CAUTIOUS: {Action.AVOID: 0.2},
    Personality.SHY: {Action.AVOID: 0.1, Action.SOCIALIZE: -0.2},
    Personality.INTROVERTED: {Action.SOCIALIZE: -0.2},
    Personality.EXTROVERTED: {Action.SOCIALIZE: 0.2},
    Personality.AMBITIOUS: {Action.PURSUE_GOAL: 0.2},
}


@dataclass
class Agent:
    """An NPC plus the state it needs to act on its own."""
    npc: NPC
    needs: Needs = field(default_factory=Needs)
    rng: random.Random = field(default_factory=random.Random)
    tie_margin: float = 0.05   # actions within this of the best may be chosen
    reflect_every: int = 10    # ticks between reflections
    traits: Optional[TraitProfile] = None
    ticks: int = 0
    last_action: Optional[Action] = None
    last_partner: Optional[str] = None   # npc_id socialized with this tick
    last_step: Optional[StepRecord] = None

    def __post_init__(self) -> None:
        if self.traits is None:
            self.traits = TraitProfile.from_personality(
                self.npc.individual.personality)

    # -- decide ------------------------------------------------------------
    def effective_threat(self, p: Perception) -> float:
        soc = self.npc.social
        enemy_near = any(soc.relationship_with(i) == RelationshipLayer.ENEMY
                         for i in p.nearby)
        return max(p.threat, 0.6) if enemy_near else p.threat

    def score(self, p: Perception) -> Dict[Action, float]:
        ind, soc = self.npc.individual, self.npc.social
        n = self.needs
        threat = self.effective_threat(p)

        fear = threat * (1.3 if ind.emotion == EmotionLayer.FEARFUL else 1.0)
        scores = {
            Action.AVOID: fear * 1.5,
            Action.SEEK_FOOD: n.hunger * (1.0 if p.food_available else 0.3),
            Action.REST: n.fatigue,
            Action.SOCIALIZE: 0.0,
            Action.PURSUE_GOAL: 0.3 if ind.goals else 0.0,
        }
        if p.nearby:
            friend = any(soc.relationship_with(i) == RelationshipLayer.FRIEND
                         for i in p.nearby)
            # Past experience with these particular NPCs colours the choice.
            feeling = sentiment(ind, [f"npc:{i}" for i in p.nearby])
            scores[Action.SOCIALIZE] = (n.loneliness + (0.2 if friend else 0.0)
                                        + (0.1 if ind.emotion == EmotionLayer.HAPPY
                                           else 0.0) + 0.3 * feeling)
        if threat > 0 and BeliefLayer.PESSIMISTIC in ind.beliefs:
            scores[Action.AVOID] += 0.1
        for action, bias in PERSONALITY_BIAS.get(ind.personality, {}).items():
            scores[action] += bias
        # Biases must not make an unavailable action attractive.
        if not p.nearby:
            scores[Action.SOCIALIZE] = 0.0
        if not ind.goals:
            scores[Action.PURSUE_GOAL] = 0.0
        for action in scores:
            scores[action] = max(0.0, scores[action])
        return scores

    def decide(self, p: Perception) -> Action:
        scores = self.score(p)
        best = max(scores.values())
        # Tie-break among near-best actions so identical NPCs don't move in lockstep.
        # Sorted for reproducibility with a seeded rng.
        top = sorted((a for a, s in scores.items() if s >= best - self.tie_margin),
                     key=lambda a: a.value)
        return self.rng.choice(top)

    def _choose_partner(self, nearby: List[str]) -> str:
        """Prefer people it trusts, but anyone nearby can be picked."""
        ids = sorted(nearby)
        soc = self.npc.social
        weights = [max(0.05, soc.trust_of(i) + 1.1) for i in ids]
        return self.rng.choices(ids, weights=weights, k=1)[0]

    # -- act ---------------------------------------------------------------
    def act(self, action: Action, p: Perception) -> Dict[str, Any]:
        """Apply the action's effects on needs; return an event for memory."""
        n = self.needs
        ind, soc = self.npc.individual, self.npc.social
        event: Dict[str, Any] = {"content": action.value, "salience": 0.3}

        if action == Action.SEEK_FOOD:
            fed = p.food_available
            n.hunger = _clamp(n.hunger - (0.5 if fed else 0.1))
            event.update(content="found food" if fed else "searched for food, found little",
                         emotion=EmotionLayer.RELIEVED if fed else EmotionLayer.SAD,
                         salience=0.4 if fed else 0.5, tags=("food",))
        elif action == Action.REST:
            n.fatigue = _clamp(n.fatigue - 0.5)
            event.update(content="rested")
        elif action == Action.SOCIALIZE and p.nearby:
            n.loneliness = _clamp(n.loneliness - 0.4)
            other = self._choose_partner(p.nearby)
            self.last_partner = other
            if soc.relationship_with(other) == RelationshipLayer.STRANGER:
                soc.set_relationship(other, RelationshipLayer.ACQUAINTANCE)
            event.update(content=f"spent time with {other}",
                         emotion=EmotionLayer.HAPPY, salience=0.4,
                         tags=("social", f"npc:{other}"))
        elif action == Action.AVOID:
            n.fatigue = _clamp(n.fatigue + 0.05)   # running is tiring
            event.update(content="avoided danger", emotion=EmotionLayer.FEARFUL,
                         salience=0.6, tags=("danger",))
        elif action == Action.PURSUE_GOAL and ind.goals:
            event.update(content=f"worked toward {ind.goals[0].value}", salience=0.3)
        return event

    # -- one tick ----------------------------------------------------------
    def step(self, p: Optional[Perception] = None,
             context: Optional[Dict[str, Any]] = None) -> Optional[Action]:
        """Needs rise, the NPC picks and performs an action, layers update.

        Returns None (and does nothing) unless the NPC is in a ticking state.
        `context["tick"]` is the simulation clock (defaults to this agent's own
        step count); it stamps the memories made. The step is described in
        `last_step`.
        """
        self.last_step = None
        if self.npc.state not in TICKING_STATES:
            return None
        p = p or Perception()
        ctx = dict(context or {})
        tick = ctx.setdefault("tick", self.ticks + 1)
        before = self.npc.individual.personality
        self.needs.tick()
        action = self.decide(p)
        self.last_partner = None
        event = self.act(action, p)
        ctx["events"] = list(ctx.get("events", [])) + [event]
        layer_changes = self.npc.tick(ctx)
        self.last_action = action
        reflection = self._adapt(action, event)
        n = self.needs
        self.last_step = StepRecord(
            tick=tick, action=action, partner=self.last_partner,
            what=event["content"], threat=p.threat, nearby=list(p.nearby),
            needs={"hunger": n.hunger, "fatigue": n.fatigue,
                   "loneliness": n.loneliness},
            emotion=self.npc.individual.emotion, personality_before=before,
            personality_after=self.npc.individual.personality,
            changes=layer_changes.get("individual", {}), reflection=reflection)
        return action

    def _adapt(self, action: Action, event: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from the outcome; periodically reflect on recent experience."""
        ind = self.npc.individual
        self.ticks += 1
        self.traits.tick()
        learn(self.traits, action.value, event.get("emotion"))
        ind.personality = self.traits.dominant(ind.personality)
        if self.ticks % self.reflect_every == 0:
            return reflect(ind)
        return {}
