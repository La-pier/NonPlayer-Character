from collections import Counter
from dataclasses import dataclass, field
from uuid import uuid4
from typing import ClassVar, Dict, Iterable, List, Any, Optional, Set, Tuple
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime
from copy import deepcopy


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------

class IntelligenceLayer(Enum):
    """Enumeration of intelligence layers in the hierarchy."""
    INDIVIDUAL = "individual"
    SOCIAL = "social"
    COLLECTIVE = "collective"
    EMERGENT = "emergent"


# ---------------------------------------------------------------------------
# Individual intelligence
# ---------------------------------------------------------------------------

class Personality(Enum):
    """Individual personality types."""
    INTROVERTED = "introverted"
    EXTROVERTED = "extroverted"
    SHY = "shy"
    BOLD = "bold"
    AMBITIOUS = "ambitious"
    CAUTIOUS = "cautious"
    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"

    def get_effects(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply personality traits to context."""
        effects: Dict[str, Any] = {}
        if self == Personality.AMBITIOUS:
            effects["goals"] = ["power", "wealth", "domination"]
        elif self == Personality.OPTIMISTIC:
            effects["goals"] = ["survival", "knowledge", "connection"]
        elif self == Personality.PESSIMISTIC:
            effects["goals"] = ["survival", "power", "domination"]
        return effects


class MemoryLayer(Enum):
    """Individual memory types."""
    SHORT_TERM = "short-term"
    LONG_TERM = "long-term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class BeliefLayer(Enum):
    """Individual belief types."""
    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"
    REALISTIC = "realistic"
    IDEALISTIC = "idealistic"
    PRAGMATIC = "pragmatic"
    MYSTICAL = "mystical"
    MATERIALIST = "materialist"
    RELIGIOUS = "religious"


class EmotionLayer(Enum):
    """Individual emotion types."""
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    CONFUSED = "confused"
    RELIEVED = "relieved"
    AWED = "awed"


class ReflectionLayer(Enum):
    """Individual reflection types."""
    REFLECTIVE = "reflective"
    SHALLOW = "shallow"
    DEEP = "deep"
    CRITICAL = "critical"
    THOROUGH = "thorough"
    HASTY = "hasty"


class GoalLayer(Enum):
    """Individual goal types."""
    SURVIVAL = "survival"
    POWER = "power"
    FREEDOM = "freedom"
    WEALTH = "wealth"
    KNOWLEDGE = "knowledge"
    CONNECTION = "connection"
    PEACE = "peace"
    DOMINATION = "domination"


# ---------------------------------------------------------------------------
# Social intelligence
# ---------------------------------------------------------------------------

class SocialLayer(Enum):
    """Social intelligence types."""
    RELATIONSHIPS = "relationships"
    COMMUNICATION = "communication"
    COOPERATION = "cooperation"
    CONFLICT = "conflict"
    INFLUENCE = "influence"


class RelationshipLayer(Enum):
    """Social relationship types."""
    ACQUAINTANCE = "acquaintance"
    FRIEND = "friend"
    LOVED_ONE = "loved_one"
    ENEMY = "enemy"
    NEUTRAL = "neutral"
    STRANGER = "stranger"


class CommunicationLayer(Enum):
    """Social communication types."""
    VERBAL = "verbal"
    NON_VERBAL = "non-verbal"
    WRITTEN = "written"
    EMOTIONAL = "emotional"
    RITUALISTIC = "ritualistic"


class CooperationLayer(Enum):
    """Social cooperation types."""
    COOPERATIVE = "cooperative"
    COMPETITIVE = "competitive"
    AMBIGUOUS = "ambiguous"
    EXPLOITATIVE = "exploitative"
    COLLECTIVE = "collective"


class ConflictLayer(Enum):
    """Social conflict types."""
    PEACEFUL = "peaceful"
    HOSTILE = "hostile"
    COMPETITIVE = "competitive"
    EXPLOITATIVE = "exploitative"
    NEUTRAL = "neutral"


class InfluenceLayer(Enum):
    """Social influence types."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    DOMINANT = "dominant"
    NEGLIGIBLE = "negligible"


# ---------------------------------------------------------------------------
# Collective intelligence
# ---------------------------------------------------------------------------

class HouseholdLayer(Enum):
    """Collective household types."""
    SINGLE = "single"
    EXTENDED = "extended"
    CLAN = "clan"
    FAMILY = "family"
    NUCLEAR = "nuclear"
    EXTENDED_FAMILY = "extended_family"


class FamilyLayer(Enum):
    """Collective family types."""
    BLOOD = "blood"
    ADOPTIVE = "adoptive"
    ADOPTED = "adopted"
    CHOSEN = "chosen"
    SPIRITUAL = "spiritual"
    LEGAL = "legal"


class FactionLayer(Enum):
    """Collective faction types."""
    MILITARY = "military"
    POLITICAL = "political"
    RELIGIOUS = "religious"
    ECONOMIC = "economic"
    INTELLECTUAL = "intellectual"
    CRIMINAL = "criminal"
    REVOLUTIONARY = "revolutionary"


class AllianceLayer(Enum):
    """Collective alliance types."""
    ALLIANCE = "alliance"
    PARTNERSHIP = "partnership"
    CONSPIRACY = "conspiracy"
    NEUTRAL = "neutral"
    HOSTILE = "hostile"


class LeadershipLayer(Enum):
    """Collective leadership types."""
    DEMOCRATIC = "democratic"
    AUTOCRATIC = "autocratic"
    CONSENSUAL = "consensual"
    HEREDITARY = "hereditary"
    ELECTORAL = "electoral"
    MILITARY = "military"


class InstitutionLayer(Enum):
    """Collective institution types."""
    MONARCHY = "monarchy"
    DEMOCRACY = "democracy"
    THEOCRACY = "theocracy"
    FEUDALISM = "feudalism"
    CORPORATION = "corporation"
    GUILD = "guild"
    STATE = "state"


# ---------------------------------------------------------------------------
# Emergent intelligence
# ---------------------------------------------------------------------------

class CultureLayer(Enum):
    """Emergent culture types."""
    INDIGENOUS = "indigenous"
    COLONIAL = "colonial"
    MESTIZO = "mestizo"
    SHAMANISTIC = "shamanistic"
    TECHNOCRATIC = "technocratic"
    URBAN = "urban"
    RURAL = "rural"


class TraditionLayer(Enum):
    """Emergent tradition types."""
    RITUALISTIC = "ritualistic"
    AGRICULTURAL = "agricultural"
    INDUSTRIAL = "industrial"
    TECHNOLOGICAL = "technological"
    RELIGIOUS = "religious"
    SECULAR = "secular"
    MYSTICAL = "mystical"


class EconomyLayer(Enum):
    """Emergent economy types."""
    SUBSISTENCE = "subsistence"
    AGRICULTURAL = "agricultural"
    INDUSTRIAL = "industrial"
    TECHNOLOGICAL = "technological"
    COMMERCIAL = "commercial"
    MILITARY = "military"
    MERITOCRATIC = "meritocratic"


class LawLayer(Enum):
    """Emergent law types."""
    CONSTITUTION = "constitution"
    CUSTOM = "custom"
    RELIGIOUS = "religious"
    CORPORATIST = "corporatist"
    MILITARY = "military"
    THEOCRATIC = "theocratic"
    SOCIAL = "social"


class SocialTransformationLayer(Enum):
    """Emergent social transformation types."""
    STAGNANT = "stagnant"
    GRADUAL = "gradual"
    RAPID = "rapid"
    REVOLUTIONARY = "revolutionary"
    REFORMIST = "reformist"
    CONSERVATIVE = "conservative"


# ---------------------------------------------------------------------------
# NPC lifecycle
# ---------------------------------------------------------------------------

class NPCState(Enum):
    """NPC state transitions."""
    INITIAL = "initial"
    ACTIVE = "active"
    DEATH_PENDING = "death_pending"
    DEATH = "death"
    RESURRECTED = "resurrected"
    TRANSFORMED = "transformed"


# ---------------------------------------------------------------------------
# Backwards-compatible *Type aliases (same enums, alternate names)
# ---------------------------------------------------------------------------

PersonalityType = Personality
MemoryType = MemoryLayer
BeliefType = BeliefLayer
EmotionType = EmotionLayer
ReflectionType = ReflectionLayer
GoalType = GoalLayer
RelationshipType = RelationshipLayer
CommunicationType = CommunicationLayer
CooperationType = CooperationLayer
ConflictType = ConflictLayer
InfluenceType = InfluenceLayer
HouseholdType = HouseholdLayer
FamilyType = FamilyLayer
FactionType = FactionLayer
AllianceType = AllianceLayer
LeadershipType = LeadershipLayer
InstitutionType = InstitutionLayer
CultureType = CultureLayer
TraditionType = TraditionLayer
EconomyType = EconomyLayer
LawType = LawLayer
SocialTransformationType = SocialTransformationLayer


# ===========================================================================
# Intelligence layers
# ===========================================================================

@dataclass
class MemoryEntry:
    """A single remembered event."""
    content: str
    kind: MemoryLayer = MemoryLayer.SHORT_TERM
    emotion: Optional[EmotionLayer] = None
    salience: float = 0.5  # 0.0 (trivial) .. 1.0 (unforgettable)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class IntelligenceBase(ABC):
    """Common interface for every layer of an NPC's intelligence."""
    layer: ClassVar[IntelligenceLayer]

    @abstractmethod
    def update(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Advance this layer one tick. Returns a dict describing what changed."""

    def snapshot(self) -> "IntelligenceBase":
        """Independent deep copy of this layer's current state."""
        return deepcopy(self)


@dataclass
class IndividualIntelligence(IntelligenceBase):
    """Personality, memory, beliefs, emotion, reflection and goals."""
    layer: ClassVar[IntelligenceLayer] = IntelligenceLayer.INDIVIDUAL

    personality: Personality = Personality.CAUTIOUS
    beliefs: Set[BeliefLayer] = field(default_factory=set)
    emotion: EmotionLayer = EmotionLayer.RELIEVED
    reflection: ReflectionLayer = ReflectionLayer.SHALLOW
    goals: List[GoalLayer] = field(default_factory=list)
    memories: List[MemoryEntry] = field(default_factory=list)
    short_term_limit: int = 10
    consolidation_threshold: float = 0.6

    def remember(self, content: str, emotion: Optional[EmotionLayer] = None,
                 salience: float = 0.5,
                 kind: MemoryLayer = MemoryLayer.SHORT_TERM) -> MemoryEntry:
        entry = MemoryEntry(content, kind, emotion, max(0.0, min(1.0, salience)))
        self.memories.append(entry)
        return entry

    def recall(self, kind: Optional[MemoryLayer] = None) -> List[MemoryEntry]:
        return [m for m in self.memories if kind is None or m.kind == kind]

    def _adopt_personality_goals(self) -> List[GoalLayer]:
        added = []
        for name in self.personality.get_effects({}).get("goals", []):
            goal = GoalLayer(name)
            if goal not in self.goals:
                self.goals.append(goal)
                added.append(goal)
        return added

    def _consolidate(self) -> int:
        """Promote salient short-term memories to long-term; forget the overflow."""
        for m in self.recall(MemoryLayer.SHORT_TERM):
            if m.salience >= self.consolidation_threshold:
                m.kind = MemoryLayer.LONG_TERM
        short = self.recall(MemoryLayer.SHORT_TERM)
        overflow = len(short) - self.short_term_limit
        if overflow > 0:
            forgotten = sorted(short, key=lambda m: (m.salience, m.timestamp))[:overflow]
            self.memories = [m for m in self.memories if m not in forgotten]
            return overflow
        return 0

    def update(self, context: Dict[str, Any]) -> Dict[str, Any]:
        changes: Dict[str, Any] = {}
        new_goals = self._adopt_personality_goals()
        if new_goals:
            changes["new_goals"] = new_goals
        for event in context.get("events", []):
            emotion = event.get("emotion")
            if emotion is not None:
                self.emotion = EmotionLayer(emotion) if isinstance(emotion, str) else emotion
                changes["emotion"] = self.emotion
            self.remember(event["content"], self.emotion, event.get("salience", 0.5))
        forgotten = self._consolidate()
        if forgotten:
            changes["forgotten"] = forgotten
        return changes


@dataclass
class SocialIntelligence(IntelligenceBase):
    """Relationships, communication, cooperation, conflict and influence."""
    layer: ClassVar[IntelligenceLayer] = IntelligenceLayer.SOCIAL

    relationships: Dict[str, RelationshipLayer] = field(default_factory=dict)
    communication: Set[CommunicationLayer] = field(
        default_factory=lambda: {CommunicationLayer.VERBAL})
    cooperation: CooperationLayer = CooperationLayer.AMBIGUOUS
    conflict: ConflictLayer = ConflictLayer.NEUTRAL
    influence: InfluenceLayer = InfluenceLayer.MODERATE

    def set_relationship(self, npc_id: str, kind: RelationshipLayer) -> None:
        self.relationships[npc_id] = kind

    def relationship_with(self, npc_id: str) -> RelationshipLayer:
        return self.relationships.get(npc_id, RelationshipLayer.STRANGER)

    def with_kind(self, kind: RelationshipLayer) -> List[str]:
        return [i for i, k in self.relationships.items() if k == kind]

    def hostility_ratio(self) -> float:
        if not self.relationships:
            return 0.0
        return len(self.with_kind(RelationshipLayer.ENEMY)) / len(self.relationships)

    def update(self, context: Dict[str, Any]) -> Dict[str, Any]:
        changes: Dict[str, Any] = {}
        for item in context.get("interactions", []):
            kind = item["kind"]
            kind = RelationshipLayer(kind) if isinstance(kind, str) else kind
            self.set_relationship(item["with"], kind)
            changes[item["with"]] = kind
        return changes


@dataclass
class CollectiveIntelligence(IntelligenceBase):
    """Household, family, factions, alliances, leadership and institutions."""
    layer: ClassVar[IntelligenceLayer] = IntelligenceLayer.COLLECTIVE

    household: HouseholdLayer = HouseholdLayer.SINGLE
    family: Optional[FamilyLayer] = None
    factions: Set[FactionLayer] = field(default_factory=set)
    alliances: Dict[str, AllianceLayer] = field(default_factory=dict)  # group name -> kind
    leadership: Optional[LeadershipLayer] = None
    institution: Optional[InstitutionLayer] = None

    def update(self, context: Dict[str, Any]) -> Dict[str, Any]:
        changes: Dict[str, Any] = {}
        for faction in context.get("join_factions", []):
            faction = FactionLayer(faction) if isinstance(faction, str) else faction
            if faction not in self.factions:
                self.factions.add(faction)
                changes.setdefault("joined", []).append(faction)
        for name, kind in context.get("alliances", {}).items():
            self.alliances[name] = AllianceLayer(kind) if isinstance(kind, str) else kind
            changes.setdefault("alliances", {})[name] = self.alliances[name]
        return changes


# ===========================================================================
# NPC
# ===========================================================================

class InvalidTransition(Exception):
    """Raised when an NPC is asked to make a state change that isn't allowed."""


ALLOWED_TRANSITIONS: Dict[NPCState, Set[NPCState]] = {
    NPCState.INITIAL: {NPCState.ACTIVE},
    NPCState.ACTIVE: {NPCState.DEATH_PENDING, NPCState.TRANSFORMED},
    NPCState.DEATH_PENDING: {NPCState.DEATH, NPCState.ACTIVE},  # ACTIVE = rescued
    NPCState.DEATH: {NPCState.RESURRECTED},
    NPCState.RESURRECTED: {NPCState.ACTIVE, NPCState.DEATH_PENDING},
    NPCState.TRANSFORMED: {NPCState.ACTIVE, NPCState.DEATH_PENDING},
}
TICKING_STATES = {NPCState.ACTIVE, NPCState.RESURRECTED, NPCState.TRANSFORMED}


@dataclass
class NPC:
    """A non-player character made of individual, social and collective layers."""
    name: str
    npc_id: str = field(default_factory=lambda: uuid4().hex[:8])
    state: NPCState = NPCState.INITIAL
    individual: IndividualIntelligence = field(default_factory=IndividualIntelligence)
    social: SocialIntelligence = field(default_factory=SocialIntelligence)
    collective: CollectiveIntelligence = field(default_factory=CollectiveIntelligence)
    history: List[Tuple[datetime, NPCState, NPCState, str]] = field(default_factory=list)

    @property
    def layers(self) -> List[IntelligenceBase]:
        return [self.individual, self.social, self.collective]

    @property
    def is_alive(self) -> bool:
        return self.state not in (NPCState.DEATH, NPCState.DEATH_PENDING)

    def transition(self, new_state: NPCState, reason: str = "") -> None:
        if new_state not in ALLOWED_TRANSITIONS[self.state]:
            raise InvalidTransition(f"{self.name}: {self.state.value} -> {new_state.value}")
        self.history.append((datetime.now(), self.state, new_state, reason))
        self.state = new_state

    def activate(self) -> None:
        self.transition(NPCState.ACTIVE, "activated")

    def mark_dying(self, reason: str = "") -> None:
        self.transition(NPCState.DEATH_PENDING, reason)

    def die(self, reason: str = "") -> None:
        """Finalise death, passing through DEATH_PENDING if needed."""
        if self.state != NPCState.DEATH_PENDING:
            self.mark_dying(reason)
        self.transition(NPCState.DEATH, reason)

    def rescue(self, reason: str = "") -> None:
        self.transition(NPCState.ACTIVE, reason or "rescued")

    def resurrect(self, reason: str = "") -> None:
        self.transition(NPCState.RESURRECTED, reason)
        self.individual.memories = [m for m in self.individual.memories
                                    if m.kind != MemoryLayer.SHORT_TERM]
        self.individual.remember("Returned from death", EmotionLayer.AWED, 1.0,
                                 MemoryLayer.EPISODIC)

    def transform(self, reason: str = "") -> None:
        self.transition(NPCState.TRANSFORMED, reason)

    def tick(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """Advance every layer one step. Does nothing unless the NPC is in a live state."""
        if self.state not in TICKING_STATES:
            return {}
        context = context or {}
        return {layer.layer.value: layer.update(context) for layer in self.layers}


# ===========================================================================
# Emergent layer (world level)
# ===========================================================================

@dataclass
class EmergentIntelligence(IntelligenceBase):
    """World-level patterns that emerge from the population as a whole.

    Culture, tradition and economy are set by the designer; law and the rate of
    social transformation are derived from the population in `observe`.
    The thresholds below are placeholder heuristics to tune for your game.
    """
    layer: ClassVar[IntelligenceLayer] = IntelligenceLayer.EMERGENT

    culture: Optional[CultureLayer] = None
    tradition: Optional[TraditionLayer] = None
    economy: Optional[EconomyLayer] = None
    law: Optional[LawLayer] = None
    transformation: SocialTransformationLayer = SocialTransformationLayer.STAGNANT
    population: List[NPC] = field(default_factory=list)

    INSTITUTION_TO_LAW: ClassVar[Dict[InstitutionLayer, LawLayer]] = {
        InstitutionLayer.MONARCHY: LawLayer.CUSTOM,
        InstitutionLayer.DEMOCRACY: LawLayer.CONSTITUTION,
        InstitutionLayer.THEOCRACY: LawLayer.THEOCRATIC,
        InstitutionLayer.FEUDALISM: LawLayer.CUSTOM,
        InstitutionLayer.CORPORATION: LawLayer.CORPORATIST,
        InstitutionLayer.GUILD: LawLayer.CORPORATIST,
        InstitutionLayer.STATE: LawLayer.CONSTITUTION,
    }

    def add(self, npc: NPC) -> None:
        self.population.append(npc)

    def living(self) -> List[NPC]:
        return [n for n in self.population if n.is_alive]

    def observe(self) -> Dict[str, Any]:
        """Derive law and transformation rate from the living population."""
        alive = self.living()
        if not alive:
            self.transformation = SocialTransformationLayer.STAGNANT
            return {"transformation": self.transformation}

        hostility = sum(n.social.hostility_ratio() for n in alive) / len(alive)
        if hostility > 0.5:
            self.transformation = SocialTransformationLayer.REVOLUTIONARY
        elif hostility > 0.3:
            self.transformation = SocialTransformationLayer.RAPID
        elif hostility > 0.1:
            self.transformation = SocialTransformationLayer.GRADUAL
        else:
            self.transformation = SocialTransformationLayer.STAGNANT

        institutions = Counter(n.collective.institution for n in alive
                               if n.collective.institution is not None)
        if institutions:
            dominant = institutions.most_common(1)[0][0]
            self.law = self.INSTITUTION_TO_LAW[dominant]

        return {"hostility": round(hostility, 3), "transformation": self.transformation,
                "law": self.law}

    def update(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return self.observe()
