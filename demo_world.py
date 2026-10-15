"""Run a small society and report who befriended whom and what emerged."""
import random
from collections import Counter

from autonomy import Agent, Needs
from classes import NPC, Personality
from social import World

SEED = 3
world = World(random.Random(SEED))
mix = [Personality.EXTROVERTED, Personality.OPTIMISTIC, Personality.SHY,
       Personality.BOLD, Personality.AMBITIOUS, Personality.PESSIMISTIC]
for i in range(18):
    npc = NPC(f"n{i}", npc_id=f"id{i:02d}")
    npc.individual.personality = mix[i % len(mix)]
    world.add(Agent(npc, Needs(), random.Random(SEED * 100 + i)))

emergent = {}
for _ in range(300):
    emergent = world.step().get("emergent", emergent)

print("interactions:", dict(Counter(i.kind.value for i in world.interactions)))
for c in world.tracker.communities.values():
    print(f"{c.name}: {len(c.members)} members, leader={c.leader}, "
          f"{c.leadership.value}/{c.institution.value}, cohesion={c.cohesion:.2f}")
print("rival communities:", world.tracker.rivals() or "none")
print("world:", {k: getattr(v, "value", v) for k, v in emergent.items()})
