"""Run a few NPCs for a while and print what each chooses."""
import random

from autonomy import Agent, Needs, Perception
from classes import NPC, Personality

rng = random.Random(7)
agents = []
for name, personality in [("Ana", Personality.EXTROVERTED), ("Bo", Personality.BOLD),
                          ("Cy", Personality.SHY), ("Di", Personality.AMBITIOUS)]:
    npc = NPC(name)
    npc.activate()
    npc.individual.personality = personality
    agents.append(Agent(npc, Needs(), random.Random(rng.random())))

for tick in range(200):
    for a in agents:
        others = [o.npc.npc_id for o in agents if o is not a and rng.random() < 0.5]
        a.step(Perception(threat=rng.random() * 0.5 if rng.random() < 0.2 else 0.0,
                          nearby=others, food_available=rng.random() < 0.4))

for a in agents:
    print(f"{a.npc.name:4} {a.npc.individual.personality.value:12} "
          f"hunger={a.needs.hunger:.2f} emotion={a.npc.individual.emotion.value:9} "
          f"goals={[g.value for g in a.npc.individual.goals]}")
