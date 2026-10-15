"""Reproducible simulation runner.

One master seed determines everything: NPC ids, personalities, every random
stream, and so the whole history. `digest()` fingerprints the final state;
`world.log` records what happened. Same config => same digest, always.

    python simulation.py --seed 7 --npcs 12 --ticks 200 --log run.jsonl
    python simulation.py --seed 7 --timeline npc-003
"""
import argparse
import hashlib
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from autonomy import Agent, Needs
from classes import NPC, Personality
from history import dumps
from social import World

DEFAULT_MIX: Tuple[Personality, ...] = (
    Personality.EXTROVERTED, Personality.OPTIMISTIC, Personality.SHY,
    Personality.BOLD, Personality.AMBITIOUS, Personality.PESSIMISTIC,
)


@dataclass(frozen=True)
class SimConfig:
    seed: int = 0
    npcs: int = 12
    personalities: Tuple[Personality, ...] = DEFAULT_MIX
    meet_base: float = 0.15
    meet_friend: float = 0.3
    food_rate: float = 0.4
    threat_rate: float = 0.05


def stream(seed: int, name: str) -> random.Random:
    """An independent, reproducible random stream. Seeding from a string is
    stable across runs and platforms (unlike hash(), which is randomised)."""
    return random.Random(f"{seed}:{name}")


def npc_id(i: int) -> str:
    return f"npc-{i:03d}"


def build_world(cfg: SimConfig) -> World:
    world = World(stream(cfg.seed, "world"), meet_base=cfg.meet_base,
                  meet_friend=cfg.meet_friend, food_rate=cfg.food_rate,
                  threat_rate=cfg.threat_rate)
    for i in range(cfg.npcs):
        npc = NPC(f"NPC{i}", npc_id=npc_id(i))
        npc.individual.personality = cfg.personalities[i % len(cfg.personalities)]
        world.add(Agent(npc, Needs(), stream(cfg.seed, npc_id(i))))
    return world


def run(cfg: SimConfig, ticks: int) -> World:
    world = build_world(cfg)
    for _ in range(ticks):
        world.step()
    return world


# -- canonical state ---------------------------------------------------------------

def snapshot(world: World) -> Dict[str, Any]:
    """Everything that defines the world's state, minus wall-clock fields."""
    npcs = {}
    for i, ag in sorted(world.agents.items()):
        n, ind = ag.npc, ag.npc.individual
        npcs[i] = {
            "state": n.state, "personality": ind.personality,
            "traits": ag.traits.strengths, "needs": ag.needs,
            "emotion": ind.emotion, "reflection": ind.reflection,
            "beliefs": ind.beliefs, "goals": ind.goals,
            "memories": [{"c": m.content, "k": m.kind, "e": m.emotion,
                          "s": m.salience, "t": m.tick, "tags": m.tags}
                         for m in ind.memories],
            "trust": n.social.trust, "relationships": n.social.relationships,
            "alliances": n.collective.alliances, "factions": n.collective.factions,
            "leadership": n.collective.leadership,
            "institution": n.collective.institution,
            "transitions": [(a, b, r) for _, a, b, r in n.history],
            "ticks": ag.ticks,
        }
    comms = {k: {"members": c.members, "leader": c.leader,
                 "leadership": c.leadership, "institution": c.institution,
                 "cohesion": c.cohesion}
             for k, c in sorted(world.tracker.communities.items())}
    return {"tick": world.tick, "npcs": npcs, "communities": comms,
            "law": world.emergent.law, "transformation": world.emergent.transformation}


def _plain(x: Any) -> Any:
    """Turn dataclasses (Needs) into dicts so they serialise."""
    if hasattr(x, "__dataclass_fields__"):
        return {k: _plain(getattr(x, k)) for k in x.__dataclass_fields__}
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def digest(world: World) -> str:
    """Fingerprint of final state AND full history."""
    h = hashlib.sha256()
    h.update(dumps(_plain(snapshot(world))).encode())
    h.update(world.log.digest().encode())
    return h.hexdigest()


# -- command line ----------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--npcs", type=int, default=12)
    ap.add_argument("--ticks", type=int, default=200)
    ap.add_argument("--log", help="write the full history here as JSON lines")
    ap.add_argument("--timeline", metavar="NPC_ID",
                    help="print the story of one NPC (e.g. npc-003)")
    args = ap.parse_args(argv)

    world = run(SimConfig(seed=args.seed, npcs=args.npcs), args.ticks)
    names = {i: ag.npc.name for i, ag in world.agents.items()}
    if args.log:
        world.log.write(args.log)
    if args.timeline:
        if args.timeline not in world.agents:
            ap.error(f"unknown NPC id {args.timeline!r}; ids are npc-000..npc-"
                     f"{args.npcs - 1:03d}")
        print(world.log.timeline(args.timeline, names))
    else:
        print("summary:", world.log.summary())
        for c in world.tracker.communities.values():
            print(f"  {c.name}: {len(c.members)} members, leader {c.leader}, "
                  f"{c.leadership.value}/{c.institution.value}")
        print("law:", getattr(world.emergent.law, "value", None),
              "| transformation:", world.emergent.transformation.value)
    print("digest:", digest(world))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
