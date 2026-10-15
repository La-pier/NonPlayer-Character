"""Measure the simulated behaviour's balance over many seeds.

    python tools/balance_report.py                      # 10 seeds, 12 NPCs, 200 ticks
    python tools/balance_report.py --seeds 100-109 --ticks 1000 --npcs 24

Prints action shares, how encounters went, how hostile the society is, and how
much of the starting personality mix survives. Use seeds you did NOT tune on to
check a change generalises.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classes import RelationshipLayer  # noqa: E402
from simulation import DEFAULT_MIX, SimConfig, run  # noqa: E402


def parse_seeds(text: str) -> range:
    lo, _, hi = text.partition("-")
    return range(int(lo), int(hi or lo) + 1)


def report(seeds: range, npcs: int, ticks: int) -> str:
    acts, kinds, final, enemies, changed, total = Counter(), Counter(), Counter(), [], 0, 0
    for s in seeds:
        w = run(SimConfig(seed=s, npcs=npcs), ticks)
        acts.update(r["action"] for r in w.log.of_type("action"))
        kinds.update(r["kind"] for r in w.log.of_type("interaction"))
        for i, ag in enumerate(w.agents.values()):
            p = ag.npc.individual.personality
            final[p.value] += 1
            changed += p != DEFAULT_MIX[i % len(DEFAULT_MIX)]
            total += 1
            enemies.append(len(ag.npc.social.with_kind(RelationshipLayer.ENEMY)))
    share = lambda c: ", ".join(f"{k} {v / sum(c.values()):.0%}"   # noqa: E731
                                for k, v in sorted(c.items()))
    return "\n".join([
        f"{len(seeds)} seeds x {npcs} NPCs x {ticks} ticks",
        f"actions:      {share(acts)}",
        f"encounters:   {share(kinds)}",
        f"enemies/NPC:  {sum(enemies) / len(enemies):.2f}",
        f"personality:  {len(final)} types, largest {max(final.values()) / total:.0%}, "
        f"{changed / total:.0%} changed from start",
    ])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", default="0-9", help="e.g. 0-9 or 100-109")
    ap.add_argument("--npcs", type=int, default=12)
    ap.add_argument("--ticks", type=int, default=200)
    a = ap.parse_args()
    print(report(parse_seeds(a.seeds), a.npcs, a.ticks))
