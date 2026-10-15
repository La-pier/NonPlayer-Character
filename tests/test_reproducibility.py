import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import classes
from classes import ALLOWED_TRANSITIONS, IndividualIntelligence, MemoryLayer
from simulation import SimConfig, digest, npc_id, run, stream

ROOT = Path(__file__).resolve().parent.parent
CFG = SimConfig(seed=7, npcs=10)


# --- same input, same output -------------------------------------------------------

def test_same_config_same_digest():
    assert digest(run(CFG, 80)) == digest(run(CFG, 80))


def test_different_seed_different_digest():
    assert digest(run(CFG, 80)) != digest(run(SimConfig(seed=8, npcs=10), 80))


def test_population_size_changes_the_digest():
    assert digest(run(CFG, 40)) != digest(run(SimConfig(seed=7, npcs=11), 40))


def test_history_is_a_prefix_of_a_longer_run():
    """Running longer must not rewrite the past."""
    short, long_ = run(CFG, 40), run(CFG, 90)
    assert long_.log.records[:len(short.log.records)] == short.log.records


def test_streams_are_independent_and_stable():
    a, b = stream(3, "x"), stream(3, "x")
    assert [a.random() for _ in range(5)] == [b.random() for _ in range(5)]
    assert stream(3, "x").random() != stream(3, "y").random()


@pytest.mark.parametrize("hashseed", ["0", "1", "424242"])
def test_digest_survives_python_hash_randomisation(hashseed):
    """str hashing (and so set order) differs per process unless pinned; the
    simulation must not depend on it."""
    expected = digest(run(CFG, 60))
    out = subprocess.run(
        [sys.executable, "simulation.py", "--seed", "7", "--npcs", "10",
         "--ticks", "60"],
        cwd=ROOT, env={**os.environ, "PYTHONHASHSEED": hashseed},
        capture_output=True, text=True, check=True).stdout
    assert out.strip().splitlines()[-1] == f"digest: {expected}"


# --- no wall-clock influence ---------------------------------------------------------

def test_digest_ignores_wall_clock_fields():
    world = run(CFG, 60)
    before = digest(world)
    for ag in world.agents.values():
        for i, m in enumerate(ag.npc.individual.memories):
            m.timestamp = datetime(1999, 1, 1) + timedelta(days=i)
        ag.npc.history[:] = [(datetime(2001, 1, 1), a, b, r)
                             for _, a, b, r in ag.npc.history]
    assert digest(world) == before


def test_forgetting_ties_break_by_tick_not_wall_clock():
    # Exactly one memory must go, and the two tied candidates have opposite
    # tick order and wall-clock order, so only the right rule picks 'first'.
    ind = IndividualIntelligence(short_term_limit=3)
    first = ind.remember("first", salience=0.2, tick=1)
    second = ind.remember("second", salience=0.2, tick=2)
    ind.remember("keep1", salience=0.5, tick=3)
    ind.remember("keep2", salience=0.5, tick=4)
    first.timestamp = datetime(2099, 1, 1)       # pretend 'first' is the newest
    second.timestamp = datetime(1970, 1, 1)
    assert ind._consolidate() == 1
    assert sorted(m.content for m in ind.memories) == ["keep1", "keep2", "second"]
    # equal salience AND tick: insertion order decides (oldest goes)
    ind = IndividualIntelligence(short_term_limit=1)
    ind.remember("a", salience=0.2, tick=1)
    ind.remember("b", salience=0.2, tick=1)
    ind._consolidate()
    assert [m.content for m in ind.memories] == ["b"]


def test_memories_are_stamped_with_the_simulation_tick():
    world = run(CFG, 30)
    for ag in world.agents.values():
        ticks = [m.tick for m in ag.npc.individual.memories]
        assert ticks == sorted(ticks) and 1 <= min(ticks) and max(ticks) <= world.tick


# --- digest sensitivity (so a green result means something) ---------------------------

def test_digest_notices_small_state_changes():
    base = digest(run(CFG, 40))
    w = run(CFG, 40)
    ag = w.agents[npc_id(0)]
    ag.npc.individual.remember("tampered", tick=w.tick)
    assert digest(w) != base
    w = run(CFG, 40)
    w.agents[npc_id(0)].needs.hunger += 0.001
    assert digest(w) != base
    w = run(CFG, 40)
    w.log.add("action", w.tick, npc="x")
    assert digest(w) != base


# --- invariants across many seeds -------------------------------------------------------

SEEDS = range(12)


@pytest.mark.parametrize("seed", SEEDS)
def test_invariants_hold(seed):
    w = run(SimConfig(seed=seed, npcs=10), 120)
    ids = set(w.agents)
    for i, ag in w.agents.items():
        soc, ind = ag.npc.social, ag.npc.individual
        assert all(-1.0 <= t <= 1.0 for t in soc.trust.values())
        assert i not in soc.trust and i not in soc.relationships
        assert set(soc.trust) <= ids and set(soc.relationships) <= ids
        assert all(0.0 <= v <= 1.0 for v in
                   (ag.needs.hunger, ag.needs.fatigue, ag.needs.loneliness))
        assert all(0.0 <= v <= 1.0 for v in ag.traits.strengths.values())
        assert all(0.0 <= m.salience <= 1.0 for m in ind.memories)
        for _, src, dst, _ in ag.npc.history:
            assert dst in ALLOWED_TRANSITIONS[src]
        assert len(ind.recall(MemoryLayer.SHORT_TERM)) <= ind.short_term_limit
        assert len(set(ind.goals)) == len(ind.goals)

    for c in w.tracker.communities.values():
        assert len(c.members) >= w.tracker.min_size and c.members <= ids
        assert c.leader in c.members
        for m in c.members:
            assert c.name in w.agents[m].npc.collective.alliances

    seen = set()
    for r in w.log.of_type("interaction"):
        assert r["a"] != r["b"] and {r["a"], r["b"]} <= ids
        key = (r["tick"], frozenset((r["a"], r["b"])))
        assert key not in seen          # one encounter per pair per tick
        seen.add(key)


@pytest.mark.parametrize("seed", SEEDS)
def test_two_runs_of_every_seed_agree(seed):
    cfg = SimConfig(seed=seed, npcs=8)
    assert digest(run(cfg, 50)) == digest(run(cfg, 50))


# --- golden run ----------------------------------------------------------------------------

GOLDEN = ROOT / "tests" / "golden.json"


def test_golden_run_has_not_drifted():
    """A pinned run. If behaviour changes ON PURPOSE, regenerate with
    `python tests/update_golden.py` and review the summary diff in the commit;
    if it changes by accident, this test is the alarm."""
    golden = json.loads(GOLDEN.read_text())
    if golden["python"] != f"{sys.version_info.major}.{sys.version_info.minor}":
        pytest.skip(f"golden recorded on Python {golden['python']}; the random "
                    f"module's algorithms may differ on this version")
    w = run(SimConfig(**golden["config"]), golden["ticks"])
    assert w.log.summary() == golden["summary"], "behaviour changed (see summary)"
    assert [c.name for c in w.tracker.communities.values()] == golden["communities"]
    assert digest(w) == golden["digest"], "state changed; summary matched"
