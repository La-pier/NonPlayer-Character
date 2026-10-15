import json

import pytest

from classes import Personality
from history import EventLog, dumps, jsonable
from simulation import SimConfig, run


# --- EventLog unit tests ---------------------------------------------------------

def test_jsonable_is_stable_and_plain():
    out = jsonable({"b": {3, 1, 2}, "a": Personality.BOLD, "c": (1, 2.123456789)})
    assert out == {"b": [1, 2, 3], "a": "bold", "c": [1, 2.123457]}
    json.dumps(out)


def test_dumps_ignores_key_and_set_order():
    assert dumps({"x": {"b", "a"}, "y": 1}) == dumps({"y": 1, "x": {"a", "b"}})


def test_add_and_query():
    log = EventLog()
    log.add("action", 1, npc="a", action="rest")
    log.add("interaction", 2, kind="argue", a="a", b="b")
    log.add("community", 3, event="formed", name="C1", members=["b", "c"])
    assert len(log) == 3
    assert [r["tick"] for r in log.for_npc("a")] == [1, 2]
    assert [r["tick"] for r in log.for_npc("c")] == [3]
    assert log.of_type("community")[0]["name"] == "C1"
    assert [r["tick"] for r in log.between(2, 3)] == [2, 3]


def test_jsonl_roundtrip(tmp_path):
    log = EventLog()
    log.add("action", 1, npc="a", needs={"hunger": 0.1}, action=Personality.SHY)
    path = tmp_path / "run.jsonl"
    log.write(str(path))
    again = EventLog.read(str(path))
    assert again.records == log.records and again.digest() == log.digest()


def test_digest_changes_with_content():
    a, b = EventLog(), EventLog()
    a.add("action", 1, npc="x")
    b.add("action", 1, npc="y")
    assert a.digest() != b.digest()


def test_timeline_and_summary_text():
    log = EventLog()
    log.add("action", 1, npc="a", action="socialize", partner="b", what="chat",
            emotion="happy", personality_before="shy", personality_after="bold",
            reflection={"adopted_belief": "optimistic"})
    log.add("interaction", 1, kind="cooperate", a="a", b="b", trust_ab=0.15, trust_ba=0.15)
    text = log.timeline("a", {"a": "Ann", "b": "Bo"})
    assert "socialize with Bo" in text and "shy -> bold" in text
    assert "adopted_belief=optimistic" in text and "trust in them now +0.15" in text
    assert log.summary() == {"records": 2, "actions": {"socialize": 1},
                             "interactions": {"cooperate": 1}}


# --- the log of a real run ------------------------------------------------------------

@pytest.fixture(scope="module")
def world():
    return run(SimConfig(seed=5, npcs=10), 150)


def test_every_live_npc_has_one_action_per_tick(world):
    for i in world.agents:
        ticks = [r["tick"] for r in world.log.of_type("action") if r["npc"] == i]
        assert ticks == list(range(1, world.tick + 1))


def test_log_is_in_time_order(world):
    ticks = [r["tick"] for r in world.log.records]
    assert ticks == sorted(ticks) and ticks[-1] == world.tick


def test_log_trust_replay_reproduces_final_trust(world):
    """The history must fully explain the state: replaying every logged trust
    change, plus the documented grudge-decay law, lands on each NPC's actual
    trust values."""
    decay = world.log.of_type("config")[0]["grudge_decay"]
    replay = {i: {} for i in world.agents}          # owner -> other -> (trust, tick)
    for r in world.log.of_type("interaction"):
        replay[r["a"]][r["b"]] = (r["trust_ab"], r["tick"])
        replay[r["b"]][r["a"]] = (r["trust_ba"], r["tick"])
        for third, t in (r.get("listener_trust") or {}).items():
            replay[r["b"]][third] = (t, r["tick"])
    for i, ag in world.agents.items():
        actual = ag.npc.social.trust
        assert set(replay[i]) == set(actual), i
        for other, (t, tick) in replay[i].items():
            if t < 0:           # grudges fade once per tick after the logged moment
                t *= (1 - decay) ** (world.tick - tick)
            assert t == pytest.approx(actual[other], abs=1e-5), (i, other)


def test_log_personality_replay_reproduces_final_personality(world):
    for i, ag in world.agents.items():
        mine = [r for r in world.log.of_type("action") if r["npc"] == i]
        assert mine[-1]["personality_after"] == ag.npc.individual.personality.value
        for prev, nxt in zip(mine, mine[1:]):     # changes are continuous
            assert prev["personality_after"] == nxt["personality_before"]


def test_log_communities_match_tracker(world):
    live = {}
    for r in world.log.of_type("community"):
        if r["event"] == "formed":
            live[r["name"]] = r
        else:
            live.pop(r["name"])
    assert set(live) == set(world.tracker.communities)


def test_log_records_deaths():
    w = run(SimConfig(seed=1, npcs=6), 20)
    w.kill("npc-002", "test")
    w.step()
    rec = w.log.of_type("lifecycle")[0]
    assert (rec["npc"], rec["event"], rec["tick"]) == ("npc-002", "died", 20)
    assert not any(r["npc"] == "npc-002" for r in w.log.of_type("action")
                   if r["tick"] > 20)
