# NonPlayer-Character
simple PyTorch neural network that takes NPC sensor inputs (e.g. distance to player, health), chooses an action, and shows how to export it to an ONNX file with torch.onnx.export

## NPC intelligence simulation

Layers (`classes.py`), autonomy (`autonomy.py`), learning (`adaptation.py`),
social life and communities (`social.py`), and a reproducible runner
(`simulation.py`) with a full history log (`history.py`).

```
pip install pytest
python -m pytest -q tests                      # 170+ tests

python simulation.py --seed 7 --npcs 12 --ticks 200 --log run.jsonl
python simulation.py --seed 7 --timeline npc-003   # the story of one NPC
```

**Reproducibility.** One seed fixes everything: same seed and config give the
same history and the same final-state `digest`, across runs and Python hash
randomisation. Simulation time is the `tick`; wall-clock time never affects
behaviour. Build worlds with `simulation.build_world` (it fixes NPC ids and
random streams); if you build by hand, pass explicit `npc_id`s and seeded rngs.

**History.** `world.log` records every action (with needs, emotion, personality
and any reflection), every interaction (with resulting trust), community
formation/dissolution, deaths, and world-level observations. Replaying the log's
trust changes reproduces each NPC's final trust exactly (tested).

**Golden run.** `tests/golden.json` pins one run. If you change behaviour on
purpose, run `python tests/update_golden.py` and review the diff in the commit.
It is recorded per Python version and skipped on a different one.

## Behaviour balance

The history log showed the first version was lopsided: goal pursuit was 35% of
all actions, a third of encounters were arguments, NPCs ended with ~3 enemies
each, and the six starting personalities collapsed to three. Causes and fixes
(constant, file):

| Cause | Fix |
| --- | --- |
| Strangers had a coin-flip chance of arguing, and every argument made the next worse | `GOODWILL_BASE = 0.6` (`social.py`) |
| Grudges never faded, so enemies only accumulated | `GRUDGE_DECAY = 0.002` per tick, applied world-wide before anyone acts (`social.py`) |
| Traits changed within tens of ticks and reinforced themselves, converging everyone | `LEARNING_RATE = 0.1`, and reinforcement has diminishing returns (`adaptation.py`) |
| Goal pursuit had a constant score, so it was the default idle action | `purpose` is a drive that builds (`purpose_rate = 0.08`) and is relieved by acting (`PURPOSE_RELIEF`) (`autonomy.py`) |
| Happy memories fed on themselves, inflating socialising | `SENTIMENT_WEIGHT = 0.2` (`autonomy.py`) |

Targets: no action above ~30% or below 5%; arguments 12-22% of encounters;
about one enemy per NPC; most personality types survive. Measure with
`python tools/balance_report.py` (use seeds you did not tune on), and
`tests/test_balance.py` guards the shape of the behaviour.

Known limits: over very long runs (1000+ ticks) the society keeps warming up
(socialising drifts to ~33%, arguments to ~10%) because positive trust never
fades, and personality is nearly frozen at this learning rate. Learning is
still driven by what an NPC does rather than by outcomes, which is why fast
learning homogenises; outcome-driven learning is the proper next step.
