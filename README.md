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
