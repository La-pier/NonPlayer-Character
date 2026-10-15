"""Regenerate tests/golden.json after an INTENTIONAL behaviour change.

    python tests/update_golden.py

Review the diff of golden.json before committing: the summary shows what shifted.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simulation import SimConfig, digest, run  # noqa: E402

CONFIG = {"seed": 7, "npcs": 12}
TICKS = 200

w = run(SimConfig(**CONFIG), TICKS)
golden = {
    "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    "config": CONFIG, "ticks": TICKS,
    "summary": w.log.summary(),
    "communities": [c.name for c in w.tracker.communities.values()],
    "digest": digest(w),
}
(ROOT / "tests" / "golden.json").write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")
print(json.dumps(golden["summary"], indent=2))
