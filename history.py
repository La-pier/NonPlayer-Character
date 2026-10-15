"""Append-only history of a simulation: what each NPC did, and what emerged.

Records are plain dicts (JSON-serialisable, with deterministic key order) so a
run can be saved, diffed, replayed into questions, and compared across seeds.
Nothing here depends on wall-clock time.
"""
import hashlib
import json
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional


def jsonable(x: Any) -> Any:
    """Recursively convert enums, sets, tuples and floats to stable JSON types."""
    if isinstance(x, Enum):
        return x.value
    if isinstance(x, dict):
        return {str(jsonable(k)): jsonable(v) for k, v in x.items()}
    if isinstance(x, (set, frozenset)):
        return sorted((jsonable(v) for v in x), key=lambda v: json.dumps(v))
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, float):
        return round(x, 6)
    return x


def dumps(obj: Any) -> str:
    return json.dumps(jsonable(obj), sort_keys=True, separators=(",", ":"))


class EventLog:
    """Ordered list of records; each has a `tick` and a `type`."""

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def add(self, type: str, tick: int, **fields: Any) -> Dict[str, Any]:
        rec = jsonable({"tick": tick, "type": type, **fields})
        self.records.append(rec)
        return rec

    def __len__(self) -> int:
        return len(self.records)

    # -- queries
    def of_type(self, type: str) -> List[Dict[str, Any]]:
        return [r for r in self.records if r["type"] == type]

    def for_npc(self, npc_id: str) -> List[Dict[str, Any]]:
        """Everything that involved this NPC, in order."""
        return [r for r in self.records
                if r.get("npc") == npc_id or npc_id in (r.get("a"), r.get("b"))
                or npc_id in r.get("members", ())]

    def between(self, first: int, last: int) -> List[Dict[str, Any]]:
        return [r for r in self.records if first <= r["tick"] <= last]

    # -- output
    def digest(self) -> str:
        return hashlib.sha256(dumps(self.records).encode()).hexdigest()

    def to_jsonl(self) -> str:
        return "\n".join(dumps(r) for r in self.records) + ("\n" if self.records else "")

    def write(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_jsonl())

    @classmethod
    def read(cls, path: str) -> "EventLog":
        log = cls()
        with open(path, encoding="utf-8") as f:
            log.records = [json.loads(line) for line in f if line.strip()]
        return log

    def timeline(self, npc_id: str, names: Optional[Dict[str, str]] = None) -> str:
        """Human-readable story of one NPC."""
        names = names or {}
        who = lambda i: names.get(i, i)          # noqa: E731
        lines = []
        for r in self.for_npc(npc_id):
            t, kind = f"t={r['tick']:>4}", r["type"]
            if kind == "action":
                s = f"{r['action']}"
                if r.get("partner"):
                    s += f" with {who(r['partner'])}"
                s += f" ({r['what']}); feels {r['emotion']}"
                if r["personality_before"] != r["personality_after"]:
                    s += f"; personality {r['personality_before']} -> {r['personality_after']}"
                for key, val in (r.get("reflection") or {}).items():
                    s += f"; reflected: {key}={val}"
                lines.append(f"{t}  {s}")
            elif kind == "interaction":
                other = r["b"] if r["a"] == npc_id else r["a"]
                mine = r["trust_ab"] if r["a"] == npc_id else r["trust_ba"]
                lines.append(f"{t}  {r['kind']} with {who(other)} "
                             f"(trust in them now {mine:+.2f})")
            elif kind == "community":
                lines.append(f"{t}  {r['event']} community {r['name']}")
            elif kind == "lifecycle":
                lines.append(f"{t}  {r['event']}: {r.get('reason', '')}".rstrip(": "))
        return "\n".join(lines)

    def summary(self) -> Dict[str, Any]:
        actions: Dict[str, int] = {}
        for r in self.of_type("action"):
            actions[r["action"]] = actions.get(r["action"], 0) + 1
        kinds: Dict[str, int] = {}
        for r in self.of_type("interaction"):
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        return {"records": len(self.records), "actions": dict(sorted(actions.items())),
                "interactions": dict(sorted(kinds.items()))}
