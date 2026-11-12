"""Audited, effective-dated paper-account inputs supplied by an operator."""

import copy
import hashlib
import json
import math
from typing import Dict, Iterable, List


ACTION_TYPES = frozenset({"borrow_terms", "borrow_charge", "financing", "dividend", "split", "settlement", "valuation_mark"})


class AccountActionSchedule:
    """Durable action definitions and status; the engine executes each action."""

    def __init__(self):
        self.actions: Dict[str, Dict] = {}

    def schedule(self, action: Dict) -> Dict:
        if not isinstance(action, dict):
            raise ValueError("account action must be an object")
        required = {"action_id", "action_type", "instrument_id", "source", "effective_ts", "payload"}
        if not required.issubset(action) or set(action) - required - {"corrects"}:
            raise ValueError("account action requires id, type, instrument, source, effective time and payload")
        item = copy.deepcopy(action)
        for key in ("action_id", "instrument_id", "source"):
            if not isinstance(item[key], str) or not item[key].strip():
                raise ValueError("action {} must be non-empty".format(key))
            item[key] = item[key].strip()
        if item["action_type"] not in ACTION_TYPES or not isinstance(item["payload"], dict):
            raise ValueError("unsupported action type or malformed payload")
        stamp = item["effective_ts"]
        if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp) or stamp < 0:
            raise ValueError("action effective time must be finite epoch seconds")
        item["effective_ts"] = float(stamp)
        if "corrects" in item:
            previous = self.actions.get(item["corrects"])
            if previous is None or previous.get("status") not in {"applied", "failed"}:
                raise ValueError("a correction must reference an applied or failed action")
            if previous["instrument_id"] != item["instrument_id"]:
                raise ValueError("a correction must reference the same instrument")
            if previous["status"] == "applied" and item["action_type"] not in {"financing", "borrow_charge"}:
                raise ValueError("only explicit sourced financing deltas can correct an applied action")
            if previous["status"] == "failed" and previous["action_type"] != item["action_type"]:
                raise ValueError("a failed input replacement must have the same action type")
        signature = hashlib.sha256(json.dumps(item, sort_keys=True, allow_nan=False).encode()).hexdigest()
        previous = self.actions.get(item["action_id"])
        if previous:
            if previous["signature"] != signature:
                raise ValueError("action ID was reused with different economic inputs")
            return copy.deepcopy(previous)
        item.update(status="pending", signature=signature, processed_ts=None, error=None)
        self.actions[item["action_id"]] = item
        return copy.deepcopy(item)

    def due(self, now: float) -> List[Dict]:
        return sorted((item for item in self.actions.values()
                       if item["status"] == "pending" and item["effective_ts"] <= now),
                      key=lambda item: (item["effective_ts"], item["action_id"]))

    def export_state(self) -> Dict:
        return copy.deepcopy(self.actions)

    def restore_state(self, actions: Dict) -> None:
        if not isinstance(actions, dict):
            raise ValueError("account schedule checkpoint must be an object")
        self.actions = {}
        for item in actions.values():
            self.schedule({key: value for key, value in item.items()
                           if key in {"action_id", "action_type", "instrument_id", "source", "effective_ts", "payload"}})
        for key, item in actions.items():
            if item.get("status") not in {"pending", "applied", "failed", "superseded"}:
                raise ValueError("invalid account action checkpoint status")
            semantic = {name: value for name, value in item.items()
                        if name in {"action_id", "action_type", "instrument_id", "source", "effective_ts", "payload", "corrects"}}
            signature = hashlib.sha256(json.dumps(semantic, sort_keys=True, allow_nan=False).encode()).hexdigest()
            if signature != item.get("signature") or key != item["action_id"]:
                raise ValueError("account action checkpoint checksum mismatch")
            self.actions[key] = copy.deepcopy(item)
        for item in self.actions.values():
            if item.get("corrects") and (item["corrects"] not in self.actions
                    or self.actions[item["corrects"]]["status"] not in {"applied", "failed", "superseded"}):
                raise ValueError("correction references an unavailable applied action")
