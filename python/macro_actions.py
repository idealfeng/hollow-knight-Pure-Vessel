"""A versioned, single-choice combat action vocabulary.

Each choice describes one ordinary 100 ms decision, never a timed combo.
The table retains directional and jumping nail attacks while excluding
simultaneous attack/dash/cast requests that compete for the game's animations.
"""

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral


MACRO_DISTRIBUTION = "discrete_macros_v1"
MACRO_SCHEMA = "combat_macros_v1"
MACRO_EXECUTION_PROFILE = "legacy_v1"


@dataclass(frozen=True)
class MacroAction:
    name: str
    buttons: tuple[int, int, int, int, int, int, int, int]


def _buttons(horizontal="facing", aim="horizontal", *, attack=0, jump=0,
             dash=0, spell=0):
    return (int(horizontal == "left"), int(horizontal == "right"),
            int(aim == "up"), int(aim == "down"), attack, jump, dash, spell)


def _build_actions():
    actions = [MacroAction("idle", _buttons())]
    actions.extend(MacroAction(f"move_{h}", _buttons(h)) for h in ("left", "right"))
    actions.extend(MacroAction(f"jump_{h}", _buttons(h, jump=1))
                   for h in ("facing", "left", "right"))
    for jumping in (False, True):
        for h in ("facing", "left", "right"):
            for aim in ("horizontal", "up", "down"):
                name = f"{'jump_' if jumping else ''}nail_{h}_{aim}"
                actions.append(MacroAction(name, _buttons(h, aim, attack=1, jump=int(jumping))))
    for jumping in (False, True):
        actions.extend(MacroAction(f"{'jump_' if jumping else ''}dash_{h}",
                                   _buttons(h, jump=int(jumping), dash=1))
                       for h in ("facing", "left", "right"))
    actions.extend(MacroAction(f"shade_soul_{h}", _buttons(h, spell=1))
                   for h in ("facing", "left", "right"))
    # Explicit effective aim agrees with quick-cast dispatch and makes the
    # vectors canonical when evaluating the old factorized policy.
    actions.append(MacroAction("abyss_shriek", _buttons(aim="up", spell=2)))
    actions.append(MacroAction("descending_dark", _buttons(aim="down", spell=3)))
    return tuple(actions)


MACRO_ACTIONS = _build_actions()
MACRO_ACTION_NAMES = tuple(action.name for action in MACRO_ACTIONS)
NUM_MACRO_ACTIONS = len(MACRO_ACTIONS)


def _table_data():
    return [{"name": action.name, "buttons": list(action.buttons)}
            for action in MACRO_ACTIONS]


MACRO_ACTION_HASH = hashlib.sha256(json.dumps(
    {"schema": MACRO_SCHEMA, "actions": _table_data()},
    sort_keys=True, separators=(",", ":"), ensure_ascii=True,
).encode("ascii")).hexdigest()
MACRO_SCHEMA_HASH = MACRO_ACTION_HASH


def decode_macro(index):
    """Return the immutable eight-button vector for a scalar macro choice."""
    if isinstance(index, bool) or not isinstance(index, Integral):
        raise ValueError("macro action must be an integer index")
    if not 0 <= int(index) < NUM_MACRO_ACTIONS:
        raise ValueError(f"macro action must be in [0, {NUM_MACRO_ACTIONS})")
    return MACRO_ACTIONS[int(index)].buttons


def macro_metadata():
    """Fresh JSON-compatible metadata; callers cannot mutate the table."""
    return {"macro_schema": MACRO_SCHEMA, "macro_schema_hash": MACRO_ACTION_HASH,
            "macro_actions": _table_data()}


def validate_macro_checkpoint(saved):
    """Reject a checkpoint whose categorical IDs do not match this table."""
    if not isinstance(saved, dict):
        raise ValueError("macro checkpoint must be a dictionary")
    if (saved.get("format_version") != 10
            or saved.get("action_distribution") != MACRO_DISTRIBUTION
            or saved.get("action_names") != ["macro"]
            or saved.get("action_sizes") != [NUM_MACRO_ACTIONS]):
        raise ValueError("macro checkpoint must declare format 10 and one macro categorical head")
    for key, expected in macro_metadata().items():
        if saved.get(key) != expected:
            raise ValueError(f"macro checkpoint {key} does not match the supported action table")
    return True
