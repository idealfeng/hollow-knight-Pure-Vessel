"""Tests for the action-space rework and FSM phase mapping (no game, no training)."""
from unittest import mock

import game_state
from agent.action import (
    PureVesselMove, PureVesselPhase, boss_move, boss_phase,
    movement_actions, look_actions, attack_actions, dash_actions, jump_actions,
    movement, look,
)
from agent.reward import calculate_reward


def make_state(boss_control_state, boss_health=1000, knight_health=9, **overrides):
    s = {
        "knight_box": (40.0, 4.0, 42.0, 6.0),
        "boss_control_state": boss_control_state,
        "boss_health": boss_health,
        "knight_health": knight_health,
        "hit_streak": 0,
        "hazards": [],
        "boss_facing_left": False,
        "boss_position": (52.0, 9.0),
        "boss_velocity": (0.0, 0.0),
        "boss_box": (51.0, 8.0, 53.0, 10.0),
        "knight_facing_right": True,
        "knight_velocity": (0.0, 0.0),
        "attacking": False,
        "attack_box": None,
        "is_dashing": False,
        "is_jumping": False,
        "is_falling": False,
        "is_on_ground": True,
        "is_recoiling": False,
        "is_touching_wall_left": False,
        "is_touching_wall_right": False,
        "wall_distance_left": 0.0,
        "wall_distance_right": 0.0,
        "can_shade_cloak_now": True,
        "is_invuln": False,
        "hit_count": 0,
        "damage_count": 0,
        "mana": 0,
    }
    s.update(overrides)
    return s


results = []


def check(name, cond):
    print("  %s  %s" % ("PASS" if cond else "FAIL", name))
    results.append(cond)


print("== 1. action heads (do_nothing removed from move/look) ==")
check("movement has 5 actions (no do_nothing)", len(movement_actions) == 5)
check("look has 3 actions (no do_nothing)", len(look_actions) == 3)
check("attack keeps do_nothing (2)", len(attack_actions) == 2)
check("dash keeps do_nothing (2)", len(dash_actions) == 2)
check("jump keeps do_nothing (3)", len(jump_actions) == 3)
check("movement idx 2 is stop", movement_actions[2] is movement.stop)
check("look idx 2 is stop_looking", look_actions[2] is look.stop_looking)

print("== 2. press/release semantics ==")
with mock.patch("agent.action.controls.key_up") as ku, \
     mock.patch("agent.action.controls.key_down") as kd:
    movement.stop()
    check("stop releases left+right",
          mock.call("left") in ku.call_args_list and mock.call("right") in ku.call_args_list)
    movement.move_left()
    check("move_left holds left + releases right",
          mock.call("left") in kd.call_args_list and mock.call("right") in ku.call_args_list)
    look.stop_looking()
    check("stop_looking releases up+down",
          mock.call("up") in ku.call_args_list and mock.call("down") in ku.call_args_list)
    look.look_up()
    check("look_up holds up + releases down",
          mock.call("up") in kd.call_args_list and mock.call("down") in ku.call_args_list)

print("== 3. FSM -> move + phase ==")
cases = [
    ("Slash1 Antic", PureVesselMove.TRIPLE_SLASH, PureVesselPhase.WINDUP),
    ("Slash1", PureVesselMove.TRIPLE_SLASH, PureVesselPhase.ACTIVE),
    ("Slash3 Recover", PureVesselMove.TRIPLE_SLASH, PureVesselPhase.RECOVERY),
    ("Dash Antic", PureVesselMove.DASH, PureVesselPhase.WINDUP),
    ("Dash", PureVesselMove.DASH, PureVesselPhase.ACTIVE),
    ("SmallShot LowHigh", PureVesselMove.DAGGER, PureVesselPhase.ACTIVE),
    ("Stun", PureVesselMove.STUN, PureVesselPhase.STUN),
    ("StunToIdle", PureVesselMove.STUN, PureVesselPhase.RECOVERY),
    ("Tendril Antic", PureVesselMove.TENDRIL, PureVesselPhase.UNKNOWN),
    ("Idle", PureVesselMove.NONE, PureVesselPhase.NEUTRAL),
    ("TotallyUnknownState", PureVesselMove.NONE, PureVesselPhase.UNKNOWN),
]
for state, move, phase in cases:
    check("%s -> %s/%s" % (state, move.name, phase.name),
          boss_move({"boss_control_state": state}) == move
          and boss_phase({"boss_control_state": state}) == phase)

check("empty state -> UNKNOWN phase", boss_phase({}) == PureVesselPhase.UNKNOWN)

print("== 4. stagger fires exactly once ==")
r, bd = calculate_reward(make_state("Slash1"), make_state("Stun"),
                         done=False, return_breakdown=True)
check("enter stun triggers stagger", "boss staggered" in [l for l, _ in bd])
r, bd = calculate_reward(make_state("Stun"), make_state("StunToIdle"),
                         done=False, return_breakdown=True)
check("Stun -> StunToIdle does NOT re-trigger stagger",
      "boss staggered" not in [l for l, _ in bd])

print("== 5. observation width ==")
from enviroment.state_to_vector import OBSERVATION_SIZE, BASE_FEATURES, state_to_vector
check("BASE_FEATURES == 61", BASE_FEATURES == 61)
check("OBSERVATION_SIZE == 125 (61 + 8*8)", OBSERVATION_SIZE == 125)
v = state_to_vector(make_state("Slash1 Antic"), time_frac=0.5)
check("vector width matches", v is not None and len(v) == OBSERVATION_SIZE)

print()
failed = sum(1 for r in results if not r)
print("%d/%d passed" % (len(results) - failed, len(results)))
if failed:
    raise SystemExit(1)
print("ALL PASS")
