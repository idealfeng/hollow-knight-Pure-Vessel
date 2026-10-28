"""Offline checks for macro actions, FSM phases and delayed reward events."""

from unittest import mock

from agent.action import (
    MACRO_ACTION_NAMES, MacroAction, PureVesselMove, PureVesselPhase,
    boss_move, boss_phase, execute_macro_action,
)
from agent.reward import CLEAN_DODGE, EMPTY_ATTACK, RewardTracker, calculate_reward
from agent.scripted_policy import approach_action
from enviroment.state_to_vector import BASE_FEATURES, OBSERVATION_SIZE, state_to_vector


def make_state(boss_control_state="Idle", boss_health=1000, knight_health=9,
               **overrides):
    state = {
        "knight_box": (40.0, 4.0, 42.0, 6.0),
        "boss_control_state": boss_control_state,
        "boss_state_age": 0.25,
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
        "wall_distance_left": 10.0,
        "wall_distance_right": 20.0,
        "can_shade_cloak_now": True,
        "is_invuln": False,
        "hit_count": 0,
        "damage_count": 0,
        "mana": 0,
    }
    state.update(overrides)
    return state


results = []


def check(name, condition):
    print("  %s  %s" % ("PASS" if condition else "FAIL", name))
    results.append(bool(condition))


print("== 1. twelve deliberate macro actions ==")
check("exactly 12 actions", len(MACRO_ACTION_NAMES) == 12)
check("stable indices", MacroAction.IDLE == 0 and MacroAction.ATTACK_DOWN == 11)

with mock.patch("agent.action.controls.key_up") as key_up, \
     mock.patch("agent.action.controls.key_down") as key_down, \
     mock.patch("agent.action.controls.tap") as tap:
    execute_macro_action(MacroAction.MOVE_LEFT)
    check("move-left holds left", mock.call("left") in key_down.call_args_list)
    execute_macro_action(MacroAction.DASH_RIGHT)
    check("dash-right holds right and taps dash",
          mock.call("right") in key_down.call_args_list
          and mock.call("dash") in tap.call_args_list)
    execute_macro_action(MacroAction.ATTACK_UP)
    check("up-attack aims and swings",
          mock.call("up") in key_down.call_args_list
          and mock.call("attack") in tap.call_args_list)

try:
    execute_macro_action(99)
    invalid_rejected = False
except ValueError:
    invalid_rejected = True
check("invalid macro rejected", invalid_rejected)


print("== 2. FSM -> move + phase ==")
cases = [
    ("Slash1 Antic", PureVesselMove.TRIPLE_SLASH, PureVesselPhase.WINDUP),
    ("Slash1", PureVesselMove.TRIPLE_SLASH, PureVesselPhase.ACTIVE),
    ("Slash3 Recover", PureVesselMove.TRIPLE_SLASH, PureVesselPhase.RECOVERY),
    ("Dash", PureVesselMove.DASH, PureVesselPhase.ACTIVE),
    ("SmallShot LowHigh", PureVesselMove.DAGGER, PureVesselPhase.ACTIVE),
    ("Stun", PureVesselMove.STUN, PureVesselPhase.STUN),
    ("Tendril Antic", PureVesselMove.TENDRIL, PureVesselPhase.UNKNOWN),
    ("Idle", PureVesselMove.NONE, PureVesselPhase.NEUTRAL),
]
for fsm, move, phase in cases:
    check(f"{fsm} -> {move.name}/{phase.name}",
          boss_move({"boss_control_state": fsm}) == move
          and boss_phase({"boss_control_state": fsm}) == phase)


print("== 3. dense but guarded reward ==")
_, breakdown = calculate_reward(
    make_state(boss_health=1000), make_state(boss_health=967), False,
    return_breakdown=True,
)
check("33 boss HP gives one reward unit",
      any(label == "boss damage" and abs(value - 1.0) < 1e-6
          for label, value in breakdown))

neutral_far = make_state(boss_control_state="Idle", boss_position=(80.0, 9.0),
                         boss_box=(79.0, 8.0, 81.0, 10.0))
active_far = make_state(boss_control_state="Slash1", boss_position=(80.0, 9.0),
                        boss_box=(79.0, 8.0, 81.0, 10.0))
_, neutral_terms = calculate_reward(neutral_far, neutral_far, False,
                                    return_breakdown=True)
_, active_terms = calculate_reward(active_far, active_far, False,
                                   return_breakdown=True)
check("far tax applies while neutral",
      "far from boss" in [label for label, _ in neutral_terms])
check("far tax disabled during active attack",
      "far from boss" not in [label for label, _ in active_terms])

tracker = RewardTracker()
before = make_state(attacking=False)
swinging = make_state(attacking=True)
after = make_state(attacking=False)
tracker.transition(before, swinging, False)
_, terms = tracker.transition(swinging, after, False, return_breakdown=True)
check("completed whiff receives delayed penalty",
      ("empty attack", EMPTY_ATTACK) in terms and tracker.misses == 1)

tracker.reset()
tracker.transition(before, swinging, False)
hit = make_state(attacking=True, hit_count=1, boss_health=967)
_, terms = tracker.transition(swinging, hit, False, return_breakdown=True)
check("landed swing is not a whiff", "empty attack" not in [x[0] for x in terms])

tracker.reset()
windup = make_state("Slash1 Antic")
active = make_state("Slash1")
recovery = make_state("Slash1 Recover")
tracker.transition(windup, active, False)
_, terms = tracker.transition(active, recovery, False, return_breakdown=True)
check("clean active phase exit gives tiny dodge reward",
      ("clean dodge", CLEAN_DODGE) in terms and tracker.dodges == 1)


print("== 4. observation ==")
vector = state_to_vector(make_state(), time_frac=0.5)
check("base features include FSM age", BASE_FEATURES == 62)
check("observation is 126 values", OBSERVATION_SIZE == 126 and len(vector) == 126)
body_state = make_state(boss_position=(52.0, 9.0),
                        boss_box=(60.0, 10.0, 64.0, 16.0))
body_vector = state_to_vector(body_state)
check("boss offset follows body collider centre",
      body_vector[24] == 21.0 and body_vector[25] == 8.0)
body_state["boss_box"] = None
root_vector = state_to_vector(body_state)
check("boss offset falls back to root during teleport",
      root_vector[24] == 11.0 and root_vector[25] == 4.0)
for boss_right, facing_left in ((True, True), (True, False), (False, True), (False, False)):
    box = (50.0, 4.0, 52.0, 6.0) if boss_right else (30.0, 4.0, 32.0, 6.0)
    observed = state_to_vector(make_state(boss_box=box, boss_facing_left=facing_left))[31]
    check(f"boss facing knight: right={boss_right}, left-facing={facing_left}",
          bool(observed) == (boss_right == facing_left))

print("== 5. deterministic reachability baseline ==")
check("approach policy closes a large gap",
      approach_action(make_state()) == MacroAction.MOVE_RIGHT)
near_right = make_state(boss_box=(42.5, 4.0, 44.5, 6.0))
check("approach policy attacks a nearby body",
      approach_action(near_right) == MacroAction.ATTACK_RIGHT)
check("approach policy waits while a swing is active",
      approach_action(make_state(boss_box=(42.5, 4.0, 44.5, 6.0),
                                 attacking=True)) == MacroAction.IDLE)
check("approach policy does not swing at a missing body",
      approach_action(make_state(boss_box=None)) == MacroAction.MOVE_RIGHT)
check("aligned approach turns toward body before swinging",
      approach_action(make_state(boss_box=(42.5, 4.0, 44.5, 6.0),
                                 knight_facing_right=False),
                      require_facing=True) == MacroAction.MOVE_RIGHT)

print()
failed = sum(not result for result in results)
print(f"{len(results) - failed}/{len(results)} passed")
if failed:
    raise SystemExit(1)
print("ALL PASS")
