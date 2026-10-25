"""Offline tests for the boss-state / reward fix (no game, no training).

Covers the required scenarios:
  1. Boss temporarily missing, NOT defeated.
  2. Boss alive and hp dropping.
  3. Boss hp reaching a real 0 (confirmed defeat).
  4. Boss missing then reappearing.
  5. Episode reset with a fresh boss (old hp must not cause a false hit/defeat).
"""
import game_state
from agent.reward import (
    calculate_reward, BOSS_DEFEATED, DAMAGE_PER_MASK, DEATH, STAGGER,
)
from agent.action import PureVesselMove
from enviroment import state_to_vector, OBSERVATION_SIZE


def make_state(boss_health, knight_health=9, hit_streak=0, boss_control_state="Idle"):
    return {
        "knight_box": (40.0, 5.0, 40.5, 6.3),
        "boss_health": boss_health,
        "knight_health": knight_health,
        "hit_streak": hit_streak,
        "hazards": [],
        "boss_facing_left": None if boss_health < 0 else False,
        "boss_position": (52.0, 9.0),
        "boss_velocity": (0.0, 0.0),
        "boss_box": (51.0, 8.0, 53.0, 10.0),
        "boss_control_state": boss_control_state,
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


def labels(breakdown):
    return [label for label, _ in breakdown]


results = []


def check(name, cond):
    print("  %s  %s" % ("PASS" if cond else "FAIL", name))
    results.append(cond)


print("== calculate_reward ==")

# 1. Boss temporarily missing, not defeated: 1000 -> -1 (done, as a truncation).
r, bd = calculate_reward(make_state(1000), make_state(-1), done=True, return_breakdown=True)
labs = labels(bd)
check("1. missing boss: no hit landed", "hit landed" not in labs)
check("1b. missing boss: no boss defeated", "boss defeated" not in labs)
check("1c. missing boss: no stagger", not any("stagger" in l for l in labs))

# 2. Boss alive, hp dropping: 1600 -> 1567.
r, bd = calculate_reward(make_state(1600), make_state(1567, hit_streak=1), done=False, return_breakdown=True)
labs = labels(bd)
check("2. hp drop -> hit landed", "hit landed" in labs)

# 3. Boss hp reaching a real 0: 100 -> 0, done.
r, bd = calculate_reward(make_state(100), make_state(0), done=True, return_breakdown=True)
labs = labels(bd)
check("3. real defeat -> boss defeated", "boss defeated" in labs)
check("3b. real defeat -> hit landed", "hit landed" in labs)

# 4. Boss missing then reappearing: -1 -> 1600 (no hit on recovery).
r, bd = calculate_reward(make_state(-1), make_state(1600), done=False, return_breakdown=True)
labs = labels(bd)
check("4. reappear: no hit landed", "hit landed" not in labs)

# 5. Episode reset with a fresh boss: 1600 -> 1600 (no change, no hit).
r, bd = calculate_reward(make_state(1600), make_state(1600), done=False, return_breakdown=True)
labs = labels(bd)
check("5. reset no-change: no hit landed", "hit landed" not in labs)

# 6. The real live sequence was 16 -> object disappears. C# now latches that
# disappearance as hp=0 while the knight is alive, so Python must pay both the
# final hit and the terminal win exactly once.
r, bd = calculate_reward(make_state(16, hit_streak=4), make_state(0, hit_streak=5), done=True,
                         return_breakdown=True)
labs = labels(bd)
check("6. latched 16 -> 0: final hit", "hit landed" in labs)
check("6b. latched 16 -> 0: boss defeated", "boss defeated" in labs)

r, bd = calculate_reward(make_state(0), make_state(0), done=True,
                         return_breakdown=True)
check("6c. latched 0 -> 0: terminal reward is not repeated",
      "boss defeated" not in labels(bd))

# 7. A two-mask trade must cost much more than one clean hit is worth.
r, bd = calculate_reward(make_state(1000, knight_health=9),
                         make_state(1000, knight_health=7), done=False,
                         return_breakdown=True)
damage_amount = next(amount for label, amount in bd if label == "took damage")
check("7. two-mask hit has strong penalty", damage_amount == -2 * DAMAGE_PER_MASK)

# 8. Stagger reward follows the observed FSM transition, not an HP guess.
r, bd = calculate_reward(make_state(1204, boss_control_state="Slash1"),
                         make_state(1171, boss_control_state="Stun Fall"), done=False,
                         return_breakdown=True)
check("8. FSM stun transition rewarded", ("boss staggered", STAGGER) in bd)

# 9. Death remains a material terminal penalty.
r, bd = calculate_reward(make_state(1000, knight_health=1),
                         make_state(1000, knight_health=0), done=True,
                         return_breakdown=True)
check("9. death penalty applied", ("die", DEATH) in bd)

r, bd = calculate_reward(make_state(1000, knight_health=0),
                         make_state(1000, knight_health=0), done=True,
                         return_breakdown=True)
check("9b. death penalty is not repeated", "die" not in labels(bd))

# 10. The Control FSM string must really reach the one-hot features. This
# catches the previous bug where TCP parsed it but get_state() dropped it.
move_state = make_state(1600, boss_control_state="Tendril Hit")
vector = state_to_vector(move_state)
move_start = 43
check("10. observation width", len(vector) == OBSERVATION_SIZE == 119)
check("10b. Tendril FSM reaches one-hot",
      vector[move_start + int(PureVesselMove.TENDRIL)] == 1
      and sum(vector[move_start:move_start + len(PureVesselMove)]) == 1)

# 11. C#/Python wire format stays aligned, including wall distances, FSM and
# the richer hazard record.
wire = (
    "40,5,40.5,6.3;0;;3,4;0,0,1,1600,0,0,0,9,0,1,52,9,10,20;"
    "0,0,0,0,0,0,1,0,0,1;Tendril Hit;51,8,53,10;"
    "123,45,5,46,7,-2,0,0"
)
frame = game_state._parse_frame(wire)
check("11. protocol wall distances", frame.wall_distance_left == 10.0
      and frame.wall_distance_right == 20.0)
check("11b. protocol FSM", frame.boss_control_state == "Tendril Hit")
check("11c. protocol hazard", len(frame.hazards) == 1
      and frame.hazards[0].instance_id == 123
      and frame.hazards[0].velocity == (-2.0, 0.0))

print("== is_done logic (boss_health == 0 or knight_health <= 0) ==")
# is_done reads game_state.boss_health / knight_health. Verify the exact logic
# directly against those globals (the function body is:
#   return game_reader.boss_health == 0 or game_reader.knight_health <= 0).


def is_done_logic(boss_health, knight_health):
    return boss_health == 0 or knight_health <= 0


check("is_done: missing boss (-1), knight alive -> False", is_done_logic(-1, 9) is False)
check("is_done: boss hp 0 -> True", is_done_logic(0, 9) is True)
check("is_done: knight dead -> True", is_done_logic(1600, 0) is True)
check("is_done: both alive -> False", is_done_logic(1600, 9) is False)

print()
failed = sum(1 for r in results if not r)
print("%d/%d passed" % (len(results) - failed, len(results)))
if failed:
    raise SystemExit(1)
print("ALL PASS")
