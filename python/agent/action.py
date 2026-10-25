"""What the agent can do, and what the boss is currently doing.

Two halves:
  - the action groups at the bottom - one dispatch table per independent
    action head, mapping the integer the policy samples to the key presses
    that carry it out.
  - boss_attack(), which reads the boss's current move back out of the game
    state so the reward function and the observation vector can both see it.
"""

from enum import IntEnum
from functools import partial

import controls
import game_state
from agent.calc import calculate_area


class movement:
    """Left/right ground movement. move_* hold the key (until stop()),
    turn_* only tap it to change facing without walking."""

    @staticmethod
    def move_left():
        controls.key_up("right")
        controls.key_down("left")

    @staticmethod
    def move_right():
        controls.key_up("left")
        controls.key_down("right")

    @staticmethod
    def turn_left():
        controls.tap("left")

    @staticmethod
    def turn_right():
        controls.tap("right")

    @staticmethod
    def stop():
        controls.key_up("right")
        controls.key_up("left")


class look:
    """Up/down aim, which is what selects the directional variants of the
    nail and spells. Held, like movement - hence stop_looking()."""

    @staticmethod
    def look_up():
        controls.key_up("down")
        controls.key_down("up")

    @staticmethod
    def look_down():
        controls.key_up("up")
        controls.key_down("down")

    @staticmethod
    def stop_looking():
        controls.key_up("down")
        controls.key_up("up")


def swing_nail():
    controls.tap("attack")


def dash():
    controls.tap("dash")


def jump(duration=0.4):
    """Hold jump for `duration` GAME-seconds - height scales with hold time,
    so this is what separates a short hop from a full jump.

    At 2x game_speed one real second is two game-seconds, so the button only
    needs to stay down half as long in real time to span the same game time.
    game_speed mirrors Unity's Time.timeScale, which commonly drops to 0 when
    the window loses focus - guard against that turning a hop into a
    multi-hour held button.

    The release is scheduled by the mod, not by a threading.Timer here: that
    fired from a Python thread which could be stuck behind the GIL during a
    PPO update, stretching a 0.1 s hop into a much longer one at exactly the
    moments that are hardest to notice.
    """
    speed = game_state.game_speed if game_state.game_speed > 0.1 else 1.0
    controls.tap("jump", hold_ms=duration / speed * 1000)


def do_nothing():
    """The no-op every head needs so it can decline to act this step."""


# ======================================================================
# BOSS ATTACK CLASSIFICATION


class BossAttack(IntEnum):
    """Which move the boss is in the middle of.

    Ordering is load-bearing: NONE and WINDUP are the two states in which
    nothing can hit the knight, and everything from DASH upwards is a live
    attack - see is_active, which reward.py uses to detect dodges.

    Fed to the network as a plain number (IntEnum), so the values are part of
    the observation and must not be renumbered without invalidating existing
    checkpoints.
    """

    NONE = 0
    WINDUP = 1        # standing still, about to dash
    DASH = 2
    NEEDLE = 3
    SILK = 4
    AIR_DASH = 5      # the catch-all for an active attack we can't name

    @property
    def is_active(self):
        """True while an attack is out and can actually connect."""
        return self >= BossAttack.DASH


# Active attacks are told apart by the area of the boss's first hazard box -
# each move spawns a differently sized hitbox. Areas come back from Unity as
# floats, hence the rounding to 2 dp on lookup.
_ATTACK_BY_HAZARD_AREA = {
    2.35: BossAttack.DASH,
    1.61: BossAttack.NEEDLE,
    2.29: BossAttack.SILK,
}


def boss_attack(state):
    """Classify the boss's current move from its live hazard boxes."""
    hazards = state["hazard_boxes"]

    # More than one hazard box means the boss has a weapon out, i.e. some
    # attack is active. Which one is a question of hitbox size.
    if len(hazards) > 1:
        area = round(calculate_area(hazards[0]), 2)
        return _ATTACK_BY_HAZARD_AREA.get(area, BossAttack.AIR_DASH)

    # No weapon out, but stopped dead: winding up for a dash.
    if state["boss_velocity"][0] == 0:
        return BossAttack.WINDUP

    return BossAttack.NONE


# ======================================================================
# ACTION GROUPS
#
# One dispatch table per independent action head. main.py samples one index
# from each and calls the matching entry, so a step can move, aim, attack,
# dash and jump all at once. The indices are what the policy learns - keep
# them stable or existing checkpoints become meaningless.
#
# ACTION_SIZES in main.py is derived from these lengths, so adding an entry
# here is enough to widen the corresponding head.

movement_actions = {
    0: movement.move_left,
    1: movement.move_right,
    2: movement.stop,
    3: movement.turn_left,
    4: movement.turn_right,
    5: do_nothing,
}

look_actions = {
    0: look.look_up,
    1: look.look_down,
    2: look.stop_looking,
    3: do_nothing,
}

attack_actions = {
    0: swing_nail,
    1: do_nothing,
}

dash_actions = {
    0: dash,
    1: do_nothing,
}

jump_actions = {
    0: partial(jump, duration=0.1),   # short hop
    1: partial(jump, duration=0.4),   # long jump
    2: do_nothing,
}
