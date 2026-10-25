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
    """Binary hazard-active signal, kept for the dodge reward."""

    NONE = 0
    ACTIVE = 1

    @property
    def is_active(self):
        """True while an attack hazard can actually connect."""
        return self == BossAttack.ACTIVE


class PureVesselMove(IntEnum):
    """Pure Vessel's moves, keyed off the Control FSM's ActiveStateName."""

    NONE = 0
    TRIPLE_SLASH = 1
    DASH = 2
    DAGGER = 3
    PLUME = 4
    DSTAB = 5
    COUNTER = 6
    FOCUS = 7
    TENDRIL = 8
    TELEPORT = 9
    STUN = 10
    EVADE = 11


# Control FSM ActiveStateName -> move. Attack-relevant states only; anything
# else (Idle, Intro, Turn Anim, Recover, Pause, ...) falls through to NONE.
_FSM_TO_MOVE = {
    "Slash1 Antic": PureVesselMove.TRIPLE_SLASH,
    "Slash1": PureVesselMove.TRIPLE_SLASH,
    "Slash1 Recover": PureVesselMove.TRIPLE_SLASH,
    "Slash2 Recover": PureVesselMove.TRIPLE_SLASH,
    "Slash 2": PureVesselMove.TRIPLE_SLASH,
    "Slash 3": PureVesselMove.TRIPLE_SLASH,
    "Slash3 Recover": PureVesselMove.TRIPLE_SLASH,
    "Dash Antic": PureVesselMove.DASH,
    "Dash": PureVesselMove.DASH,
    "Dash Recover": PureVesselMove.DASH,
    "SmallShot Antic": PureVesselMove.DAGGER,
    "SmallShot LowHigh": PureVesselMove.DAGGER,
    "SmallShot Recover": PureVesselMove.DAGGER,
    "Plume Up": PureVesselMove.PLUME,
    "Plume Pause": PureVesselMove.PLUME,
    "Burst Pause": PureVesselMove.PLUME,
    "Dstab Antic": PureVesselMove.DSTAB,
    "Dstab Jump": PureVesselMove.DSTAB,
    "Dstab Air": PureVesselMove.DSTAB,
    "Stomp Antic": PureVesselMove.DSTAB,
    "Stomp Down": PureVesselMove.DSTAB,
    "Stomp Recover": PureVesselMove.DSTAB,
    "Counter Stance": PureVesselMove.COUNTER,
    "Counter Antic": PureVesselMove.COUNTER,
    "Counter End": PureVesselMove.COUNTER,
    "CS Antic": PureVesselMove.COUNTER,
    "CSlash": PureVesselMove.COUNTER,
    "CSlash Recover": PureVesselMove.COUNTER,
    "Ball Up": PureVesselMove.FOCUS,
    "Focus Charge": PureVesselMove.FOCUS,
    "Focus Wait": PureVesselMove.FOCUS,
    "Focus Burst": PureVesselMove.FOCUS,
    "Hit End": PureVesselMove.FOCUS,
    "Focus Recover": PureVesselMove.FOCUS,
    "Tendril Antic": PureVesselMove.TENDRIL,
    "Tendril Burst": PureVesselMove.TENDRIL,
    "Tendril Start": PureVesselMove.TENDRIL,
    "Tendril Hit": PureVesselMove.TENDRIL,
    "T2": PureVesselMove.TENDRIL,
    "T3": PureVesselMove.TENDRIL,
    "Tendril Hit End": PureVesselMove.TENDRIL,
    "Tendril Recover": PureVesselMove.TENDRIL,
    "Tele Out": PureVesselMove.TELEPORT,
    "Tele In": PureVesselMove.TELEPORT,
    "Tele Cancel": PureVesselMove.TELEPORT,
    "Stun": PureVesselMove.STUN,
    "Stun Fall": PureVesselMove.STUN,
    "Stun Air": PureVesselMove.STUN,
    "StunHitUp": PureVesselMove.STUN,
    "Evade Antic": PureVesselMove.EVADE,
    "Evade": PureVesselMove.EVADE,
    "EvadeRecover": PureVesselMove.EVADE,
}


def boss_move(state):
    """Return Pure Vessel's current move from the Control FSM state."""
    fsm_state = state.get("boss_control_state", "")
    return _FSM_TO_MOVE.get(fsm_state, PureVesselMove.NONE)


def boss_attack(state):
    """Binary hazard-active signal, kept for the dodge reward.

    Pure Vessel normally contributes one persistent contact collider. An
    additional collider, or any detached DamageHero object, means an attack
    hazard is live.
    """
    hazards = state.get("hazards", [])
    if len(hazards) > 1 or any(not hazard.boss_attached for hazard in hazards):
        return BossAttack.ACTIVE
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
