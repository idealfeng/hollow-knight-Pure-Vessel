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
    "StunToIdle": PureVesselMove.STUN,
    "Evade Antic": PureVesselMove.EVADE,
    "Evade": PureVesselMove.EVADE,
    "EvadeRecover": PureVesselMove.EVADE,
}


def boss_move(state):
    """Return Pure Vessel's current move from the Control FSM state."""
    fsm_state = state.get("boss_control_state", "")
    return _FSM_TO_MOVE.get(fsm_state, PureVesselMove.NONE)


class PureVesselPhase(IntEnum):
    """Attack phase, keyed off the same Control FSM state (verified sequences)."""

    NEUTRAL = 0
    WINDUP = 1
    ACTIVE = 2
    RECOVERY = 3
    STUN = 4
    UNKNOWN = 5


# Control FSM ActiveStateName -> phase, from the verified state sequences in the
# diagnostic TSVs. States NOT listed here fall through to UNKNOWN (never guessed
# as a specific attack phase).
_FSM_TO_PHASE = {
    "Slash1 Antic": PureVesselPhase.WINDUP,
    "Slash1": PureVesselPhase.ACTIVE,
    "Slash1 Recover": PureVesselPhase.RECOVERY,
    "Slash2 Recover": PureVesselPhase.RECOVERY,
    "Slash 2": PureVesselPhase.ACTIVE,
    "Slash 3": PureVesselPhase.ACTIVE,
    "Slash3 Recover": PureVesselPhase.RECOVERY,
    "Dash Antic": PureVesselPhase.WINDUP,
    "Dash": PureVesselPhase.ACTIVE,
    "Dash Recover": PureVesselPhase.RECOVERY,
    "SmallShot Antic": PureVesselPhase.WINDUP,
    "SmallShot LowHigh": PureVesselPhase.ACTIVE,
    "SmallShot Recover": PureVesselPhase.RECOVERY,
    "Plume Up": PureVesselPhase.ACTIVE,
    "Plume Pause": PureVesselPhase.ACTIVE,
    "Burst Pause": PureVesselPhase.ACTIVE,
    "Dstab Antic": PureVesselPhase.WINDUP,
    "Dstab Jump": PureVesselPhase.ACTIVE,
    "Dstab Air": PureVesselPhase.ACTIVE,
    "Stomp Antic": PureVesselPhase.WINDUP,
    "Stomp Down": PureVesselPhase.ACTIVE,
    "Stomp Recover": PureVesselPhase.RECOVERY,
    "Counter Stance": PureVesselPhase.WINDUP,
    "Counter Antic": PureVesselPhase.WINDUP,
    "Counter End": PureVesselPhase.ACTIVE,
    "CS Antic": PureVesselPhase.WINDUP,
    "CSlash": PureVesselPhase.ACTIVE,
    "CSlash Recover": PureVesselPhase.RECOVERY,
    "Ball Up": PureVesselPhase.WINDUP,
    "Focus Charge": PureVesselPhase.WINDUP,
    "Focus Wait": PureVesselPhase.WINDUP,
    "Focus Burst": PureVesselPhase.ACTIVE,
    "Hit End": PureVesselPhase.RECOVERY,
    "Focus Recover": PureVesselPhase.RECOVERY,
    "Tendril Antic": PureVesselPhase.UNKNOWN,
    "Tendril Burst": PureVesselPhase.UNKNOWN,
    "Tendril Start": PureVesselPhase.UNKNOWN,
    "Tendril Hit": PureVesselPhase.UNKNOWN,
    "T2": PureVesselPhase.UNKNOWN,
    "T3": PureVesselPhase.UNKNOWN,
    "Tendril Hit End": PureVesselPhase.UNKNOWN,
    "Tendril Recover": PureVesselPhase.UNKNOWN,
    "Tele Out": PureVesselPhase.NEUTRAL,
    "Tele In": PureVesselPhase.NEUTRAL,
    "Tele Cancel": PureVesselPhase.NEUTRAL,
    "Stun": PureVesselPhase.STUN,
    "Stun Fall": PureVesselPhase.STUN,
    "Stun Air": PureVesselPhase.STUN,
    "StunHitUp": PureVesselPhase.STUN,
    "StunToIdle": PureVesselPhase.RECOVERY,
    "Evade Antic": PureVesselPhase.WINDUP,
    "Evade": PureVesselPhase.ACTIVE,
    "EvadeRecover": PureVesselPhase.RECOVERY,
    "Idle": PureVesselPhase.NEUTRAL,
    "Turn Anim": PureVesselPhase.NEUTRAL,
    "Recover": PureVesselPhase.RECOVERY,
    "Pause": PureVesselPhase.NEUTRAL,
    "Intro 1": PureVesselPhase.NEUTRAL,
    "Intro 2": PureVesselPhase.NEUTRAL,
    "Intro 3": PureVesselPhase.NEUTRAL,
    "Intro 4": PureVesselPhase.NEUTRAL,
    "Intro 5": PureVesselPhase.NEUTRAL,
    "Intro 6": PureVesselPhase.NEUTRAL,
    "Intro Roar": PureVesselPhase.NEUTRAL,
    "Intro Roar End": PureVesselPhase.NEUTRAL,
    "Intro Idle": PureVesselPhase.NEUTRAL,
}


def boss_phase(state):
    """Return the attack phase from the Control FSM state.

    Unknown/missing states map to UNKNOWN, never to a guessed attack phase.
    """
    fsm_state = state.get("boss_control_state", "")
    if not fsm_state:
        return PureVesselPhase.UNKNOWN
    return _FSM_TO_PHASE.get(fsm_state, PureVesselPhase.UNKNOWN)


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

# move/look heads have no do_nothing: every step explicitly sets the held key
# state (left/right/up/down), and stop releases it. do_nothing remains only on
# the instantaneous heads (attack/dash/jump) where "decline to act" is clear.
movement_actions = {
    0: movement.move_left,
    1: movement.move_right,
    2: movement.stop,
    3: movement.turn_left,
    4: movement.turn_right,
}

look_actions = {
    0: look.look_up,
    1: look.look_down,
    2: look.stop_looking,
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
