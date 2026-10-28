"""What the agent can do, and what the boss is currently doing.

Two halves:
  - the action groups at the bottom - one dispatch table per independent
    action head, mapping the integer the policy samples to the key presses
    that carry it out.
  - boss_attack(), which reads the boss's current move back out of the game
    state so the reward function and the observation vector can both see it.
"""

from enum import IntEnum
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


class BossMove(IntEnum):
    """Traitor Lord's moves, keyed off the Mantis FSM's ActiveStateName."""

    NONE = 0
    DASH = 1        # 横向冲锋：地面冲刺/挥砍突进
    DIVE = 2        # 跳起来斜冲锋：空中俯冲
    SHOCKWAVE = 3   # 放波：砸地冲击波
    SICKLE = 4      # 放飞镖：回旋飞刀


# Mantis FSM ActiveStateName -> move. Attack-relevant states only; anything
# else (Idle, Walk, Turn, Feint, Intro/roar, ...) falls through to NONE.
_FSM_TO_MOVE = {
    # 横向冲锋（地面 Dash / 挥砍突进）
    "Attack Antic": BossMove.DASH,
    "Attack 1": BossMove.DASH,
    "Attack Swipe": BossMove.DASH,
    "Attack Recover": BossMove.DASH,
    # 跳起来斜冲锋（空中俯冲）
    "Jump Antic": BossMove.DIVE,
    "Jump": BossMove.DIVE,
    "DSlash Antic": BossMove.DIVE,
    "DSlash": BossMove.DIVE,
    "Land": BossMove.DIVE,
    # 放波（砸地冲击波）
    "Slam Antic": BossMove.SHOCKWAVE,
    "Slamming": BossMove.SHOCKWAVE,
    "Waves": BossMove.SHOCKWAVE,
    "Slam End": BossMove.SHOCKWAVE,
    # 放飞镖（回旋飞刀）
    "Sickle Antic": BossMove.SICKLE,
    "Sickle Throw": BossMove.SICKLE,
    "Throw?": BossMove.SICKLE,
    "Sickle Throw Recover": BossMove.SICKLE,
    "Sick Throw CD": BossMove.SICKLE,
}


def boss_move(state):
    """Return Traitor Lord's current move from the Mantis FSM state."""
    fsm_state = state.get("boss_control_state", "")
    return _FSM_TO_MOVE.get(fsm_state, BossMove.NONE)


class BossPhase(IntEnum):
    """Attack phase, keyed off the same Mantis FSM state (verified sequences)."""

    NEUTRAL = 0
    WINDUP = 1
    ACTIVE = 2
    RECOVERY = 3
    UNKNOWN = 4


# Mantis FSM ActiveStateName -> phase, from the verified state sequences in the
# diagnostic TSVs. States NOT listed here fall through to UNKNOWN (never guessed
# as a specific attack phase).
_FSM_TO_PHASE = {
    # 前摇（Antic 状态）
    "Attack Antic": BossPhase.WINDUP,
    "Jump Antic": BossPhase.WINDUP,
    "DSlash Antic": BossPhase.WINDUP,
    "Slam Antic": BossPhase.WINDUP,
    "Sickle Antic": BossPhase.WINDUP,
    # 危险/进行中
    "Attack 1": BossPhase.ACTIVE,
    "Attack Swipe": BossPhase.ACTIVE,
    "Jump": BossPhase.ACTIVE,
    "DSlash": BossPhase.ACTIVE,
    "Slamming": BossPhase.ACTIVE,
    "Waves": BossPhase.ACTIVE,
    "Sickle Throw": BossPhase.ACTIVE,
    "Throw?": BossPhase.ACTIVE,
    # 收招
    "Attack Recover": BossPhase.RECOVERY,
    "Land": BossPhase.RECOVERY,
    "Slam End": BossPhase.RECOVERY,
    "Sickle Throw Recover": BossPhase.RECOVERY,
    "Sick Throw CD": BossPhase.RECOVERY,
    "Cooldown": BossPhase.RECOVERY,
    # 开场（Intro：出场 + 吼叫）
    "Init": BossPhase.NEUTRAL,
    "Emerge Dust": BossPhase.NEUTRAL,
    "Fall": BossPhase.NEUTRAL,
    "Intro Land": BossPhase.NEUTRAL,
    "Roar": BossPhase.NEUTRAL,
    "Roar End": BossPhase.NEUTRAL,
    "Roar Recover": BossPhase.NEUTRAL,
    # 中性（走位/选择/转身/假动作）
    "Idle": BossPhase.NEUTRAL,
    "Walk": BossPhase.NEUTRAL,
    "Turn": BossPhase.NEUTRAL,
    "Set Speed": BossPhase.NEUTRAL,
    "Speed Check": BossPhase.NEUTRAL,
    "Active": BossPhase.NEUTRAL,
    "Attack Choice": BossPhase.NEUTRAL,
    "Stop At Edge": BossPhase.NEUTRAL,
    "Feint": BossPhase.NEUTRAL,
    "Feint?": BossPhase.NEUTRAL,
    "Feint 2": BossPhase.NEUTRAL,
    "Repeat": BossPhase.NEUTRAL,
    "Repeat?": BossPhase.NEUTRAL,
    "Repeat Pos Check": BossPhase.NEUTRAL,
    "Check L": BossPhase.NEUTRAL,
    "Check R": BossPhase.NEUTRAL,
    "Shift L": BossPhase.NEUTRAL,
    "Shift R": BossPhase.NEUTRAL,
    "Ready": BossPhase.NEUTRAL,
}


def boss_phase(state):
    """Return the attack phase from the Mantis FSM state.

    Unknown/missing states map to UNKNOWN, never to a guessed attack phase.
    """
    fsm_state = state.get("boss_control_state", "")
    if not fsm_state:
        return BossPhase.UNKNOWN
    return _FSM_TO_PHASE.get(fsm_state, BossPhase.UNKNOWN)


def boss_attack(state):
    """Binary hazard-active signal, kept for the dodge reward.

    Traitor Lord's body contributes one persistent contact collider (an
    attached hazard). An additional collider, or any detached DamageHero object
    (sickle, shockwave), means an attack hazard is live.
    """
    hazards = state.get("hazards", [])
    if len(hazards) > 1 or any(not hazard.boss_attached for hazard in hazards):
        return BossAttack.ACTIVE
    return BossAttack.NONE


# Traitor Lord's intro states: the boss has spawned and is doing its entrance
# (dust + roar) but combat has not actually begun.
INTRO_STATES = frozenset({
    "Init", "Emerge Dust", "Fall", "Intro Land",
    "Roar", "Roar End", "Roar Recover",
})


def is_intro_state(fsm_state):
    return fsm_state in INTRO_STATES


# ======================================================================
# MACRO ACTIONS
#
# Independent move/look/attack/dash/jump heads compacted the network output,
# but still explored 5*3*2*2*3 = 180 joint button combinations.  Many were
# contradictory or physically impossible.  One categorical head now chooses
# a deliberate option; useful combinations such as "jump left" stay explicit.


class MacroAction(IntEnum):
    IDLE = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    DASH_LEFT = 3
    DASH_RIGHT = 4
    JUMP = 5
    JUMP_LEFT = 6
    JUMP_RIGHT = 7
    ATTACK_LEFT = 8
    ATTACK_RIGHT = 9
    ATTACK_UP = 10
    ATTACK_DOWN = 11


def _idle():
    movement.stop()
    look.stop_looking()


def _move_left():
    look.stop_looking()
    movement.move_left()


def _move_right():
    look.stop_looking()
    movement.move_right()


def _dash_left():
    _move_left()
    dash()


def _dash_right():
    _move_right()
    dash()


def _jump_neutral():
    _idle()
    jump(duration=0.4)


def _jump_left():
    _move_left()
    jump(duration=0.4)


def _jump_right():
    _move_right()
    jump(duration=0.4)


def _attack_left():
    _idle()
    movement.turn_left()
    swing_nail()


def _attack_right():
    _idle()
    movement.turn_right()
    swing_nail()


def _attack_up():
    movement.stop()
    look.look_up()
    swing_nail()


def _attack_down():
    movement.stop()
    look.look_down()
    swing_nail()


macro_actions = {
    MacroAction.IDLE: _idle,
    MacroAction.MOVE_LEFT: _move_left,
    MacroAction.MOVE_RIGHT: _move_right,
    MacroAction.DASH_LEFT: _dash_left,
    MacroAction.DASH_RIGHT: _dash_right,
    MacroAction.JUMP: _jump_neutral,
    MacroAction.JUMP_LEFT: _jump_left,
    MacroAction.JUMP_RIGHT: _jump_right,
    MacroAction.ATTACK_LEFT: _attack_left,
    MacroAction.ATTACK_RIGHT: _attack_right,
    MacroAction.ATTACK_UP: _attack_up,
    MacroAction.ATTACK_DOWN: _attack_down,
}

MACRO_ACTION_NAMES = tuple(action.name.lower() for action in MacroAction)
ATTACK_MACRO_ACTIONS = frozenset({
    MacroAction.ATTACK_LEFT,
    MacroAction.ATTACK_RIGHT,
    MacroAction.ATTACK_UP,
    MacroAction.ATTACK_DOWN,
})


def execute_macro_action(action):
    """Execute one policy option and fail loudly on an invalid index."""
    try:
        action = MacroAction(int(action))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid macro action {action!r}") from exc
    macro_actions[action]()


def is_attack_action(action):
    try:
        return MacroAction(int(action)) in ATTACK_MACRO_ACTIONS
    except (TypeError, ValueError):
        return False


# ======================================================================
# MULTI-BINARY BUTTON ACTIONS
#
# One independent Bernoulli head per button. This is what lets the agent
# press "left + attack" at the same time, which the single 12-way macro head
# could not express. Movement and jump stay held while selected. Attack is a
# repeated tap request while selected, matching human held-attack labels;
# dash remains edge-triggered so one decision cannot spend multiple dashes.
# ======================================================================

BUTTON_ORDER = ("left", "right", "up", "down", "attack", "jump", "dash")
NUM_BUTTONS = len(BUTTON_ORDER)

HELD_BUTTONS = frozenset({"left", "right", "up", "down", "jump"})
TAP_BUTTONS = frozenset({"attack", "dash"})


def execute_buttons(buttons, prev_buttons=None):
    """Apply a multi-binary button vector and return the new previous state.

    ``buttons`` is a length-``NUM_BUTTONS`` sequence of 0/1 aligned with
    ``BUTTON_ORDER``. ``prev_buttons`` is the vector from the previous step
    (or None on the first step) and is used only to detect press/release edges.
    """
    if len(buttons) != NUM_BUTTONS:
        raise ValueError(f"expected {NUM_BUTTONS} buttons, got {len(buttons)}")
    if (buttons[0] and buttons[1]) or (buttons[2] and buttons[3]):
        raise ValueError("opposing directions cannot be pressed together")
    if prev_buttons is None:
        prev_buttons = [0] * NUM_BUTTONS
    for index, name in enumerate(BUTTON_ORDER):
        on = bool(buttons[index])
        was_on = bool(prev_buttons[index])
        if name == "attack":
            if on:
                controls.tap(name)
        elif name == "dash":
            if on and not was_on:
                controls.tap(name)
        else:
            if on and not was_on:
                controls.key_down(name)
            elif not on and was_on:
                controls.key_up(name)
    return list(buttons)
