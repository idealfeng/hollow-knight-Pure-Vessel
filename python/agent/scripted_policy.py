"""Small deterministic reachability baseline for Pure Vessel diagnostics.

This is deliberately simple: it tests whether the current observations and
macro actions can approach and hit the boss at all. It is not an expert policy
or a replacement for PPO.
"""

from agent.action import MacroAction, PureVesselMove, PureVesselPhase, boss_move, boss_phase
from agent.calc import box_center, box_distance


def approach_action(state, require_facing=False):
    knight_box = state.get("knight_box")
    if knight_box is None or state.get("boss_health", -1) <= 0:
        return MacroAction.IDLE

    knight_x, knight_y = box_center(knight_box)
    boss_box = state.get("boss_box")
    boss_x, boss_y = (box_center(boss_box) if boss_box is not None
                      else state["boss_position"])
    dx, dy = boss_x - knight_x, boss_y - knight_y

    # A missing body box is normal briefly during teleport. Follow the root,
    # but do not swing at a location whose body cannot be observed.
    if boss_box is None:
        if abs(dx) <= 1.0:
            return MacroAction.IDLE
        return MacroAction.MOVE_RIGHT if dx > 0 else MacroAction.MOVE_LEFT

    gap = box_distance(knight_box, boss_box)
    if gap > 2.5 or abs(dx) > 4.0:
        return MacroAction.MOVE_RIGHT if dx > 0 else MacroAction.MOVE_LEFT

    if state.get("attacking", False):
        return MacroAction.IDLE
    if dy > 3.0:
        return MacroAction.ATTACK_UP
    if dy < -3.0 and not state.get("is_on_ground", False):
        return MacroAction.ATTACK_DOWN
    if require_facing and state.get("knight_facing_right") != (dx >= 0):
        return MacroAction.MOVE_RIGHT if dx >= 0 else MacroAction.MOVE_LEFT
    return MacroAction.ATTACK_RIGHT if dx >= 0 else MacroAction.ATTACK_LEFT


def evasive_action(state):
    """Candidate demonstration policy: react to visible windups, then approach.

    It uses only the same current memory fields exposed to the learner.
    Performance must be measured before treating these actions as good labels.
    """
    knight_box = state.get("knight_box")
    if knight_box is None or state.get("boss_health", -1) <= 0:
        return MacroAction.IDLE
    if state.get("is_recoiling") or state.get("is_dashing"):
        return MacroAction.IDLE

    kx, ky = box_center(knight_box)
    body = state.get("boss_box")
    bx, by = box_center(body) if body is not None else state["boss_position"]
    dx, dy = bx - kx, by - ky
    toward = MacroAction.MOVE_RIGHT if dx >= 0 else MacroAction.MOVE_LEFT
    away = MacroAction.MOVE_LEFT if dx >= 0 else MacroAction.MOVE_RIGHT
    dash_toward = MacroAction.DASH_RIGHT if dx >= 0 else MacroAction.DASH_LEFT
    dash_away = MacroAction.DASH_LEFT if dx >= 0 else MacroAction.DASH_RIGHT
    move, phase = boss_move(state), boss_phase(state)
    ground = state.get("is_on_ground", False)
    shade = state.get("can_shade_cloak_now", False)
    dangerous = phase in (PureVesselPhase.WINDUP, PureVesselPhase.ACTIVE)

    # Do not trigger the counter stance by swinging directly into it.
    if move == PureVesselMove.COUNTER and dangerous:
        return away if abs(dx) < 6 else MacroAction.IDLE

    if move == PureVesselMove.FOCUS and dangerous and abs(dx) < 10:
        return dash_away if shade and abs(dx) < 5 else away

    if move == PureVesselMove.DSTAB and dangerous and abs(dx) < 4:
        return dash_away if shade else away

    # Ground sword swings and lunges can be cleared by a held jump. A dash
    # through the boss is a fallback if an active lunge is already close.
    if move in (PureVesselMove.TRIPLE_SLASH, PureVesselMove.DASH) and dangerous:
        if ground and abs(dx) < 16:
            if phase == PureVesselPhase.ACTIVE and shade and abs(dx) < 5:
                return dash_toward
            return MacroAction.JUMP

    if move == PureVesselMove.DAGGER and dangerous and ground:
        return MacroAction.JUMP_RIGHT if dx >= 0 else MacroAction.JUMP_LEFT

    if move == PureVesselMove.TENDRIL and not state.get("boss_control_state", "").endswith("Recover"):
        if ground:
            return MacroAction.JUMP

    if move == PureVesselMove.PLUME:
        half_width = (knight_box[2] - knight_box[0]) / 2
        for hazard in state.get("hazards", []):
            left, bottom, right, top = hazard.box
            if not hazard.boss_attached and top - bottom > 3 and left - half_width < kx < right + half_width:
                return MacroAction.MOVE_LEFT if kx - left < right - kx else MacroAction.MOVE_RIGHT

    if (not ground and body is not None and dy < -1.0 and abs(dx) < 3.5
            and box_distance(knight_box, body) < 3.0 and not state.get("attacking")):
        return MacroAction.ATTACK_DOWN

    return approach_action(state, require_facing=True)
