"""Small deterministic reachability baseline for Pure Vessel diagnostics.

This is deliberately simple: it tests whether the current observations and
macro actions can approach and hit the boss at all. It is not an expert policy
or a replacement for PPO.
"""

from agent.action import MacroAction, BossMove, BossPhase, boss_move, boss_phase
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
    dangerous = phase in (BossPhase.WINDUP, BossPhase.ACTIVE)

    # 横向冲锋（地面 Dash / 挥砍）：跳起躲过
    if move == BossMove.DASH and dangerous and ground and abs(dx) < 14:
        return MacroAction.JUMP

    # 空中俯冲（跳起来斜冲锋）：冲刺穿过或远离
    if move == BossMove.DIVE and dangerous:
        if shade and abs(dx) < 8:
            return dash_toward
        return away

    # 冲击波（放波）：跳起躲地面波
    if move == BossMove.SHOCKWAVE and dangerous and ground:
        return MacroAction.JUMP

    # 飞刀（放飞镖）：跳起或远离
    if move == BossMove.SICKLE and dangerous:
        if ground and abs(dx) < 10:
            return MacroAction.JUMP_RIGHT if dx >= 0 else MacroAction.JUMP_LEFT
        return away

    return approach_action(state, require_facing=True)
