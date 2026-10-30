"""Compact observations around health loss; diagnostics rather than policy inputs."""
from agent.action import boss_move, boss_phase


def combat_context(state, step, action=None):
    hazards = []
    for hazard in state.get("hazards", [])[:8]:
        hazards.append({"box": list(hazard.box), "velocity": list(hazard.velocity),
                        "boss_attached": bool(hazard.boss_attached)})
    return {
        "step": int(step), "health": int(state.get("knight_health", 0)),
        "damage_count": int(state.get("damage_count", 0)),
        "boss_health": int(state.get("boss_health", -1)),
        "boss_fsm": state.get("boss_control_state", ""),
        "boss_move": boss_move(state).name, "boss_phase": boss_phase(state).name,
        "shade_ready": bool(state.get("can_shade_cloak_now", False)),
        "dashing": bool(state.get("is_dashing", False)),
        "invulnerable": bool(state.get("is_invuln", False)),
        "knight_box": list(state["knight_box"]) if state.get("knight_box") is not None else None,
        "boss_box": list(state["boss_box"]) if state.get("boss_box") is not None else None,
        "hazards": hazards,
        "action": [int(value) for value in action] if action is not None else None,
    }
