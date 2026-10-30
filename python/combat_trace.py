"""Optional full-fight diagnostics; no policy inputs or commands are changed."""
import json
import os
from pathlib import Path
import time

from agent.action import boss_move, boss_phase


def trace_state(state):
    """Keep all collider IDs, including hazards outside the eight policy slots."""
    return {
        "health": int(state.get("knight_health", 0)),
        "damage_count": int(state.get("damage_count", 0)),
        "hit_count": int(state.get("hit_count", 0)),
        "boss_health": int(state.get("boss_health", -1)),
        "boss_fsm": state.get("boss_control_state", ""),
        "boss_move": boss_move(state).name,
        "boss_phase": boss_phase(state).name,
        "boss_state_age": float(state.get("boss_state_age", 0.0)),
        "shade_ready": bool(state.get("can_shade_cloak_now", False)),
        "dashing": bool(state.get("is_dashing", False)),
        "invulnerable": bool(state.get("is_invuln", False)),
        "knight_box": list(state["knight_box"]) if state.get("knight_box") is not None else None,
        "boss_box": list(state["boss_box"]) if state.get("boss_box") is not None else None,
        "knight_velocity": list(state.get("knight_velocity", (0.0, 0.0))),
        "hazards": [{"instance_id": int(h.instance_id), "box": list(h.box),
                     "velocity": list(h.velocity), "boss_attached": bool(h.boss_attached)}
                    for h in state.get("hazards", [])],
    }


class CombatTrace:
    def __init__(self, directory, *, host, port, step_dt):
        self.directory = Path(directory)
        self.host, self.port, self.step_dt = str(host), int(port), float(step_dt)
        self.stream = None
        self.started = None
        self.path = None

    @classmethod
    def from_environment(cls, step_dt):
        directory = os.environ.get("HOLLOWRL_TRACE_DIR")
        return (cls(directory, host=os.environ.get("HOLLOWRL_HOST", "127.0.0.1"),
                    port=os.environ.get("HOLLOWRL_PORT", "9000"), step_dt=step_dt)
                if directory else None)

    def _write(self, value):
        self.stream.write(json.dumps(value, separators=(",", ":")) + "\n")
        self.stream.flush()

    def begin(self, state, reset_count, *, boss_level, reward_profile):
        self.close("reset_before_terminal")
        self.directory.mkdir(parents=True, exist_ok=True)
        name = f"{self.host.replace('.', '_')}_{self.port}_{os.getpid()}_{reset_count:03d}.jsonl"
        self.path = self.directory / name
        self.stream = self.path.open("x", encoding="utf-8")
        self.started = time.monotonic()
        self._write({"event": "reset", "schema_version": 1, "host": self.host,
                     "port": self.port, "reset_count": reset_count,
                     "boss_level": boss_level, "reward_profile": reward_profile,
                     "step_dt": self.step_dt, "state": trace_state(state)})

    def transition(self, previous, current, action, previous_action, *, step,
                   reward, reward_parts, terminated, truncated, won, truncated_reason):
        if self.stream is None:
            raise RuntimeError("Trace transition has no matching reset")
        buttons = [int(v) for v in action]
        previous_buttons = ([int(v) for v in previous_action]
                            if previous_action is not None else [0] * len(buttons))
        self._write({"event": "transition", "step": int(step),
                     "elapsed_control_seconds": int(step) * self.step_dt,
                     "elapsed_wall_seconds": time.monotonic() - self.started,
                     "action": buttons, "previous_action": previous_buttons,
                     "dash_pressed": bool(buttons[6] and not previous_buttons[6]),
                     "before": trace_state(previous), "after": trace_state(current),
                     "reward": float(reward), "reward_parts": dict(reward_parts),
                     "terminated": bool(terminated), "truncated": bool(truncated),
                     "won": bool(won), "truncated_reason": truncated_reason})
        if terminated or truncated:
            self.stream.close()
            self.stream = None

    def close(self, reason="environment_closed"):
        if self.stream is not None:
            self._write({"event": "interrupted", "reason": reason})
            self.stream.close()
            self.stream = None
