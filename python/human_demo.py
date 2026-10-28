"""Lossless per-frame human input records; no Gym or live reader is started."""

import math

from game_state import _parse_frame

BUTTON_NAMES = (
    "left", "right", "up", "down", "attack", "jump", "dash",
    "cast", "quick_cast", "dream_nail", "super_dash",
)
BUTTON_MASK = (1 << len(BUTTON_NAMES)) - 1


def parse_demo_frame(line):
    parts = line.rstrip("\r\n").split("\t", 7)
    if len(parts) != 8:
        raise ValueError("demonstration frame must contain eight tab-separated fields")
    frame_id, wall_time, game_time = int(parts[0]), float(parts[1]), float(parts[2])
    held, pressed, released = map(int, parts[3:6])
    if (frame_id < 0 or not math.isfinite(wall_time) or not math.isfinite(game_time)
            or any(mask < 0 or mask & ~BUTTON_MASK for mask in (held, pressed, released))
            or parts[6] not in ("0", "1")):
        raise ValueError("invalid demonstration timestamps or input masks")
    state = _parse_frame(parts[7])
    record = {
        "frame": frame_id, "wall_time": wall_time, "game_time": game_time,
        "held": held, "pressed": pressed, "released": released,
        "input_enabled": parts[6] == "1", "state": parts[7],
    }
    return record, state


class EpisodeTracker:
    """Annotate combat while retaining the raw menu/intro frames separately."""

    def __init__(self):
        self.current = None
        self.completed = 0
        self.last_frame = None
        self.previous_frame_id = -1
        self.waiting_for_new_fight = False

    def observe(self, record, state):
        if record["frame"] <= self.previous_frame_id:
            raise ValueError("recorded game frames are not strictly increasing")
        self.previous_frame_id = record["frame"]
        self.last_frame = record
        began = False
        if self.waiting_for_new_fight:
            # On death, PlayerData can restore hero HP before the dying arena
            # unloads. Its old, wounded boss is not the next demonstration.
            # Re-arm only after the old boss is absent or a new intro appears.
            if state.boss_health <= 0 or state.boss_control_state.startswith("Intro"):
                self.waiting_for_new_fight = False
        if (self.current is None and not self.waiting_for_new_fight
                and state.knight_box is not None
                and state.knight_health > 0 and state.boss_health > 0
                and state.boss_box is not None and state.boss_control_state
                and not state.boss_control_state.startswith("Intro")):
            self.current = {
                "episode": self.completed + 1,
                "start_frame": record["frame"],
                "start_game_time": record["game_time"],
                "initial_boss_hp": state.boss_health,
                "initial_knight_hp": state.knight_health,
                "min_boss_hp": state.boss_health,
                "start_hits": state.hit_count, "start_damage": state.damage_count,
                "frames": 0, "button_presses": {name: 0 for name in BUTTON_NAMES},
            }
            began = True
        record["episode"] = self.current["episode"] if self.current else None
        if self.current is None:
            return began, None
        current = self.current
        current["frames"] += 1
        current["hits"] = max(0, state.hit_count - current["start_hits"])
        current["damage"] = max(0, state.damage_count - current["start_damage"])
        if state.boss_health >= 0:
            current["min_boss_hp"] = min(current["min_boss_hp"], state.boss_health)
        for i, name in enumerate(BUTTON_NAMES):
            current["button_presses"][name] += int(bool(record["pressed"] & (1 << i)))
        # A missing boss during teleport is not a terminal event.
        if state.boss_health == 0 or state.knight_health <= 0:
            current.update(
                complete=True, won=state.boss_health == 0,
                reason="win" if state.boss_health == 0 else "death",
                end_frame=record["frame"],
                game_seconds=max(0, record["game_time"] - current["start_game_time"]),
            )
            self.completed += 1
            self.current = None
            self.waiting_for_new_fight = True
            return began, current
        return began, None

    def finish_partial(self):
        if self.current is None:
            return None
        result = self.current
        result.update(complete=False, won=False, reason="recording_stopped",
                      end_frame=self.last_frame["frame"],
                      game_seconds=max(0, self.last_frame["game_time"] - result["start_game_time"]))
        self.current = None
        return result
