"""Minimal live viewer for state_to_vector() - lists every field in the
observation vector with its current value, updated a few times a second.

Just for eyeballing the raw inputs the model sees; no reward/policy display
here (see test_gui_client.py for that).
"""

import tkinter as tk

import game_state
from enviroment import get_state, state_to_vector

FEATURE_NAMES = [
    "knight_health", "mana", "knight_facing_right",
    "knight_vel_x", "knight_vel_y", "attacking", "is_dashing",
    "is_jumping", "is_falling", "is_recoiling", "is_on_ground",
    "touching_wall_left", "touching_wall_right",
    "wall_distance_left", "wall_distance_right", "shade_cloak_ready",
    "is_invuln",
    "attack_dx", "attack_dy", "attack_width", "attack_height", "attack_present",
    "boss_present", "boss_health", "boss_dx", "boss_dy",
    "boss_rel_vel_x", "boss_rel_vel_y", "boss_width", "boss_height",
    "boss_facing_left", "boss_faces_knight", "knight_faces_boss",
    "hit_count", "damage_count", "time_frac", "boss_state_age",
    "hazard_count", "nearest_hazard_edge_distance",
    "nearest_left", "nearest_right", "nearest_above", "nearest_below",
    "attack_active",
    "move_NONE", "move_TRIPLE_SLASH", "move_DASH", "move_DAGGER",
    "move_PLUME", "move_DSTAB", "move_COUNTER", "move_FOCUS",
    "move_TENDRIL", "move_TELEPORT", "move_STUN", "move_EVADE",
    "phase_NEUTRAL", "phase_WINDUP", "phase_ACTIVE",
    "phase_RECOVERY", "phase_STUN", "phase_UNKNOWN",
]

for slot in range(1, 9):
    FEATURE_NAMES.extend([
        f"hazard{slot}_dx", f"hazard{slot}_dy",
        f"hazard{slot}_width", f"hazard{slot}_height",
        f"hazard{slot}_rel_vel_x", f"hazard{slot}_rel_vel_y",
        f"hazard{slot}_boss_attached", f"hazard{slot}_present",
    ])

COLUMNS = 3
REFRESH_MS = 150


class StateVectorViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("HollowRL - State Vector Viewer")

        self.status_label = tk.Label(
            root, text="Connecting...", font=("Consolas", 11, "bold")
        )
        self.status_label.grid(
            row=0, column=0, columnspan=COLUMNS * 2, sticky="w", padx=8, pady=(8, 6)
        )

        self.value_labels = []
        rows_per_col = -(-len(FEATURE_NAMES) // COLUMNS)  # ceil division

        for i, name in enumerate(FEATURE_NAMES):
            col = i // rows_per_col
            row = i % rows_per_col + 1

            tk.Label(
                root, text=f"[{i:2d}] {name}", font=("Consolas", 10), anchor="w"
            ).grid(row=row, column=col * 2, sticky="w", padx=(10, 4), pady=1)

            value_label = tk.Label(
                root, text="-", font=("Consolas", 10, "bold"), anchor="w", width=10
            )
            value_label.grid(row=row, column=col * 2 + 1, sticky="w", padx=(0, 14), pady=1)
            self.value_labels.append(value_label)

        self._tick()

    def _tick(self):
        if not game_state.connected or game_state.knight_box is None:
            self.status_label.config(text="Disconnected - waiting for game...", fg="red")
            for lbl in self.value_labels:
                lbl.config(text="-", fg="gray")
            self.root.after(REFRESH_MS, self._tick)
            return

        state = get_state()
        vector = state_to_vector(state)

        if vector is None:
            self.status_label.config(text="Connected, no valid state yet...", fg="orange")
            for lbl in self.value_labels:
                lbl.config(text="-", fg="gray")
        else:
            self.status_label.config(text="Connected", fg="green")
            for lbl, value in zip(self.value_labels, vector):
                text = f"{value:.3f}" if isinstance(value, float) else str(value)
                lbl.config(text=text, fg="black")

        self.root.after(REFRESH_MS, self._tick)


if __name__ == "__main__":
    game_state.start()
    root = tk.Tk()
    StateVectorViewer(root)
    root.mainloop()
