"""Minimal live viewer for state_to_vector() - lists every field in the
observation vector with its current value, updated a few times a second.

Just for eyeballing the raw inputs the model sees; no reward/policy display
here (see test_gui_client.py for that).
"""

import tkinter as tk

import game_state
from enviroment import get_state, state_to_vector

# Must match the order state_to_vector() in enviroment.py builds the list in.
# If that function's field order ever changes, update this list to match.
FEATURE_NAMES = [
    "knight_x", "knight_y", "knight_width", "knight_height",
    "knight_health", "mana", "knight_facing_right",
    "knight_vel_x", "knight_vel_y",
    "attackbox_x", "attackbox_y", "attackbox_width", "attackbox_height",
    "attacking",
    "hazard1_x", "hazard1_y", "hazard1_width", "hazard1_height",
    "hazard2_x", "hazard2_y", "hazard2_width", "hazard2_height",
    "boss_health", "boss_facing_direction",
    "hazard_vel_x", "hazard_vel_y",
    "hit_count", "damage_count",
    "attack_type", "nearest_hazard_distance",
    "relative_horizontal_position",
    "is_dashing", "is_jumping", "is_falling", "is_recoiling",
    "is_touching_wall_left", "is_touching_wall_right",
    "can_shade_cloak_now",
    "is_invuln",
    # This viewer has no episode clock of its own, so state_to_vector()'s
    # default applies and this always reads 0.0 here. It is a live value
    # during training - see main.py's episode_time_frac().
    "time_frac",
    # knight-relative offsets to the two tracked hazards
    "offset1_x", "offset1_y",
    "offset2_x", "offset2_y",
]

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
