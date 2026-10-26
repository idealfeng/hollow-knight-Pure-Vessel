"""Testing / observation dashboard for the PPO agent.

Distinct from monitor.py (which only plots per-episode / per-update training
curves): this client watches the live game feed frame by frame so you can
actually *see* the reward function and the policy doing their thing while
you play or while the agent trains in another process.

Panels:
 - game view (knight / boss hazards / attack box)
 - live reward graph: per-frame reward + rolling average + cumulative total
 - reward breakdown for the current frame (which terms fired, and by how much)
 - a scrolling event log of notable reward events (hits, damage, dodges,
   death/defeat) so you can scroll back through what just happened
 - the current policy's action probabilities + value estimate + entropy,
   read straight from ppo_checkpoint.pth if one exists
 - a prominent "Reset Reward" button that clears the reward totals/graphs
   without restarting the dashboard
"""

import os
import time
import tkinter as tk
from tkinter import scrolledtext
from collections import deque
from functools import partial

import numpy as np
import torch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import controls
import game_state
from enviroment import (
    is_done, get_state, state_to_vector, nearest_hazard_distance,
    relative_horizontal_position, on_ground, HazardSide, OBSERVATION_SIZE,
)
from agent import calculate_reward, boss_attack, boss_move, boss_phase
from agent.action import (
    BossAttack,
    movement_actions, look_actions, attack_actions,
    dash_actions, jump_actions,
)
from ppo_model import ActorCritic
from obs_norm import RunningNorm

CANVAS_SIZE = 480
SCALE = 25  # pixels per game unit
HISTORY_LEN = 300
LOG_MAX_LINES = 400
GUI_STEP_DT = 0.03

STATE_SIZE = OBSERVATION_SIZE

# Longer than main.py's 60 s training cap: this is a manual-observation tool,
# so a 2-minute cap gives room to watch boss states without the 60 s KILL
# cutting a fight short. Still forces a death (Y hotkey) rather than running
# forever.
EPISODE_TIME_LIMIT = 120.0

# Same five independent action heads as main.py - keep these two files in sync.
GROUP_LABELS = ["move", "look", "attack", "dash", "jump"]
ACTION_GROUPS = [movement_actions, look_actions, attack_actions,
                  dash_actions, jump_actions]
ACTION_SIZES = [len(g) for g in ACTION_GROUPS]   # [5, 3, 2, 2, 3]

def _fn_name(fn):
    """Human-readable name for a dispatch entry - unwraps functools.partial
    (used for parameterized actions like jump duration) so the bound kwargs
    show up instead of raising on the missing __name__."""
    if isinstance(fn, partial):
        kwargs = ",".join(f"{k}={v}" for k, v in fn.keywords.items())
        return f"{fn.func.__name__}({kwargs})" if kwargs else fn.func.__name__
    return fn.__name__


# Read the human-readable name straight off each dispatch entry so this can
# never drift out of sync with agent/action.py.
ACTION_NAMES_PER_GROUP = [
    {i: _fn_name(fn) for i, fn in group.items()} for group in ACTION_GROUPS
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "ppo_checkpoint.pth")

ATTACK_NAMES = {
    BossAttack.NONE: "not attacking",
    BossAttack.ACTIVE: "active hazard (move not mapped yet)",
}

RELATIVE_POS_NAMES = {
    HazardSide.NONE: "no hazard",
    HazardSide.LEFT: "left",
    HazardSide.RIGHT: "right",
    HazardSide.SAME: "same",
}

# reward terms worth writing to the scrolling event log; "time step" fires
# every frame and would just be noise there
LOGGED_LABELS = {
    "hit landed",
    "boss staggered",
    "boss defeated",
    "took damage",
    "die",
}


def _flag(label, value):
    return f"{label}: {'YES' if value else 'no'}"


def _format_flags(state):
    wall_side = "-"
    if state["is_touching_wall_left"]:
        wall_side = "left"
    elif state["is_touching_wall_right"]:
        wall_side = "right"

    return (
        f"{_flag('Dashing', state['is_dashing'])}\n"
        f"{_flag('Jumping', state['is_jumping'])}   {_flag('Falling', state['is_falling'])}   "
        f"{_flag('Recoiling', state['is_recoiling'])}\n"
        f"Touching wall: {wall_side}   "
        f"distance L/R: {state['wall_distance_left']:.2f}/{state['wall_distance_right']:.2f}\n"
        f"{_flag('Can shade cloak', state['can_shade_cloak_now'])}\n"
        f"{_flag('Invulnerable', state['is_invuln'])}"
    )


def _rolling_average(values, window):
    return [
        sum(values[max(0, i - window + 1):i + 1]) / len(values[max(0, i - window + 1):i + 1])
        for i in range(len(values))
    ]


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("HollowRL PPO - Test Dashboard")

        self.paused = False
        self.previous_state = None
        self.total_reward = 0.0
        self.session_reward = 0.0
        self.frame_count = 0
        self.terminal_handled = False
        self.fight_hit_baseline = game_state.hit_count
        self.fight_damage_baseline = game_state.damage_count

        # None = no live fight to time yet; set the moment a fight is seen,
        # cleared on episode end so the next fight starts its own 60 s clock.
        self.episode_start_time = None
        # True once this episode has hit the cap, so the resulting death is
        # scored as scheduled rather than as a combat death (see reward.py).
        self.timed_out = False

        self.reward_history = deque(maxlen=HISTORY_LEN)
        self.total_reward_history = deque(maxlen=HISTORY_LEN)
        self.value_history = deque(maxlen=HISTORY_LEN)

        self._build_layout()
        self._load_model()

        game_state.start()
        self._tick_fast()
        self._tick_graphs()

    # ------------------------------------------------------------------
    # layout
    # ------------------------------------------------------------------
    def _build_layout(self):
        self.canvas = tk.Canvas(self.root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black")
        self.canvas.grid(row=0, column=0, rowspan=2, padx=5, pady=5, sticky="n")

        info = tk.Frame(self.root, padx=8, pady=8)
        info.grid(row=0, column=1, rowspan=2, sticky="n")

        self.status_label = tk.Label(info, text="Connecting...", font=("Consolas", 12, "bold"))
        self.status_label.pack(anchor="w")

        controls = tk.Frame(info)
        controls.pack(anchor="w", pady=(8, 4), fill="x")

        self.reset_button = tk.Button(
            controls, text="Reset Reward", width=14,
            font=("Consolas", 10, "bold"), fg="white", bg="firebrick",
            activebackground="tomato", command=self._reset_reward,
        )
        self.reset_button.pack(side="left")

        self.pause_button = tk.Button(controls, text="Pause", width=10, command=self._toggle_pause)
        self.pause_button.pack(side="left", padx=(6, 0))

        self.clear_log_button = tk.Button(controls, text="Clear log", width=10, command=self._clear_log)
        self.clear_log_button.pack(side="left", padx=(6, 0))

        self.reward_label = tk.Label(info, text="Reward: -", font=("Consolas", 13, "bold"))
        self.reward_label.pack(anchor="w", pady=(8, 0))

        self.total_label = tk.Label(info, text="Episode total: -   Session total: -", font=("Consolas", 10))
        self.total_label.pack(anchor="w")

        self.avg_label = tk.Label(info, text=f"Avg reward (last {HISTORY_LEN}): -", font=("Consolas", 10))
        self.avg_label.pack(anchor="w")

        self.done_label = tk.Label(info, text="Done: -", font=("Consolas", 12, "bold"))
        self.done_label.pack(anchor="w", pady=(8, 0))

        self.timer_label = tk.Label(info, text="Episode timer: -", font=("Consolas", 10))
        self.timer_label.pack(anchor="w")

        self.state_label = tk.Label(info, text="", font=("Consolas", 10), justify="left")
        self.state_label.pack(anchor="w", pady=(10, 0))

        flags_frame = tk.LabelFrame(info, text="Movement / wall / ability flags", padx=6, pady=6)
        flags_frame.pack(anchor="w", pady=(10, 0), fill="x")
        self.flags_label = tk.Label(
            flags_frame, text="-", font=("Consolas", 10), justify="left", anchor="w"
        )
        self.flags_label.pack(anchor="w")

        breakdown_frame = tk.LabelFrame(info, text="Reward breakdown (this frame)", padx=6, pady=6)
        breakdown_frame.pack(anchor="w", pady=(10, 0), fill="x")
        self.breakdown_label = tk.Label(
            breakdown_frame, text="-", font=("Consolas", 10), justify="left", anchor="w"
        )
        self.breakdown_label.pack(anchor="w")

        model_frame = tk.LabelFrame(info, text="Policy", padx=6, pady=6)
        model_frame.pack(anchor="w", pady=(10, 0), fill="x")
        self.model_status_label = tk.Label(model_frame, text="Loading...", font=("Consolas", 10), justify="left")
        self.model_status_label.pack(anchor="w")
        self.model_action_label = tk.Label(model_frame, text="", font=("Consolas", 10, "bold"), justify="left")
        self.model_action_label.pack(anchor="w")
        self.model_value_label = tk.Label(model_frame, text="", font=("Consolas", 10), justify="left")
        self.model_value_label.pack(anchor="w")

        # right column: graphs on top, scrolling event log below
        right = tk.Frame(self.root)
        right.grid(row=0, column=2, padx=5, pady=5, sticky="n")

        self.fig, (self.reward_ax, self.action_ax) = plt.subplots(1, 2, figsize=(8.5, 3.6))
        self.graph_canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.graph_canvas.get_tk_widget().pack()

        log_frame = tk.LabelFrame(self.root, text="Event log (notable reward events)", padx=4, pady=4)
        log_frame.grid(row=1, column=2, padx=5, pady=5, sticky="nsew")

        self.log_text = scrolledtext.ScrolledText(
            log_frame, width=68, height=16, font=("Consolas", 9), state="disabled", bg="black", fg="white"
        )
        self.log_text.pack(fill="both", expand=True)
        self.log_text.tag_config("good", foreground="lime")
        self.log_text.tag_config("bad", foreground="red")
        self.log_text.tag_config("neutral", foreground="light gray")
        self.log_text.tag_config("marker", foreground="cyan")

    def _toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.config(text="Resume" if self.paused else "Pause")

    def _reset_reward(self):
        self.total_reward = 0.0
        self.frame_count = 0
        self.previous_state = None
        self.terminal_handled = is_done()
        self.fight_hit_baseline = game_state.hit_count
        self.fight_damage_baseline = game_state.damage_count
        self.reward_history.clear()
        self.total_reward_history.clear()
        self.value_history.clear()

        self.reward_label.config(text="Reward: -", fg="black")
        self.total_label.config(text=f"Episode total: 0.00   Session total: {self.session_reward:.2f}")
        self.avg_label.config(text=f"Avg reward (last {HISTORY_LEN}): -")
        self.breakdown_label.config(text="-")

        self._log("--- reward reset by user ---", "marker")

    def _clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state="disabled")

    def _log(self, text, tag="neutral"):
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, text + "\n", tag)

        line_count = int(self.log_text.index("end-1c").split(".")[0])
        if line_count > LOG_MAX_LINES:
            self.log_text.delete("1.0", f"{line_count - LOG_MAX_LINES}.0")

        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

    # ------------------------------------------------------------------
    # model
    # ------------------------------------------------------------------
    def _load_model(self):
        self.model = None
        self.obs_norm = None

        if not os.path.exists(CHECKPOINT_PATH):
            self.model_status_label.config(text="No ppo_checkpoint.pth found")
            return

        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
            model = ActorCritic(STATE_SIZE, ACTION_SIZES)
            model.load_state_dict(checkpoint["model"])
            model.eval()
            self.model = model

            obs_norm = RunningNorm(STATE_SIZE)
            if "obs_norm" in checkpoint:
                obs_norm.load_state_dict(checkpoint["obs_norm"])
            self.obs_norm = obs_norm

            update = checkpoint.get("update", "?")
            episode = checkpoint.get("episode", "?")
            self.model_status_label.config(text=f"Loaded checkpoint (update {update}, episode {episode})")
        except Exception as exc:
            self.model_status_label.config(text=f"Failed to load checkpoint: {exc}")

    def _episode_time_frac(self):
        """Match main.py's time feature so the policy sees the same input
        shape it was trained on. Wall-clock here, since this dashboard paces
        itself off the OS timer rather than a fixed control rate."""
        if self.episode_start_time is None:
            return 0.0
        elapsed = time.time() - self.episode_start_time
        return min(elapsed / EPISODE_TIME_LIMIT, 1.0)

    def _policy_snapshot(self, state):
        if self.model is None:
            return None

        vector = state_to_vector(state, self._episode_time_frac())
        if vector is None:
            return None

        raw = np.asarray(vector, dtype=np.float64)
        obs = self.obs_norm.normalize(raw) if self.obs_norm is not None else raw.astype(np.float32)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits_list, value = self.model(obs_tensor)
            probs_list = [torch.softmax(logits, dim=-1).squeeze(0) for logits in logits_list]
            entropies = [torch.distributions.Categorical(probs=p).entropy().item() for p in probs_list]
            greedy = [int(torch.argmax(p).item()) for p in probs_list]

        return {
            "probs_list": [p.tolist() for p in probs_list],
            "value": value.item(),
            "entropy": sum(entropies),           # joint entropy = sum across heads
            "greedy": greedy,                    # one action index per head
        }

    # ------------------------------------------------------------------
    # fast loop: canvas + text panels
    # ------------------------------------------------------------------
    def _tick_fast(self):
        self.canvas.delete("all")

        if not game_state.connected or game_state.knight_box is None:
            self.status_label.config(text="Disconnected", fg="red")
            self.done_label.config(text="Done: -")
            self.state_label.config(text="")
            self.flags_label.config(text="-")
            self.breakdown_label.config(text="-")
            self.timer_label.config(text="Episode timer: -")
            self.previous_state = None
            self.episode_start_time = None
            self.root.after(200, self._tick_fast)
            return

        self.status_label.config(text="Connected" + (" (paused)" if self.paused else ""), fg="lime")

        knight_box = game_state.knight_box
        attacking = game_state.attacking
        attack_box = game_state.attack_box
        hazard_boxes = game_state.hazard_boxes
        knight_health = game_state.knight_health
        mana = game_state.mana
        boss_health = game_state.boss_health

        # Same 60 s episode cap as main.py's training loop, reproduced here so
        # the dashboard forces the same death/reset boundary on its own -
        # useful for watching the reward graph/event log react to a forced
        # death without needing main.py running at the same time. Timed off
        # a live fight (boss present, knight alive) rather than steps, since
        # this dashboard doesn't drive a fixed-rate control loop.
        live_fight = boss_health > 0 and knight_health > 0

        if live_fight and self.episode_start_time is None:
            self.episode_start_time = time.time()
            self.timed_out = False
            self.terminal_handled = False
            self.fight_hit_baseline = game_state.hit_count
            self.fight_damage_baseline = game_state.damage_count
            self.total_reward = 0.0
            self.previous_state = None
            self.reward_history.clear()
            self.total_reward_history.clear()
            self.value_history.clear()
        elif not live_fight:
            self.episode_start_time = None

        if self.episode_start_time is not None:
            elapsed = time.time() - self.episode_start_time
            self.timer_label.config(
                text=f"Episode timer: {elapsed:4.1f}s / {EPISODE_TIME_LIMIT:.0f}s",
                fg="red" if elapsed >= EPISODE_TIME_LIMIT - 5 else "black",
            )
            if elapsed >= EPISODE_TIME_LIMIT:
                self.timed_out = True
                controls.kill()
        else:
            self.timer_label.config(text="Episode timer: -", fg="black")

        kmin_x, kmin_y, kmax_x, kmax_y = knight_box
        center_x = (kmin_x + kmax_x) / 2
        center_y = (kmin_y + kmax_y) / 2

        def to_canvas(wx, wy):
            cx = CANVAS_SIZE / 2 + (wx - center_x) * SCALE
            cy = CANVAS_SIZE / 2 - (wy - center_y) * SCALE
            return cx, cy

        def draw_box(box, color, width=2):
            min_x, min_y, max_x, max_y = box
            x0, y0 = to_canvas(min_x, max_y)
            x1, y1 = to_canvas(max_x, min_y)
            self.canvas.create_rectangle(x0, y0, x1, y1, outline=color, width=width)

        for hazard in hazard_boxes:
            draw_box(hazard, "red")

        draw_box(knight_box, "cyan", width=3)

        if attacking and attack_box is not None:
            draw_box(attack_box, "yellow", width=2)

        current_state = get_state()
        current_state["hit_count"] = game_state.hit_count - self.fight_hit_baseline
        current_state["damage_count"] = game_state.damage_count - self.fight_damage_baseline

        if self.frame_count % 30 == 0:
            print(current_state)

        done = is_done()

        if boss_health == 0:
            done_reason = "boss defeated"
        elif knight_health <= 0:
            done_reason = "knight died"
        else:
            done_reason = "no"
        self.done_label.config(text=f"Done: {done_reason}", fg="red" if done else "black")

        attack_type = boss_attack(current_state)
        move_type = boss_move(current_state)
        phase_type = boss_phase(current_state)
        distance_text = f"{nearest_hazard_distance(knight_box, hazard_boxes):.2f}" if hazard_boxes else "-"
        relative_pos = relative_horizontal_position(current_state)
        grounded = on_ground(current_state)

        self.state_label.config(
            text=(
                f"Knight health: {knight_health}   Mana: {mana}   Boss health: {boss_health}\n"
                f"Hits this fight: {current_state['hit_count']}   "
                f"Damage this fight: {current_state['damage_count']}\n"
                f"Attacking: {'YES' if attacking else 'no'}\n"
                f"Boss attack: {ATTACK_NAMES.get(attack_type, attack_type)}\n"
                f"Boss move: {move_type.name}   Phase: {phase_type.name}\n"
                f"Distance to hazard: {distance_text}\n"
                f"Hazard relative pos: {RELATIVE_POS_NAMES.get(relative_pos, relative_pos)}\n"
                f"On ground: {'YES' if grounded else 'no'}\n"
                f"Frame: {self.frame_count}"
            )
        )

        self.flags_label.config(text=_format_flags(current_state))

        snapshot = self._policy_snapshot(current_state)
        self._latest_snapshot = snapshot
        if snapshot is not None:
            greedy_text = "  ".join(
                f"{label}:{ACTION_NAMES_PER_GROUP[i].get(a, a)}"
                for i, (label, a) in enumerate(zip(GROUP_LABELS, snapshot["greedy"]))
            )
            self.model_action_label.config(text=f"Greedy: {greedy_text}")
            self.model_value_label.config(
                text=f"V(s): {snapshot['value']:+.3f}   "
                f"Entropy (sum of {len(ACTION_GROUPS)} heads): {snapshot['entropy']:.3f}"
            )

        if (not self.paused
                and self.previous_state is not None
                and not (done and self.terminal_handled)):
            reward, breakdown = calculate_reward(
                self.previous_state, current_state, done,
                timed_out=self.timed_out, return_breakdown=True,
                step_dt=GUI_STEP_DT,
            )
            self.total_reward += reward
            self.session_reward += reward
            self.frame_count += 1

            self.reward_history.append(reward)
            self.total_reward_history.append(self.total_reward)
            if snapshot is not None:
                self.value_history.append(snapshot["value"])

            self.reward_label.config(
                text=f"Reward: {reward:+.4f}",
                fg="green" if reward > 0 else ("red" if reward < 0 else "black"),
            )
            self.total_label.config(
                text=f"Episode total: {self.total_reward:.2f}   Session total: {self.session_reward:.2f}"
            )
            if self.reward_history:
                avg = sum(self.reward_history) / len(self.reward_history)
                self.avg_label.config(text=f"Avg reward (last {len(self.reward_history)}): {avg:+.4f}")

            if breakdown:
                lines = "\n".join(f"{label:<20s} {amount:+.4f}" for label, amount in breakdown)
            else:
                lines = "(no reward terms fired)"
            self.breakdown_label.config(text=lines)

            notable = [item for item in breakdown if item[0] in LOGGED_LABELS]
            if notable:
                summary = ", ".join(f"{label} {amount:+.2f}" for label, amount in notable)
                tag = "good" if reward > 0 else ("bad" if reward < 0 else "neutral")
                self._log(f"[frame {self.frame_count:6d}] {summary}  (total {self.total_reward:+.2f})", tag)

            if done:
                self._log(f"=== episode end: {done_reason} | total reward {self.total_reward:+.2f} ===", "marker")
                self.terminal_handled = True

        if not self.paused:
            self.previous_state = current_state

        self.root.after(30, self._tick_fast)

    # ------------------------------------------------------------------
    # slow loop: matplotlib graphs
    # ------------------------------------------------------------------
    def _tick_graphs(self):
        self.reward_ax.clear()
        self.reward_ax.set_title("Reward (recent frames)")
        self.reward_ax.set_xlabel("Frame")
        self.reward_ax.set_ylabel("Reward")

        if self.reward_history:
            values = list(self.reward_history)
            self.reward_ax.plot(values, color="tab:blue", label="Reward")
            if len(values) >= 2:
                window = min(20, len(values))
                self.reward_ax.plot(
                    _rolling_average(values, window),
                    color="tab:orange",
                    label=f"Avg ({window})",
                )
            self.reward_ax.legend(loc="upper left", fontsize="small")

        self.action_ax.clear()
        self.action_ax.set_title("Action probabilities")
        self.action_ax.set_ylabel("P(action)")

        snapshot = getattr(self, "_latest_snapshot", None)
        if snapshot is not None:
            # Flatten all 6 heads into one bar chart, with a divider between
            # groups - simplest way to show every head without reworking the
            # 1x2 subplot layout into 7 panels.
            all_probs, all_labels, all_colors, boundaries = [], [], [], []
            for gi, (label, probs, greedy_a) in enumerate(
                zip(GROUP_LABELS, snapshot["probs_list"], snapshot["greedy"])
            ):
                names = ACTION_NAMES_PER_GROUP[gi]
                for a, p in enumerate(probs):
                    all_probs.append(p)
                    all_labels.append(f"{label}:{names.get(a, a)}")
                    all_colors.append("tab:green" if a == greedy_a else "tab:gray")
                boundaries.append(len(all_probs) - 0.5)

            self.action_ax.bar(range(len(all_probs)), all_probs, color=all_colors)
            self.action_ax.set_ylim(0, 1)
            self.action_ax.set_xticks(range(len(all_probs)))
            self.action_ax.set_xticklabels(all_labels, rotation=75, ha="right", fontsize=6)
            for b in boundaries[:-1]:
                self.action_ax.axvline(b, color="white", linewidth=0.6, linestyle=":")
        else:
            self.action_ax.text(0.5, 0.5, "no model", ha="center", va="center", transform=self.action_ax.transAxes)

        self.fig.tight_layout(pad=1.5)
        self.graph_canvas.draw()

        self.root.after(250, self._tick_graphs)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
