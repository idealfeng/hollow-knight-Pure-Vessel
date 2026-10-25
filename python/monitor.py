import atexit
import math
import os
import time

import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

AVG_WINDOW = 20
MIN_SAVE_INTERVAL = 5.0  # seconds between history writes to disk

# ppo_model.evaluate() SUMS entropy across the independent action heads, so the
# ceiling is sum(ln(n)) over the head sizes - not ln(8), which was the old
# single-8-action-space value and is far below the real maximum (5.11 for
# [5,3,2,2,3] after move/look lost their do_nothing). Drawn as "max (uniform)"
# on the entropy plot, so a stale value makes a healthy policy look off the chart.
DEFAULT_ACTION_SIZES = [5, 3, 2, 2, 3]


def max_entropy(action_sizes):
    return sum(math.log(n) for n in action_sizes)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_PATH = os.path.join(SCRIPT_DIR, "monitor_history.npz")


def _rolling_average(values, window):
    """Same result as the naive "sum each trailing slice" version, but O(n)
    instead of O(n*window): that version rebuilt and summed a fresh `window`-
    sized list slice at every index, recomputed from scratch on every 1s UI
    tick for the WHOLE history - measured at 93ms/call at 50,000 episodes
    (up from 0.28ms at 166), a cost that only ever grows for the life of a
    run. This does one cumulative sum instead."""
    n = len(values)
    if n == 0:
        return []

    values = np.asarray(values, dtype=np.float64)
    csum = np.cumsum(values)
    counts = np.minimum(np.arange(1, n + 1), window)

    window_sums = csum.copy()
    lo = np.arange(n) - window
    valid = lo >= 0
    window_sums[valid] -= csum[lo[valid]]

    return (window_sums / counts).tolist()


class TrainingMonitor:

    def __init__(self, target_kl=0.03, action_sizes=None):

        self.target_kl = target_kl
        self.action_sizes = action_sizes or DEFAULT_ACTION_SIZES
        self.num_heads = len(self.action_sizes)
        self.max_entropy = max_entropy(self.action_sizes)

        self.root = tk.Tk()
        self.root.title("PPO Training Monitor")

        # per-episode series
        self.rewards = []
        self.steps = []
        self.hits = []
        self.damages = []

        # per-update series (one point per ppo_update() call)
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        self.approx_kls = []
        self.clipfracs = []
        self.explained_variances = []
        self.stopped_earlys = []

        # Per-head diagnostics and trunk/critic gradient norms - one row per
        # update, one column per head. value_loss's gradient never reaches
        # any action head (only the shared trunk and the critic layer do),
        # so grad_norm_trunk/grad_norm_critic are what actually explains a
        # value-loss spike; the per-head arrays answer a different question
        # ("is one action head moving oddly"), not that one.
        self.entropy_per_head = []
        self.approx_kl_per_head = []
        self.clipfrac_per_head = []
        self.grad_norm_per_head = []
        self.grad_norm_trunk = []
        self.grad_norm_critic = []

        self._load_history()
        self._last_save_time = 0.0
        # add_reward()/add_update() run on the TRAINING thread (main.py calls
        # them directly inside train_loop()), and previously called
        # _save_history() - a full np.savez of every array - on EVERY single
        # one of those calls. Measured cost grows with total history: 3.4 ms
        # at 166 episodes, 11.9 ms at 10,000 - real wall-clock stolen from
        # the training loop on every episode boundary, for the entire life of
        # a run, exactly the kind of hidden cost the reset.py timing fix
        # addressed a few sessions back. Throttled to at most once every
        # MIN_SAVE_INTERVAL seconds; atexit below guarantees a final save on
        # a clean shutdown so throttling can't lose more than that window.
        atexit.register(self._save_history)

        self.run_info_label = tk.Label(
            self.root,
            text="",
            font=("Consolas", 10),
            fg="gray30",
        )
        self.run_info_label.pack(pady=(6, 0))

        self.status_label = tk.Label(
            self.root,
            text="Episode: 0",
            font=("Consolas", 14, "bold"),
        )
        self.status_label.pack(pady=(8, 4))

        self.update_status_label = tk.Label(
            self.root,
            text="Update: 0",
            font=("Consolas", 11),
        )
        self.update_status_label.pack(pady=(0, 4))

        self.warning_label = tk.Label(
            self.root,
            text="",
            font=("Consolas", 11, "bold"),
            fg="red",
        )
        self.warning_label.pack(pady=(0, 4))

        # Figure(), NOT plt.subplots(). pyplot's TkAgg backend builds a
        # FigureManagerTk for every figure it creates, and that manager brings
        # its own tk.Tk() window and its own FigureCanvasTkAgg bound to this
        # same figure. Embedding then repoints fig.canvas at the canvas below
        # and leaves that second window alive, invisible, still holding
        # resize/idle-draw callbacks against a figure it no longer owns - two
        # canvases and two Tk interpreters drawing one figure, in a process
        # that is also running a training thread. Going through Figure
        # directly is the supported way to embed and leaves exactly one canvas.
        # 15x10in at 100dpi is 1500x1000px of canvas plus the labels above it -
        # taller than the usable area of a 1080p screen, so the bottom row was
        # cut off. Smaller default + a canvas that actually follows the window
        # (fill/expand, and a figure that reflows on resize) instead of a fixed
        # block that can only be too big or too small.
        self.fig = Figure(figsize=(13, 8))
        # constrained layout, applied once, replaces a tight_layout() call that
        # ran on every 1s redraw - it re-solves the whole 9-axes layout each
        # time, which is the expensive part of the tick.
        self.fig.set_layout_engine("constrained")
        axes = self.fig.subplots(3, 3)
        (self.reward_ax, self.steps_ax, self.combat_ax) = axes[0]
        (self.loss_ax, self.entropy_ax, self.kl_clip_ax) = axes[1]
        (self.ev_ax, self.reward_per_step_ax, self.summary_ax) = axes[2]
        self.summary_ax.axis("off")

        # Twin axes are created ONCE here, not in update(). Policy loss lives
        # around ±0.02 while value loss can sit in the hundreds; on one shared
        # y-axis the policy curve is a flat line pinned to zero and unreadable.
        # Same story for approx KL (~0.01) against clipfrac (0..1). Creating
        # these inside update() would add a new axes to the figure every second.
        self.value_loss_ax = self.loss_ax.twinx()
        self.clipfrac_ax = self.kl_clip_ax.twinx()

        self.canvas = FigureCanvasTkAgg(
            self.fig,
            master=self.root
        )

        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.root.after(
            1000,
            self.update
        )

    def set_run_info(self, resumed, start_update=0, start_episode=0):
        """Called once from main.py right after it knows whether it loaded a
        checkpoint. Previously this only ever went to a print() - visible in
        a terminal you may not be looking at, not in the monitor window
        itself, which is the thing actually on screen during a run."""
        if resumed:
            text = f"RESUMED from update {start_update}, episode {start_episode}"
        else:
            text = "FRESH RUN (no checkpoint loaded)"
        self.run_info_label.config(text=text)

    # ------------------------------------------------------------------
    # persistence - so resuming training from a checkpoint does not also
    # wipe the graphs back to a blank slate
    # ------------------------------------------------------------------
    def _load_history(self):
        if not os.path.exists(HISTORY_PATH):
            return

        try:
            data = np.load(HISTORY_PATH)
            self.rewards = data["rewards"].tolist()
            self.steps = data["steps"].tolist()
            self.hits = data["hits"].tolist()
            self.damages = data["damages"].tolist()
            self.policy_losses = data["policy_losses"].tolist()
            self.value_losses = data["value_losses"].tolist()
            self.entropies = data["entropies"].tolist()
            self.approx_kls = data["approx_kls"].tolist()
            self.clipfracs = data["clipfracs"].tolist()
            self.explained_variances = data["explained_variances"].tolist()
            self.stopped_earlys = data["stopped_earlys"].tolist()
            print(f"monitor: resumed {len(self.rewards)} episodes / {len(self.policy_losses)} updates of history")
        except Exception as exc:
            print(f"monitor: could not load history ({exc}), starting fresh")
            return

        # Per-head/grad-norm fields are newer than the fields above. A history
        # file saved before they existed simply won't have these keys - that
        # must NOT blow away the legacy fields already loaded successfully
        # above, so it gets its own try/except rather than sharing the one
        # above. Losing 45 updates of history to a KeyError on a field added
        # later would be exactly the kind of bug this monitor exists to catch.
        try:
            self.entropy_per_head = data["entropy_per_head"].tolist()
            self.approx_kl_per_head = data["approx_kl_per_head"].tolist()
            self.clipfrac_per_head = data["clipfrac_per_head"].tolist()
            self.grad_norm_per_head = data["grad_norm_per_head"].tolist()
            self.grad_norm_trunk = data["grad_norm_trunk"].tolist()
            self.grad_norm_critic = data["grad_norm_critic"].tolist()
        except KeyError:
            print("monitor: history predates per-head logging - "
                  "per-head panel starts fresh, everything else is intact")

    def _save_history(self):
        try:
            np.savez(
                HISTORY_PATH,
                rewards=np.array(self.rewards, dtype=np.float32),
                steps=np.array(self.steps, dtype=np.int32),
                hits=np.array(self.hits, dtype=np.int32),
                damages=np.array(self.damages, dtype=np.int32),
                policy_losses=np.array(self.policy_losses, dtype=np.float32),
                value_losses=np.array(self.value_losses, dtype=np.float32),
                entropies=np.array(self.entropies, dtype=np.float32),
                approx_kls=np.array(self.approx_kls, dtype=np.float32),
                clipfracs=np.array(self.clipfracs, dtype=np.float32),
                explained_variances=np.array(self.explained_variances, dtype=np.float32),
                stopped_earlys=np.array(self.stopped_earlys, dtype=bool),
                entropy_per_head=np.array(self.entropy_per_head, dtype=np.float32),
                approx_kl_per_head=np.array(self.approx_kl_per_head, dtype=np.float32),
                clipfrac_per_head=np.array(self.clipfrac_per_head, dtype=np.float32),
                grad_norm_per_head=np.array(self.grad_norm_per_head, dtype=np.float32),
                grad_norm_trunk=np.array(self.grad_norm_trunk, dtype=np.float32),
                grad_norm_critic=np.array(self.grad_norm_critic, dtype=np.float32),
            )
        except Exception as exc:
            print(f"monitor: could not save history ({exc})")

    def _save_history_throttled(self):
        """Write to disk at most once every MIN_SAVE_INTERVAL seconds.
        Correctness note: this can only delay a write, never drop one - each
        call still appends to the in-memory lists first, so a save some
        seconds later still captures everything since the last write. The
        only data at risk is whatever happened in the last MIN_SAVE_INTERVAL
        seconds before an UNCLEAN kill (not a normal Ctrl+C - atexit covers
        that)."""
        now = time.time()
        if now - self._last_save_time >= MIN_SAVE_INTERVAL:
            self._save_history()
            self._last_save_time = now

    def update(self):

        self.reward_ax.clear()
        self.reward_ax.set_title("Reward per Episode")
        self.reward_ax.set_xlabel("Episode")
        self.reward_ax.set_ylabel("Reward")

        if self.rewards:
            # Raw per-episode reward is extremely noisy; at full opacity and
            # the same line width as the average it drowns out the only line
            # that shows whether the agent is actually learning. Backgrounded
            # to a faint trace, average brought forward.
            self.reward_ax.plot(
                self.rewards, color="tab:blue", alpha=0.25, linewidth=0.8,
                label="Reward (raw)"
            )

            if len(self.rewards) >= 2:
                window = min(AVG_WINDOW, len(self.rewards))
                self.reward_ax.plot(
                    _rolling_average(self.rewards, window),
                    color="tab:orange",
                    linewidth=2.0,
                    label=f"Avg (last {window})"
                )

            self.reward_ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            self.reward_ax.legend(loc="upper left", fontsize="small")

        self.steps_ax.clear()
        self.steps_ax.set_title("Episode Length")
        self.steps_ax.set_xlabel("Episode")
        self.steps_ax.set_ylabel("Steps")

        if self.steps:
            self.steps_ax.plot(
                self.steps, color="tab:blue", alpha=0.25, linewidth=0.8,
                label="Steps (raw)"
            )
            if len(self.steps) >= 2:
                window = min(AVG_WINDOW, len(self.steps))
                self.steps_ax.plot(
                    _rolling_average(self.steps, window),
                    color="tab:orange",
                    linewidth=2.0,
                    label=f"Avg (last {window})"
                )
            self.steps_ax.legend(loc="upper left", fontsize="small")

        self.combat_ax.clear()
        self.combat_ax.set_title("Hits vs Damage Taken per Episode")
        self.combat_ax.set_xlabel("Episode")
        self.combat_ax.set_ylabel("Count")

        if self.hits:
            # Hits/damage are small integers that jump between 0 and 3 every
            # episode - as raw lines the two series are a solid scribble and
            # you cannot tell which one is trending. Smoothed on top.
            # No faint raw trace here, unlike the reward/steps panels: these are
            # small integers bouncing 0..3 every episode, so even at alpha 0.2
            # the raw lines fill the panel as a solid block of colour and hide
            # the averages underneath rather than adding context.
            window = min(AVG_WINDOW, len(self.hits))
            self.combat_ax.plot(
                _rolling_average(self.hits, window),
                color="tab:green", linewidth=2.0, label="Hits (avg)"
            )
            self.combat_ax.plot(
                _rolling_average(self.damages, window),
                color="tab:red", linewidth=2.0, label="Damage taken (avg)"
            )
            self.combat_ax.legend(loc="upper left", fontsize="small")

        self.loss_ax.clear()
        self.value_loss_ax.clear()
        self.loss_ax.set_title("Policy / Value Loss")
        self.loss_ax.set_xlabel("Update")
        # No y-axis labels on the twin panels: at this subplot size the label
        # text lands on top of its own tick numbers. The coloured ticks plus
        # the "(left)"/"(right)" legend entries carry the same information.
        self.loss_ax.tick_params(axis="y", labelcolor="tab:red", labelsize="small")
        self.value_loss_ax.tick_params(axis="y", labelcolor="tab:purple", labelsize="small")

        if self.policy_losses:
            # Separate y-axes: policy loss is a small signed number near zero,
            # value loss is unbounded and positive. Sharing one axis flattened
            # the policy curve into the zero line.
            l1, = self.loss_ax.plot(self.policy_losses, color="tab:red", label="Policy (left)")
            l2, = self.value_loss_ax.plot(self.value_losses, color="tab:purple", label="Value (right)")
            self.loss_ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            self.loss_ax.legend(handles=[l1, l2], loc="upper left", fontsize="small")

        self.entropy_ax.clear()
        self.entropy_ax.set_title("Policy Entropy")
        self.entropy_ax.set_xlabel("Update")
        self.entropy_ax.set_ylabel("Entropy")

        if self.entropies:
            self.entropy_ax.plot(self.entropies, color="tab:gray", label="Entropy")
            self.entropy_ax.axhline(
                self.max_entropy, color="tab:blue", linewidth=0.8, linestyle="--",
                label=f"max (uniform) = {self.max_entropy:.2f}"
            )
            self.entropy_ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            self.entropy_ax.legend(loc="upper right", fontsize="small")

        self.kl_clip_ax.clear()
        self.clipfrac_ax.clear()
        self.kl_clip_ax.set_title("Approx KL / Clip Fraction")
        self.kl_clip_ax.set_xlabel("Update")
        self.kl_clip_ax.tick_params(axis="y", labelcolor="tab:brown", labelsize="small")
        self.clipfrac_ax.tick_params(axis="y", labelcolor="tab:olive", labelsize="small")
        # clipfrac is a fraction, so pinning it to 0..1 keeps the "how close to
        # the 0.5 warning threshold am I" reading stable instead of autoscaling
        # to whatever range the last few updates happened to occupy.
        self.clipfrac_ax.set_ylim(0, 1)

        if self.approx_kls:
            # KL sits near 0.01-0.03 and clipfrac near 0.1-0.5; on one axis the
            # KL curve - the one the target line is about - was invisible.
            handles = []
            handles.append(self.kl_clip_ax.plot(
                self.approx_kls, color="tab:brown", label="approx KL (left)")[0])
            handles.append(self.clipfrac_ax.plot(
                self.clipfracs, color="tab:olive", label="clipfrac (right)")[0])
            handles.append(self.kl_clip_ax.axhline(
                self.target_kl, color="tab:red", linewidth=0.8, linestyle="--",
                label=f"target KL = {self.target_kl:g}"
            ))

            stop_x = [i for i, flag in enumerate(self.stopped_earlys) if flag]
            if stop_x:
                stop_y = [self.approx_kls[i] for i in stop_x]
                handles.append(self.kl_clip_ax.scatter(
                    stop_x, stop_y, color="tab:red", marker="x", s=40,
                    label="stopped early", zorder=3,
                ))

            self.kl_clip_ax.legend(handles=handles, loc="upper left", fontsize="small")

        self.ev_ax.clear()
        self.ev_ax.set_title("Explained Variance (critic quality)")
        self.ev_ax.set_xlabel("Update")
        self.ev_ax.set_ylabel("Explained variance")

        if self.explained_variances:
            self.ev_ax.axhspan(-10, 0, color="tab:red", alpha=0.08)
            self.ev_ax.plot(self.explained_variances, color="tab:cyan")
            self.ev_ax.axhline(1.0, color="tab:green", linewidth=0.8, linestyle="--")
            self.ev_ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
            finite = [v for v in self.explained_variances if not math.isnan(v)]
            if finite:
                self.ev_ax.set_ylim(min(-0.5, min(finite) - 0.1), 1.1)

        self.reward_per_step_ax.clear()
        self.reward_per_step_ax.set_title("Reward per Step")
        self.reward_per_step_ax.set_xlabel("Episode")
        self.reward_per_step_ax.set_ylabel("Reward / step")

        per_step = [r / s for r, s in zip(self.rewards, self.steps) if s > 0]
        if per_step:
            self.reward_per_step_ax.plot(per_step, color="tab:pink", alpha=0.25, linewidth=0.8)
            window = min(AVG_WINDOW, len(per_step))
            self.reward_per_step_ax.plot(
                _rolling_average(per_step, window),
                color="tab:pink", linewidth=2.0, label=f"Avg (last {window})"
            )
            self.reward_per_step_ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            self.reward_per_step_ax.legend(loc="upper left", fontsize="small")

        # No tight_layout() here - the constrained layout engine set in
        # __init__ reflows on every draw, including window resizes, which
        # tight_layout did not.
        self.canvas.draw()

        episode = len(self.rewards)
        latest_reward = self.rewards[-1] if self.rewards else 0.0
        latest_steps = self.steps[-1] if self.steps else 0
        latest_hits = self.hits[-1] if self.hits else 0
        latest_damage = self.damages[-1] if self.damages else 0

        status = (
            f"Episode: {episode}   Reward: {latest_reward:.2f}   "
            f"Steps: {latest_steps}   Hits: {latest_hits}   Damage: {latest_damage}"
        )
        self.status_label.config(text=status)

        warnings = []
        update_count = len(self.policy_losses)
        if update_count:
            latest_kl = self.approx_kls[-1]
            latest_clip = self.clipfracs[-1]
            latest_ev = self.explained_variances[-1]

            update_status = (
                f"Update: {update_count}   "
                f"pi {self.policy_losses[-1]:+.4f}   "
                f"v {self.value_losses[-1]:.4f}   "
                f"H {self.entropies[-1]:.3f}   "
                f"kl {latest_kl:.4f}   "
                f"clip {latest_clip:.2f}   "
                f"ev {latest_ev:+.2f}"
            )
            self.update_status_label.config(text=update_status)

            if self.stopped_earlys and self.stopped_earlys[-1]:
                warnings.append("last update stopped early (KL over target)")
            if latest_clip > 0.5:
                warnings.append(f"high clip fraction ({latest_clip:.2f})")
            if not math.isnan(latest_ev) and latest_ev < 0:
                warnings.append(f"critic worse than useless (ev {latest_ev:+.2f})")
            # value_loss's gradient reaches only the trunk and the critic
            # layer (never an action head, see ppo_training.py), so this -
            # not any per-head number - is what a value_loss spike shows up
            # as. Flagged relative to this run's own recent baseline rather
            # than a fixed number, since raw grad norm scales with reward
            # magnitude and network size.
            if len(self.grad_norm_trunk) >= 6:
                baseline = np.mean(self.grad_norm_trunk[-6:-1])
                if baseline > 1e-8 and self.grad_norm_trunk[-1] > 3 * baseline:
                    warnings.append(
                        f"trunk grad norm spiked ({self.grad_norm_trunk[-1]:.2f} "
                        f"vs recent avg {baseline:.2f}) - check value_loss, "
                        f"not the action heads"
                    )
            # Relative to the real ceiling: a flat 0.05 only caught a total
            # collapse of every head at once, and would stay silent while 4 of
            # 5 heads went fully deterministic.
            if self.entropies and self.entropies[-1] < 0.1 * self.max_entropy:
                warnings.append(
                    f"policy nearly deterministic (H {self.entropies[-1]:.2f} "
                    f"of max {self.max_entropy:.2f})"
                )

        self.warning_label.config(text=("  |  ".join(warnings)))

        self._render_summary()

        self.root.after(
            1000,
            self.update
        )

    def _render_summary(self):
        self.summary_ax.clear()
        self.summary_ax.axis("off")

        lines = [
            f"Episodes logged:  {len(self.rewards)}",
            f"Updates logged:   {len(self.policy_losses)}",
        ]
        if self.rewards:
            lines.append(f"Best episode reward:  {max(self.rewards):.2f}")
            window = min(AVG_WINDOW, len(self.rewards))
            lines.append(f"Recent avg reward:    {sum(self.rewards[-window:]) / window:.2f}")
        if self.stopped_earlys:
            early_rate = sum(self.stopped_earlys) / len(self.stopped_earlys)
            lines.append(f"Early-stop rate:      {early_rate:.1%}")

        if self.grad_norm_trunk:
            lines.append("")
            lines.append(f"grad norm  trunk={self.grad_norm_trunk[-1]:.3f}  "
                         f"critic={self.grad_norm_critic[-1]:.3f}")
            lines.append("(value_loss's gradient only ever reaches these two -")
            lines.append(" never an action head - so a spike shows up here)")

        if self.entropy_per_head:
            labels = (["move", "look", "atk", "dash", "jump"]
                      if self.num_heads == 5 else
                      [f"h{i}" for i in range(self.num_heads)])
            e = self.entropy_per_head[-1]
            k = self.approx_kl_per_head[-1]
            c = self.clipfrac_per_head[-1]
            lines.append("")
            lines.append("per-head (latest update):")
            # Row labels padded to the same width as the header indent so the
            # columns line up under their head names in the monospace block.
            lines.append("     " + "  ".join(f"{lb:>5}" for lb in labels))
            lines.append("H    " + "  ".join(f"{v:5.2f}" for v in e))
            lines.append("kl   " + "  ".join(f"{v:5.3f}" for v in k))
            lines.append("clip " + "  ".join(f"{v:5.2f}" for v in c))

        self.summary_ax.text(
            0.02, 0.98, "\n".join(lines),
            transform=self.summary_ax.transAxes,
            va="top", ha="left", fontsize=9, family="monospace",
        )

    def add_reward(self, reward, steps=None, hits=None, damage=None):
        self.rewards.append(reward)
        if steps is not None:
            self.steps.append(steps)
        if hits is not None:
            self.hits.append(hits)
        if damage is not None:
            self.damages.append(damage)
        self._save_history_throttled()

    def add_update(self, stats):
        """Feed the dict returned by ppo_training.ppo_update() straight in."""
        if stats is None:
            return
        self.policy_losses.append(stats["policy_loss"])
        self.value_losses.append(stats["value_loss"])
        self.entropies.append(stats["entropy"])
        self.approx_kls.append(stats["approx_kl"])
        self.clipfracs.append(stats["clipfrac"])
        self.explained_variances.append(stats["explained_variance"])
        self.stopped_earlys.append(bool(stats.get("stopped_early", False)))

        # .get() with a same-shaped default so a stats dict from a version of
        # ppo_update predating these keys does not crash the monitor - it
        # just logs zeros for that one update instead.
        self.entropy_per_head.append(stats.get("entropy_per_head", [0.0] * self.num_heads))
        self.approx_kl_per_head.append(stats.get("approx_kl_per_head", [0.0] * self.num_heads))
        self.clipfrac_per_head.append(stats.get("clipfrac_per_head", [0.0] * self.num_heads))
        self.grad_norm_per_head.append(stats.get("grad_norm_per_head", [0.0] * self.num_heads))
        self.grad_norm_trunk.append(stats.get("grad_norm_trunk", 0.0))
        self.grad_norm_critic.append(stats.get("grad_norm_critic", 0.0))

        self._save_history_throttled()

    def start(self):
        self.root.mainloop()
