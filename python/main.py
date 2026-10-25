import faulthandler
import os
import subprocess
import time
import threading


faulthandler.enable(all_threads=True)

import numpy as np
import torch
import torch.optim as optim

import controls
import game_state
from enviroment import (
    OBSERVATION_SIZE, get_state, state_to_vector, reset_environment, is_done,
)
from agent import calculate_reward
from agent.action import (
    movement_actions, look_actions, attack_actions,
    dash_actions, jump_actions,
    movement, look,
)

from monitor import TrainingMonitor
from ppo_model import ActorCritic
from rollout_buffer import RolloutBuffer
from ppo_training import ppo_update
from obs_norm import RunningNorm


# ======================================================================
# OBSERVATION / ACTION SPACE
# Must match state_to_vector()'s output width and agent/action.py's groups.
STATE_SIZE = OBSERVATION_SIZE

ACTION_GROUPS = [movement_actions, look_actions, attack_actions, dash_actions, jump_actions]
ACTION_SIZES = [len(g) for g in ACTION_GROUPS]   # [6, 4, 2, 2, 3]


# ======================================================================
# CONTROL RATE
CONTROL_HZ = 15
STEP_DT = 1.0 / CONTROL_HZ


# ======================================================================
# EPISODE LENGTH
# Cap every episode at 60 s of game time - past that we force-kill the
# knight (Y hotkey, see connection.cs) instead of letting it run indefinitely.
EPISODE_TIME_LIMIT = 60.0

# ======================================================================
# ROLLOUT / UPDATE
ROLLOUT_STEPS = 2048         # ~136 s of play at 15 Hz, ~2 episodes at the 60 s cap
TOTAL_UPDATES = 10000
EPOCHS = 4                   # passes over each rollout
MINIBATCH_SIZE = 128         # -> 4 minibatches x 4 epochs = 16 grad steps

# ======================================================================
# PPO CORE
# Effective horizon is 1/(1-GAMMA) steps. At CONTROL_HZ=15 an episode is 900
# steps, so the classic 0.99 sees only 100 steps = 6.7 s of a 60 s fight -
# from the episode start a +100 boss-defeat bonus discounts to 0.012, i.e.
# the terminal rewards may as well not exist. 0.997 gives a 333-step / 22 s
# horizon, which actually reaches back over a meaningful part of the fight.
GAMMA = 0.997
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 3e-4                    # PPO's classic LR; higher than DQN's 1e-4
ENT_COEF = 0.01              # exploration knob (replaces epsilon decay)
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

# KL is ADDITIVE over independent action heads, and approx_kl is measured on
# the summed joint logprob. The usual 0.01-0.03 assumes ONE head; across our
# five that is a per-head budget of 0.006 and stops nearly every update early.
# 0.1 keeps roughly the conventional per-head budget (0.1/5 = 0.02).
TARGET_KL = 0.1

# CLIP_VALUE_LOSS was False (fully unclipped) after the previous session's
# fix for a critic that couldn't reach the right return magnitude under the
# old shared-epsilon clipping. The side effect showed up as periodic
# value_loss spikes (~0.008 baseline -> ~0.05) with explained_variance
# dipping to +0.24-0.5 at the same updates: unclipped, one outlier-return
# minibatch (a big stagger/defeat, or right after the reward function
# changed) produces a huge squared-error gradient with nothing bounding it,
# and that gradient runs through the SHARED TRUNK - it cannot be isolated to
# the critic, and it visibly perturbed entropy at the same updates. A value
# clip range decoupled from the policy's clip_eps keeps a ceiling on any
# single update's swing without reintroducing the "too tight to reach
# magnitude" problem: 1.0 is 5x wider than the old shared 0.2.
CLIP_VALUE_LOSS = True
VALUE_CLIP_EPS = 1.0


# ======================================================================
# REWARDS
REWARD_SCALE = 0.05
CHECKPOINT_PATH = "ppo_checkpoint.pth"
SAVE_EVERY_UPDATES = 5



# ======================================================================
# SETUP

model = ActorCritic(STATE_SIZE, ACTION_SIZES)
optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)
buffer = RolloutBuffer(ROLLOUT_STEPS, STATE_SIZE, num_action_heads=len(ACTION_SIZES))
obs_norm = RunningNorm(STATE_SIZE)

# ======================================================================
start_update = 0
global_step = 0
episode = 0
resumed = os.path.exists(CHECKPOINT_PATH)

if resumed:
    ckpt = torch.load(CHECKPOINT_PATH, weights_only=False)
    try:
        model.load_state_dict(ckpt["model"])
    except RuntimeError as exc:
        raise SystemExit(
            f"{CHECKPOINT_PATH} does not match the current model architecture "
            f"(action heads: {ACTION_SIZES}). It was likely saved before the "
            f"multi-head action refactor. Delete or rename the checkpoint "
            f"(and monitor_history.npz, since it's a different training run) "
            f"and start fresh.\nOriginal error: {exc}"
        )
    optimizer.load_state_dict(ckpt["optimizer"])
    obs_norm.load_state_dict(ckpt["obs_norm"])
    start_update = ckpt["update"] + 1
    global_step = ckpt["global_step"]
    episode = ckpt["episode"]
    print(f"resumed at update {start_update}, step {global_step}")

# Built after the resume check above (not before) so its header can show
# whether this run is fresh or resumed, and from where - previously that
# information only ever went to stdout, not somewhere you could see it while
# already looking at the monitor window.
monitor = TrainingMonitor(target_kl=TARGET_KL, action_sizes=ACTION_SIZES)
monitor.set_run_info(resumed, start_update=start_update, start_episode=episode)

# Seeded from whatever history the monitor loaded (not from the checkpoint -
# the checkpoint has no notion of "best reward"), so resuming a run does not
# treat every episode as a new record just because best_reward restarted at
# -inf.
best_reward = max(monitor.rewards) if monitor.rewards else float("-inf")



# ======================================================================
# HELPERS


def episode_time_frac(steps):
    """How far into its time limit the current episode is, 0.0 -> 1.0.
    Derived from episode_steps (game-time, like the cap itself) rather than
    wall-clock, so it stays correct under the in-game 2x speed toggle.
    Clamped so an over-running episode can't push the feature outside the
    range obs_norm has calibrated on."""
    return min(steps * STEP_DT / EPISODE_TIME_LIMIT, 1.0)


def read_observation(time_frac=0.0, retries=40, delay=0.02):

    for _ in range(retries):
        state = get_state()
        try:
            vec = state_to_vector(state, time_frac)
        except (IndexError, TypeError, ValueError):
            vec = None
        if vec is not None:
            return state, np.asarray(vec, dtype=np.float64)
        time.sleep(delay)
    return None, None


def observe(raw):
    """Update normaliser stats, then normalise. Returns the network input."""
    obs_norm.update(raw)
    return obs_norm.normalize(raw)


def safe_game_speed():
    """game_state.game_speed mirrors Unity's Time.timeScale, which commonly
    drops to 0 when the game window loses OS focus (or during menus/loading).
    Used as a divisor, 0 would turn a 1/15s sleep into ~66,700 seconds with
    no error - so anything not clearly a real "sped up" value falls back to
    normal speed instead."""
    speed = game_state.game_speed
    return speed if speed > 0.1 else 1.0


def stop_all():
    """Release every held key across the action groups - movement (left/right)
    and look (up/down). Attack/dash/jump are one-shot presses and don't need
    releasing."""
    movement.stop()
    look.stop_looking()


def wait_while_paused():
    """Blocks (releasing held keys) while the in-game P hotkey has paused
    training. Returns True if it actually waited, so the caller knows to
    refresh its observation before acting on it again."""
    if not game_state.training_paused:
        return False

    stop_all()
    while game_state.training_paused:
        time.sleep(0.1)
    return True


# Replay capture used to be pyautogui.hotkey("alt", "f10") - NVIDIA Instant
# Replay's Windows hotkey, which does not exist here and, being a global
# keystroke, was exactly the kind of desktop-wide input we moved away from.
# There is no portable Linux equivalent, so this is a command you configure:
#
#   OBS (replay buffer hotkey via obs-cli):
#     export HOLLOWRL_REPLAY_CMD='obs-cli replaybuffer save'
#   gpu-screen-recorder (replay mode):
#     export HOLLOWRL_REPLAY_CMD='pkill -SIGUSR1 -x gpu-screen-recorder'
#
# Unset, replays are simply skipped - training does not depend on them.
REPLAY_CMD = os.environ.get("HOLLOWRL_REPLAY_CMD")
_replay_warned = False


def save_replay(reason):
    global _replay_warned

    if not REPLAY_CMD:
        if not _replay_warned:
            print("replay capture disabled (set HOLLOWRL_REPLAY_CMD to enable)")
            _replay_warned = True
        return

    # Fire and forget: a recorder that hangs must not stall the rollout.
    subprocess.Popen(REPLAY_CMD, shell=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"saved replay - {reason}")


def save_checkpoint(update):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "obs_norm": obs_norm.state_dict(),
        "update": update,
        "global_step": global_step,
        "episode": episode,
    }, CHECKPOINT_PATH)



# ======================================================================
# TRAIN

def train_loop():
    global global_step, episode, best_reward

    state, raw_obs = read_observation(0.0)
    if raw_obs is None:
        print("could not read game state - is the game running?")
        return

    reset_environment(state)
    state, raw_obs = read_observation(0.0)

    episode_reward = 0.0
    episode_steps = 0
    last_state = state

    for update in range(start_update, TOTAL_UPDATES):

        buffer.clear()

        # ---------- PHASE 1: COLLECT (no learning happens here) ----------
        for _ in range(ROLLOUT_STEPS):

            if wait_while_paused():
                # Stale by however long we were paused - re-read before acting on it.
                state, raw_obs = read_observation(episode_time_frac(episode_steps))
                if raw_obs is None:
                    print("lost the game state while paused - stopping")
                    return

            obs = observe(raw_obs)

            # Sampled from the policy - this IS the exploration.
            # logprobs_per_head (not just the joint logprob) is what the
            # buffer stores now, so ppo_update can compute a real per-head
            # PPO ratio later - see rollout_buffer.py / ppo_training.py.
            action, logprobs_per_head, logprob, value = model.act(obs)
            for group, a in zip(ACTION_GROUPS, action):
                group[a]()

            # Episode time limit: episode_steps * STEP_DT is intended
            # game-time elapsed this episode (not wall-clock, same convention
            # as STEP_DT itself). Past the cap, force a death via the Y
            # hotkey instead of letting the episode run on indefinitely -
            # is_done() picks up knight_health == 0 next read and the usual
            # reset flow takes over. `timed_out` stays true for the remaining
            # steps until reset, so the death it causes is still recognised as
            # scheduled however many frames the death sequence takes to land.
            timed_out = episode_steps * STEP_DT >= EPISODE_TIME_LIMIT
            if timed_out:
                controls.kill()

            # STEP_DT is intended game-time; at 2x game_speed, half the real
            # sleep produces the same amount of game-time per step.
            time.sleep(STEP_DT / safe_game_speed())

            next_state, next_raw = read_observation(episode_time_frac(episode_steps + 1))
            done = is_done()

            if next_raw is None:
                # A Pure Vessel victory can remove the HeroController during
                # the same transition that removes the boss. The mod latches
                # boss_health=0 even though no vector can be built without a
                # knight box; synthesize only that terminal field from the
                # last valid state so the final hit/win reward is not lost.
                done = True
                if game_state.boss_health == 0:
                    next_state = dict(state)
                    next_state["boss_health"] = 0
                    next_state["boss_facing_left"] = None
                    next_state["boss_control_state"] = ""
                    next_raw = raw_obs
                else:
                    # A genuine disconnect/transition is a truncation: cut
                    # the advantage chain instead of joining unrelated states.
                    next_state, next_raw = state, raw_obs

            reward = calculate_reward(
                state, next_state, done, timed_out=timed_out, step_dt=STEP_DT,
            )

            episode_reward += reward          # log the RAW reward
            episode_steps += 1
            global_step += 1

            buffer.add(obs, action, logprobs_per_head, value,
                       reward * REWARD_SCALE,  # train on the SCALED reward
                       done)

            state, raw_obs = next_state, next_raw
            last_state = next_state

            if done:
                monitor.add_reward(
                    episode_reward,
                    steps=episode_steps,
                    hits=last_state["hit_count"] if last_state else 0,
                    damage=last_state["damage_count"] if last_state else 0,
                )
                episode += 1

                # Save a replay every 100 episodes
                if episode % 100 == 0:
                    save_replay(f"episode {episode}")

                # New best episode reward - save a replay for it too, on top
                # of the every-100-episodes one above.
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    save_replay(f"new best reward {best_reward:.2f} at episode {episode}")

                episode_reward = 0.0
                episode_steps = 0

                stop_all()
                reset_environment(state)
                # Fresh episode - clock back to 0.
                state, raw_obs = read_observation(0.0)
                if raw_obs is None:
                    print("lost the game state after reset - stopping")
                    return

        # ---------- PHASE 2: LEARN ----------
        # Release the keys first. The game keeps running while we do gradient
        # descent, and we do not want the knight sprinting into a wall for the
        # ~100 ms this takes.
        stop_all()

        # Bootstrap value for the state we stopped on. If that step ended the
        # episode, compute_gae multiplies this by nonterminal=0 and ignores it.
        last_value = model.value_only(observe(raw_obs))

        stats = ppo_update(
            model, optimizer, buffer, last_value,
            gamma=GAMMA, gae_lambda=GAE_LAMBDA, clip_eps=CLIP_EPS,
            epochs=EPOCHS, minibatch_size=MINIBATCH_SIZE,
            ent_coef=ENT_COEF, vf_coef=VF_COEF,
            max_grad_norm=MAX_GRAD_NORM, target_kl=TARGET_KL,
            clip_value_loss=CLIP_VALUE_LOSS, value_clip_eps=VALUE_CLIP_EPS,
        )

        if stats is not None:
            monitor.add_update(stats)
            print(
                f"upd {update:5d} | step {global_step:8d} | ep {episode:4d} | "
                f"pi {stats['policy_loss']:+.4f} | v {stats['value_loss']:.4f} | "
                f"H {stats['entropy']:.3f} | kl {stats['approx_kl']:.4f} | "
                f"clip {stats['clipfrac']:.2f} | ev {stats['explained_variance']:+.2f}"
            )

        # The freshly-read state is now off-policy by one update, but it is a
        # single step out of ROLLOUT_STEPS - harmless.

        if update % SAVE_EVERY_UPDATES == 0:
            save_checkpoint(update)


if __name__ == "__main__":
    threading.Thread(target=train_loop, daemon=True).start()
    monitor.start()
