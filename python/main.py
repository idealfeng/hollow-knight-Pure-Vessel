"""Train a PPO policy against the Gymnasium-wrapped Traitor Lord fight."""

import faulthandler
import os
import subprocess
import threading

faulthandler.enable(all_threads=True)

import numpy as np
import torch
import torch.optim as optim

from agent.action import BUTTON_ORDER, NUM_BUTTONS
from hollow_knight_env import HollowKnightEnv, DEFAULT_CONTROL_HZ, DEFAULT_FRAME_STACK
from monitor import TrainingMonitor
from obs_norm import RunningNorm
from ppo_model import ActorCritic
from ppo_training import ppo_update
from rollout_buffer import RolloutBuffer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "ppo_checkpoint.pth")
CHECKPOINT_FORMAT = 4

# Environment -----------------------------------------------------------
CONTROL_HZ = int(os.environ.get("HOLLOWRL_CONTROL_HZ", DEFAULT_CONTROL_HZ))
FRAME_STACK = int(os.environ.get("HOLLOWRL_FRAME_STACK", DEFAULT_FRAME_STACK))
EPISODE_TIME_LIMIT = 60.0
ACTION_SIZES = [1] * NUM_BUTTONS
MODEL_HIDDEN = 256

env = HollowKnightEnv(
    control_hz=CONTROL_HZ,
    frame_stack=FRAME_STACK,
    episode_time_limit=EPISODE_TIME_LIMIT,
)
STATE_SIZE = env.observation_space.shape[0]

# PPO -------------------------------------------------------------------
ROLLOUT_STEPS = 2048
TOTAL_UPDATES = 10000
EPOCHS = 4
MINIBATCH_SIZE = 128

# 0.9955 at 10 Hz has almost the same per-second discount as 0.997 at 15 Hz.
GAMMA = 0.9955
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 3e-4
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
TARGET_KL = 0.03
CLIP_VALUE_LOSS = True
VALUE_CLIP_EPS = 1.0
REWARD_SCALE = 0.05
SAVE_EVERY_UPDATES = 5

model = ActorCritic(STATE_SIZE, ACTION_SIZES, hidden=MODEL_HIDDEN,
                    bernoulli=True, exclusive_directions=True)
optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)
buffer = RolloutBuffer(ROLLOUT_STEPS, STATE_SIZE, num_action_heads=NUM_BUTTONS)
obs_norm = RunningNorm(STATE_SIZE)

start_update = 0
global_step = 0
episode = 0
resumed = os.path.exists(CHECKPOINT_PATH)


def _validate_checkpoint(checkpoint):
    expected = {
        "format_version": CHECKPOINT_FORMAT,
        "observation_size": STATE_SIZE,
        "frame_stack": FRAME_STACK,
        "action_names": list(BUTTON_ORDER),
        "action_distribution": "exclusive_directions_v1",
        "model_hidden": MODEL_HIDDEN,
    }
    mismatches = []
    for key, wanted in expected.items():
        got = checkpoint.get(key)
        if got != wanted:
            mismatches.append(f"{key}: checkpoint={got!r}, expected={wanted!r}")
    if mismatches:
        detail = "\n  ".join(mismatches)
        raise SystemExit(
            "ppo_checkpoint.pth is incompatible with this environment:\n  "
            + detail
            + "\nArchive/remove the checkpoint and start a fresh run."
        )


if resumed:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    _validate_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    obs_norm.load_state_dict(checkpoint["obs_norm"])
    start_update = checkpoint["update"] + 1
    global_step = checkpoint["global_step"]
    episode = checkpoint["episode"]
    print(f"resumed at update {start_update}, step {global_step}")

monitor = TrainingMonitor(
    target_kl=TARGET_KL,
    action_sizes=[3, 1, 3, 1, 2, 2, 2],
    action_labels=["left", "right", "up", "down",
                   "attack", "jump", "dash"],
)
monitor.set_run_info(resumed, start_update=start_update, start_episode=episode)
best_reward = max(monitor.rewards) if monitor.rewards else float("-inf")

REPLAY_CMD = os.environ.get("HOLLOWRL_REPLAY_CMD", "").strip()
_replay_warned = False


def observe(raw, update=True):
    if update:
        obs_norm.update(raw)
    return obs_norm.normalize(raw)


def save_replay(reason):
    global _replay_warned
    if not REPLAY_CMD:
        if not _replay_warned:
            print("replay capture disabled (set HOLLOWRL_REPLAY_CMD to enable)")
            _replay_warned = True
        return
    subprocess.Popen(
        REPLAY_CMD,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"saved replay - {reason}")


def save_checkpoint(update):
    torch.save({
        "format_version": CHECKPOINT_FORMAT,
        "observation_size": STATE_SIZE,
        "frame_stack": FRAME_STACK,
        "action_names": list(BUTTON_ORDER),
        "action_distribution": "exclusive_directions_v1",
        "action_sizes": ACTION_SIZES,
        "model_hidden": MODEL_HIDDEN,
        "control_hz": CONTROL_HZ,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "obs_norm": obs_norm.state_dict(),
        "update": update,
        "global_step": global_step,
        "episode": episode,
    }, CHECKPOINT_PATH)


def train_loop():
    global global_step, episode, best_reward

    try:
        raw_obs, _ = env.reset()

        episode_reward = 0.0
        episode_steps = 0

        for update in range(start_update, TOTAL_UPDATES):
            buffer.clear()

            for _ in range(ROLLOUT_STEPS):
                obs = observe(raw_obs)
                action, logprobs_per_head, _, value = model.act(obs)

                next_raw, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                episode_steps += 1
                global_step += 1

                buffer.add(
                    obs,
                    action,
                    logprobs_per_head,
                    value,
                    reward * REWARD_SCALE,
                    done,
                )
                raw_obs = next_raw

                if done:
                    monitor.add_reward(
                        episode_reward,
                        steps=episode_steps,
                        hits=info["hits"],
                        damage=info["damage"],
                        misses=info["misses"],
                        staggers=info["staggers"],
                        min_boss_hp=info["min_boss_health"],
                        won=info["won"],
                    )
                    episode += 1

                    if episode % 100 == 0:
                        save_replay(f"episode {episode}")
                    if episode_reward > best_reward:
                        best_reward = episode_reward
                        save_replay(
                            f"new best reward {best_reward:.2f} at episode {episode}"
                        )

                    episode_reward = 0.0
                    episode_steps = 0
                    raw_obs, _ = env.reset()

            # Do not update normalisation statistics a second time for the
            # bootstrap-only observation.
            last_value = model.value_only(observe(raw_obs, update=False))
            stats = ppo_update(
                model,
                optimizer,
                buffer,
                last_value,
                gamma=GAMMA,
                gae_lambda=GAE_LAMBDA,
                clip_eps=CLIP_EPS,
                epochs=EPOCHS,
                minibatch_size=MINIBATCH_SIZE,
                ent_coef=ENT_COEF,
                vf_coef=VF_COEF,
                max_grad_norm=MAX_GRAD_NORM,
                target_kl=TARGET_KL,
                clip_value_loss=CLIP_VALUE_LOSS,
                value_clip_eps=VALUE_CLIP_EPS,
            )

            if stats is not None:
                monitor.add_update(stats)
                print(
                    f"upd {update:5d} | step {global_step:8d} | ep {episode:4d} | "
                    f"pi {stats['policy_loss']:+.4f} | v {stats['value_loss']:.4f} | "
                    f"H {stats['entropy']:.3f} | kl {stats['approx_kl']:.4f} | "
                    f"clip {stats['clipfrac']:.2f} | ev {stats['explained_variance']:+.2f}"
                )

            if update % SAVE_EVERY_UPDATES == 0:
                save_checkpoint(update)
    finally:
        env.close()


if __name__ == "__main__":
    threading.Thread(target=train_loop, daemon=True).start()
    monitor.start()
