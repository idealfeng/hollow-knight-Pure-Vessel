"""Central PPO learner for multiple live Hollow Knight game processes."""

import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import json
import math
import multiprocessing as mp
from multiprocessing.connection import wait
import os
import socket
import threading
import time
import traceback

import numpy as np
import torch
import torch.optim as optim

from agent.action import BUTTON_ORDER, NUM_BUTTONS, is_intro_state
from env_worker import env_worker_main
from monitor import TrainingMonitor
from obs_norm import RunningNorm
from ppo_model import ActorCritic
from ppo_training import ppo_update
from rollout_buffer import VectorRolloutBuffer


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FORMAT = 8
MODEL_HIDDEN = 256
ACTION_SIZES = [1] * NUM_BUTTONS

GAMMA = 0.9955
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 3e-4
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
TARGET_KL = 0.03
VALUE_CLIP_EPS = 1.0
REWARD_SCALE = 0.05
EPOCHS = 4
MINIBATCH_SIZE = 128
SAVE_EVERY_UPDATES = 1


class ExternalEnvPool:
    """Small IPC pool whose children each own one socket-backed environment."""

    def __init__(self, ports, control_hz, frame_stack, episode_time_limit,
                 game_speed):
        context = mp.get_context("spawn")
        self.ports = tuple(int(port) for port in ports)
        self.connections = []
        self.processes = []
        self.index_by_connection = {}

        for index, port in enumerate(self.ports):
            parent, child = context.Pipe()
            process = context.Process(
                target=env_worker_main,
                args=(child, port, control_hz, frame_stack, episode_time_limit,
                      game_speed),
                name=f"HollowRL-env-{port}",
                daemon=True,
            )
            # On Windows the spawned interpreter imports main_multi before it
            # calls env_worker_main. Modules imported at that stage must see
            # the worker's own port, not the default 9000.
            previous_port = os.environ.get("HOLLOWRL_PORT")
            os.environ["HOLLOWRL_PORT"] = str(port)
            try:
                process.start()
            finally:
                if previous_port is None:
                    os.environ.pop("HOLLOWRL_PORT", None)
                else:
                    os.environ["HOLLOWRL_PORT"] = previous_port
            child.close()
            self.connections.append(parent)
            self.processes.append(process)
            self.index_by_connection[parent] = index

        for index in range(len(self.ports)):
            message = self.recv(index)
            if (message.get("event") != "ready"
                    or message.get("control_port") != self.ports[index]
                    or message.get("state_port") != self.ports[index]):
                raise RuntimeError(
                    f"worker {self.ports[index]} bound incorrect ports: {message}"
                )

    def send(self, index, command, payload=None):
        if payload is None:
            self.connections[index].send((command,))
        else:
            self.connections[index].send((command, payload))

    def recv(self, index):
        ok, payload = self.connections[index].recv()
        if not ok:
            raise RuntimeError(
                f"environment on port {self.ports[index]} failed:\n{payload}"
            )
        return payload

    def ready(self, pending, timeout=1.0):
        ready_connections = wait(list(pending), timeout=timeout)
        if not ready_connections:
            dead = [
                self.ports[i] for i, process in enumerate(self.processes)
                if not process.is_alive()
            ]
            if dead:
                raise RuntimeError(f"environment workers exited: {dead}")
        return ready_connections

    def close(self):
        for index, process in enumerate(self.processes):
            if process.is_alive():
                try:
                    self.send(index, "close")
                except (BrokenPipeError, EOFError, OSError):
                    pass
        for connection in self.connections:
            connection.close()
        for process in self.processes:
            process.join(timeout=3.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)


def default_steps_per_env(num_envs):
    # Preserve roughly the old 2048-transition update size while keeping the
    # per-environment horizon aligned to the 128-sample minibatch size.
    return max(512, int(math.ceil(2048 / num_envs / 128.0) * 128))


def prepare_fight(port):
    """Use the training save's Traitor Lord statue to start a fresh fight."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=4.0) as sock:
            sock.settimeout(4.0)
            with sock.makefile("rwb", buffering=0) as stream:
                def command(value):
                    stream.write(value.encode("ascii") + b"\n")
                    answer = stream.readline(65536).decode("utf-8").strip()
                    if not answer or answer.startswith("err "):
                        raise RuntimeError(f"Mod command {value!r} failed: {answer!r}")
                    return answer

                def snapshot():
                    parts = command("pos").split(";")
                    scalars = parts[4].split(",")
                    box = tuple(float(v) for v in parts[0].split(",")) if parts[0] else None
                    return (box, int(scalars[3]), int(scalars[7]),
                            float(scalars[9]), parts[6], bool(parts[7]))

                box, boss_hp, knight_hp, speed, fsm, has_boss_box = snapshot()
                if boss_hp <= 0:
                    if box is None or knight_hp <= 0:
                        raise RuntimeError(
                            f"port {port}: no living knight (boss_hp={boss_hp}, "
                            f"knight_hp={knight_hp}). Load the training save first."
                        )
                    center_x = (box[0] + box[2]) / 2
                    center_y = (box[1] + box[3]) / 2
                    if not (184 <= center_x <= 192 and 4 <= center_y <= 9):
                        raise RuntimeError(
                            f"port {port}: boss is absent and knight is at "
                            f"({center_x:.1f}, {center_y:.1f}), outside the "
                            "known Traitor Lord statue position. Stand at the "
                            "statue with the training save."
                        )
                    print(f"port {port}: entering Traitor Lord fight from statue", flush=True)
                    for attempt in range(1, 4):
                        command("TAP up 120")
                        time.sleep(1.2)
                        command("TAP submit 120")
                        deadline = time.perf_counter() + 6.0
                        while time.perf_counter() < deadline:
                            box, boss_hp, knight_hp, speed, fsm, has_boss_box = snapshot()
                            if box is not None and boss_hp > 0 and knight_hp > 0:
                                break
                            time.sleep(0.15)
                        if box is not None and boss_hp > 0 and knight_hp > 0:
                            break
                        print(
                            f"port {port}: statue attempt {attempt}/3 did not "
                            "start a fight; retrying",
                            flush=True,
                        )
                    else:
                        raise RuntimeError(
                            f"port {port}: statue input did not start a live fight "
                            f"(boss_hp={boss_hp}, knight_hp={knight_hp})."
                        )

                # The HP object appears long before the intro ends. Keep the
                # game moving without policy actions until its first real
                # combat state, then freeze for the workers' setup.
                if is_intro_state(fsm) and speed == 0.0:
                    command("SPEED 1")
                ready_deadline = time.perf_counter() + 25.0
                while time.perf_counter() < ready_deadline:
                    box, boss_hp, knight_hp, speed, fsm, has_boss_box = snapshot()
                    if box is None or boss_hp <= 0 or knight_hp <= 0:
                        raise RuntimeError(
                            f"port {port}: fight ended during boss intro "
                            f"(boss_hp={boss_hp}, knight_hp={knight_hp})."
                        )
                    if fsm and not is_intro_state(fsm) and has_boss_box:
                        break
                    time.sleep(0.05)
                else:
                    raise RuntimeError(
                        f"port {port}: boss intro did not reach combat "
                        f"(fsm={fsm!r}, boss_box={has_boss_box}, speed={speed})."
                    )

                freeze_deadline = time.perf_counter() + 4.0
                while time.perf_counter() < freeze_deadline:
                    command("SPEED 0")
                    time.sleep(0.15)
                    box, boss_hp, knight_hp, speed, fsm, has_boss_box = snapshot()
                    if boss_hp > 0 and knight_hp > 0 and speed == 0.0:
                        break
                else:
                    raise RuntimeError(
                        f"port {port}: could not freeze the live fight "
                        f"(boss_hp={boss_hp}, knight_hp={knight_hp}, speed={speed}). "
                        "Install the rebuilt HollowRL.dll in both games."
                    )
                print(
                    f"port {port}: fight ready, boss_hp={boss_hp}, "
                    f"knight_hp={knight_hp}; frozen during setup",
                    flush=True,
                )
    except (OSError, IndexError, ValueError, UnicodeError) as exc:
        raise RuntimeError(f"port {port}: cannot prepare fight ({exc})") from exc


def prepare_fights(ports):
    # Check every game before pressing anything. If one window is still at
    # the title screen, leave any already staged fight frozen and intact.
    unloaded = []
    for port in ports:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=4.0) as sock:
                sock.settimeout(4.0)
                sock.sendall(b"pos\n")
                with sock.makefile("rb") as reply:
                    line = reply.readline(65536).decode("utf-8").strip()
            parts = line.split(";")
            scalars = parts[4].split(",")
            if not parts[0] or int(scalars[7]) <= 0:
                unloaded.append(f"port {port}: no living knight; load save slot 1")
        except (OSError, IndexError, ValueError, UnicodeError) as exc:
            unloaded.append(f"port {port}: cannot read the game state ({exc})")
    if unloaded:
        raise SystemExit("Prepare both game windows first:\n" + "\n".join(unloaded))

    # Start all statues concurrently so neither idle knight must wait for
    # another game's menu sequence before the learner can begin.
    with ThreadPoolExecutor(max_workers=len(ports)) as executor:
        futures = {port: executor.submit(prepare_fight, port) for port in ports}
        problems = []
        for port, future in futures.items():
            try:
                future.result()
            except Exception as exc:
                problems.append(str(exc))
    if problems:
        raise SystemExit(
            "Could not prepare every Traitor Lord fight:\n"
            + "\n".join(problems)
            + "\nAny ready fight remains frozen; fix the other window and rerun."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one PPO policy from multiple Hollow Knight instances."
    )
    parser.add_argument("--ports", type=int, nargs="+", default=[9001, 9002])
    parser.add_argument("--control-hz", type=int, default=10)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--episode-time-limit", type=float, default=60.0)
    parser.add_argument("--game-speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--steps-per-env", type=int, default=0,
                        help="0 chooses about 2048 total transitions/update")
    parser.add_argument("--total-updates", type=int, default=10000)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--init-checkpoint", default="",
                        help="start from this checkpoint when the run dir has no ppo_checkpoint.pth yet (e.g. a BC bc_initial_checkpoint.pth)")
    parser.add_argument("--lr", type=float, default=LR,
                        help=f"Adam learning rate (default {LR:g})")
    parser.add_argument("--ent-coef", type=float, default=ENT_COEF,
                        help=f"entropy bonus coefficient (default {ENT_COEF:g})")
    parser.add_argument("--eval-every", type=int, default=0,
                        help="run fixed greedy evaluation every N updates; 0 disables")
    parser.add_argument("--eval-episodes", type=int, default=1,
                        help="greedy episodes per environment during each evaluation")
    parser.add_argument("--freeze-obs-norm", action="store_true",
                        help="keep the loaded observation normalizer fixed during this run")
    parser.add_argument("--stop-below-hits", type=float, default=0.0,
                        help="stop after an update if the last 20 complete fights average fewer hits; 0 disables")
    parser.add_argument("--no-monitor", action="store_true",
                        help="run in the terminal without the Tk dashboard")
    args = parser.parse_args()

    if len(set(args.ports)) != len(args.ports):
        parser.error("ports must be unique")
    if any(port < 1024 or port > 65535 for port in args.ports):
        parser.error("ports must be in 1024..65535")
    if args.control_hz <= 0 or args.frame_stack <= 0:
        parser.error("control-hz and frame-stack must be positive")
    if not 0.5 <= args.game_speed <= 4.0:
        parser.error("game-speed must be in 0.5..4.0")
    if args.steps_per_env < 0:
        parser.error("steps-per-env must be 0 or positive")
    if args.torch_threads < 1:
        parser.error("torch-threads must be positive")
    if not math.isfinite(args.stop_below_hits) or args.stop_below_hits < 0:
        parser.error("stop-below-hits must be finite and nonnegative")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr must be finite and positive")
    if not math.isfinite(args.ent_coef) or args.ent_coef < 0:
        parser.error("ent-coef must be finite and nonnegative")
    if args.eval_every < 0:
        parser.error("eval-every must be 0 or positive")
    if args.eval_episodes < 1:
        parser.error("eval-episodes must be positive")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    num_envs = len(args.ports)
    steps_per_env = args.steps_per_env or default_steps_per_env(num_envs)
    # Boss relative position now uses the body-box centre rather than the
    # HealthManager root. Keep those observation semantics in a new run.
    run_name = "traitor_exclusive_v2_" + "_".join(str(port) for port in args.ports)
    run_dir = os.path.abspath(args.run_dir or os.path.join(SCRIPT_DIR, "runs", run_name))
    os.makedirs(run_dir, exist_ok=True)
    checkpoint_path = os.path.join(run_dir, "ppo_checkpoint.pth")
    history_path = os.path.join(run_dir, "monitor_history.npz")
    metrics_path = os.path.join(run_dir, "training_metrics.jsonl")
    eval_metrics_path = os.path.join(run_dir, "eval_metrics.jsonl")

    print(
        f"starting {num_envs} environments on ports {args.ports}; "
        f"{steps_per_env} steps/env = {steps_per_env * num_envs} transitions/update"
    )
    print(f"run artifacts: {run_dir}")

    prepare_fights(args.ports)

    pool = ExternalEnvPool(
        args.ports, args.control_hz, args.frame_stack,
        args.episode_time_limit, args.game_speed
    )

    # Reset concurrently.  Waiting for replies in index order does not make
    # the work sequential: all child processes are already navigating their
    # own menus at this point.
    try:
        for i in range(num_envs):
            pool.send(i, "reset")
        raw_observations = [None] * num_envs
        initial_infos = [None] * num_envs
        for i in range(num_envs):
            raw_observations[i], initial_infos[i] = pool.recv(i)
    except Exception:
        pool.close()
        raise

    state_size = int(np.asarray(raw_observations[0]).shape[0])
    if any(np.asarray(obs).shape != (state_size,) for obs in raw_observations):
        pool.close()
        raise RuntimeError("environment observation shapes do not match")

    model = ActorCritic(state_size, ACTION_SIZES, hidden=MODEL_HIDDEN,
                        bernoulli=True, exclusive_directions=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    obs_norm = RunningNorm(state_size)
    buffer = VectorRolloutBuffer(
        steps_per_env, num_envs, state_size, num_action_heads=NUM_BUTTONS
    )

    start_update = 0
    global_step = 0
    episode = 0
    resumed = os.path.exists(checkpoint_path)
    freeze_obs_norm = args.freeze_obs_norm
    pretraining = None

    init_source = None
    if resumed:
        init_source = checkpoint_path
    elif args.init_checkpoint:
        init_source = os.path.abspath(args.init_checkpoint)
        if not os.path.isfile(init_source):
            pool.close()
            raise SystemExit(f"--init-checkpoint not found: {init_source}")
        print(f"starting from initial checkpoint: {init_source}", flush=True)

    if init_source is not None:
        checkpoint = torch.load(init_source, map_location="cpu", weights_only=False)
        expected = {
            "format_version": CHECKPOINT_FORMAT,
            "observation_size": state_size,
            "frame_stack": args.frame_stack,
            "action_names": list(BUTTON_ORDER),
            "model_hidden": MODEL_HIDDEN,
            "num_envs": num_envs,
            "action_distribution": "exclusive_directions_v1",
        }
        mismatches = [
            f"{key}: checkpoint={checkpoint.get(key)!r}, expected={wanted!r}"
            for key, wanted in expected.items() if checkpoint.get(key) != wanted
        ]
        if mismatches:
            pool.close()
            raise SystemExit(
                "multi-environment checkpoint is incompatible:\n  "
                + "\n  ".join(mismatches)
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        obs_norm.load_state_dict(checkpoint["obs_norm"])
        freeze_obs_norm = freeze_obs_norm or checkpoint.get("freeze_obs_norm", False)
        pretraining = checkpoint.get("pretraining")
        start_update = int(checkpoint["update"]) + 1
        global_step = int(checkpoint["global_step"])
        episode = int(checkpoint["episode"])
        print(f"resumed at update {start_update}, step {global_step}")

    # The saved optimizer may hold a different learning rate (the BC checkpoint
    # stores the module default). Honour --lr explicitly for this run.
    for group in optimizer.param_groups:
        group["lr"] = args.lr

    if freeze_obs_norm and init_source is None:
        pool.close()
        raise SystemExit("--freeze-obs-norm requires a checkpoint with fitted statistics")
    if freeze_obs_norm:
        print("observation normalization: fixed checkpoint statistics")

    monitor = None
    if not args.no_monitor:
        try:
            monitor = TrainingMonitor(
                target_kl=TARGET_KL,
                action_sizes=[3, 1, 3, 1, 2, 2, 2],
                action_labels=["left", "right", "up", "down",
                               "attack", "jump", "dash"],
                history_path=history_path,
            )
        except Exception as exc:
            import tkinter
            if not isinstance(exc, tkinter.TclError):
                pool.close()
                raise
            reason = str(exc).splitlines()[0]
            print(f"monitor unavailable ({reason}); continuing in terminal mode", flush=True)
    if monitor is not None:
        monitor.set_run_info(resumed, start_update=start_update, start_episode=episode)

    episode_rewards = [0.0] * num_envs
    episode_steps = [0] * num_envs
    episode_actions = [Counter() for _ in range(num_envs)]
    episode_reward_parts = [Counter() for _ in range(num_envs)]
    recent_hits = deque(maxlen=20)
    needs_reset = [False] * num_envs

    def log_metrics(record):
        with open(metrics_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, default=float) + "\n")

    def observe(raw, update=True):
        if update and not freeze_obs_norm:
            obs_norm.update(raw)
        return obs_norm.normalize(raw)

    def run_greedy_eval(update):
        """Deterministic progress check: play each env with greedy_act, no
        buffer writes, no obs_norm updates and no gradient step. Results are
        appended to eval_metrics.jsonl so progress can be judged on a fixed
        policy rather than exploratory training rollouts."""
        model.eval()
        try:
            for env_index in range(num_envs):
                for ep_index in range(args.eval_episodes):
                    pool.send(env_index, "reset")
                    obs, _ = pool.recv(env_index)
                    pool.send(env_index, "start")
                    pool.recv(env_index)
                    episode_reward = 0.0
                    episode_steps = 0
                    done = False
                    info = {}
                    while not done:
                        action = model.greedy_act(observe(obs, update=False))
                        pool.send(env_index, "step", action)
                        obs, reward, terminated, truncated, info = pool.recv(env_index)
                        episode_reward += reward
                        episode_steps += 1
                        done = bool(terminated or truncated)
                    record = {
                        "event": "greedy_eval",
                        "update": update,
                        "global_step": global_step,
                        "env": env_index,
                        "port": args.ports[env_index],
                        "episode_in_eval": ep_index + 1,
                        "reward": episode_reward,
                        "steps": episode_steps,
                        "hits": info.get("hits"),
                        "damage": info.get("damage"),
                        "misses": info.get("misses"),
                        "staggers": info.get("staggers"),
                        "min_boss_health": info.get("min_boss_health"),
                        "won": info.get("won"),
                    }
                    with open(eval_metrics_path, "a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, default=float) + "\n")
                    print(
                        f"greedy eval upd {update} | env {env_index} "
                        f"port {args.ports[env_index]} | ep {ep_index + 1} | "
                        f"reward {episode_reward:+.2f} | hits {info.get('hits')} | "
                        f"damage {info.get('damage')} | "
                        f"min_boss_hp {info.get('min_boss_health')} | "
                        f"won {bool(info.get('won'))}",
                        flush=True,
                    )
            for env_index in range(num_envs):
                needs_reset[env_index] = True
        finally:
            model.train()

    def add_episode(info, env_index):
        nonlocal episode
        if monitor is not None:
            monitor.add_reward(
                episode_rewards[env_index],
                steps=episode_steps[env_index],
                hits=info["hits"],
                damage=info["damage"],
                misses=info["misses"],
                staggers=info["staggers"],
                min_boss_hp=info["min_boss_health"],
                won=info["won"],
            )
        episode += 1
        recent_hits.append(info["hits"])
        log_metrics({
            **info,
            "event": "episode", "episode": episode,
            "global_step": global_step, "env": env_index,
            "reward": episode_rewards[env_index], "steps": episode_steps[env_index],
            "actions": {name: episode_actions[env_index][name] for name in BUTTON_ORDER},
            "reward_parts": dict(episode_reward_parts[env_index]),
        })
        print(
            f"ep {episode:5d} | env {env_index} port {args.ports[env_index]} | "
            f"reward {episode_rewards[env_index]:+.2f} | "
            f"steps {episode_steps[env_index]} | hits {info['hits']} | "
            f"damage {info['damage']} | misses {info['misses']} | "
            f"won {bool(info['won'])}"
        )
        episode_rewards[env_index] = 0.0
        episode_steps[env_index] = 0
        episode_actions[env_index].clear()
        episode_reward_parts[env_index].clear()

    def save_checkpoint(update):
        checkpoint = {
            "format_version": CHECKPOINT_FORMAT,
            "observation_size": state_size,
            "frame_stack": args.frame_stack,
            "action_names": list(BUTTON_ORDER),
            "action_sizes": ACTION_SIZES,
            "model_hidden": MODEL_HIDDEN,
            "num_envs": num_envs,
            "action_distribution": "exclusive_directions_v1",
            "ports": list(args.ports),
            "steps_per_env": steps_per_env,
            "control_hz": args.control_hz,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "obs_norm": obs_norm.state_dict(),
            "freeze_obs_norm": freeze_obs_norm,
            "pretraining": pretraining,
            "update": update,
            "global_step": global_step,
            "episode": episode,
        }
        temporary_path = checkpoint_path + ".tmp"
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, checkpoint_path)

    def train_loop():
        nonlocal global_step
        try:
            for update in range(start_update, args.total_updates):
                update_started = time.perf_counter()
                buffer.clear()
                counts = [0] * num_envs
                pending = {}
                update_reset_seconds = 0.0
                update_reset_retries = 0

                for env_index in range(num_envs):
                    if not needs_reset[env_index]:
                        pool.send(env_index, "start")
                for env_index in range(num_envs):
                    if not needs_reset[env_index]:
                        pool.recv(env_index)

                def dispatch_action(env_index):
                    obs = observe(raw_observations[env_index])
                    action, logprobs, _, value = model.act(obs)
                    pool.send(env_index, "step", action)
                    pending[pool.connections[env_index]] = {
                        "kind": "step",
                        "env": env_index,
                        "obs": obs,
                        "action": action,
                        "logprobs": logprobs,
                        "value": value,
                    }

                def dispatch_reset(env_index):
                    pool.send(env_index, "reset")
                    pending[pool.connections[env_index]] = {
                        "kind": "reset", "env": env_index,
                    }

                for env_index in range(num_envs):
                    if needs_reset[env_index]:
                        dispatch_reset(env_index)
                    else:
                        dispatch_action(env_index)

                while not all(count == steps_per_env for count in counts):
                    for connection in pool.ready(pending):
                        meta = pending.pop(connection)
                        env_index = meta["env"]
                        payload = pool.recv(env_index)

                        if meta["kind"] == "reset":
                            raw_observations[env_index], reset_info = payload
                            needs_reset[env_index] = False
                            seconds = float(reset_info.get("reset_seconds", 0.0))
                            attempts = int(reset_info.get("reset_attempts", 0))
                            update_reset_seconds += seconds
                            update_reset_retries += max(attempts - 1, 0)
                            if attempts > 1 or seconds > 12.0:
                                print(
                                    f"reset port {args.ports[env_index]}: "
                                    f"{seconds:.1f}s, {attempts} menu attempts"
                                )
                            if counts[env_index] < steps_per_env:
                                pool.send(env_index, "start")
                                pool.recv(env_index)
                                dispatch_action(env_index)
                            continue

                        next_raw, reward, terminated, truncated, info = payload
                        done = bool(terminated or truncated)
                        buffer.add(
                            env_index,
                            meta["obs"],
                            meta["action"],
                            meta["logprobs"],
                            meta["value"],
                            reward * REWARD_SCALE,
                            done,
                        )
                        raw_observations[env_index] = next_raw
                        counts[env_index] += 1
                        global_step += 1
                        episode_rewards[env_index] += reward
                        episode_steps[env_index] += 1
                        for index, name in enumerate(BUTTON_ORDER):
                            if meta["action"][index]:
                                episode_actions[env_index][name] += 1
                        episode_reward_parts[env_index].update(info.get("reward_parts", {}))

                        if done:
                            add_episode(info, env_index)
                            needs_reset[env_index] = True

                        if counts[env_index] < steps_per_env:
                            if needs_reset[env_index]:
                                dispatch_reset(env_index)
                            else:
                                dispatch_action(env_index)
                        else:
                            # Stop the completed game while slower workers
                            # finish and PPO updates. Its stored next state
                            # stays valid at the next rollout boundary.
                            pool.send(env_index, "freeze")
                            pool.recv(env_index)

                last_values = [
                    0.0 if needs_reset[i]
                    else model.value_only(observe(raw_observations[i], update=False))
                    for i in range(num_envs)
                ]
                stats = ppo_update(
                    model, optimizer, buffer, last_values,
                    gamma=GAMMA,
                    gae_lambda=GAE_LAMBDA,
                    clip_eps=CLIP_EPS,
                    epochs=EPOCHS,
                    minibatch_size=MINIBATCH_SIZE,
                    ent_coef=args.ent_coef,
                    vf_coef=VF_COEF,
                    max_grad_norm=MAX_GRAD_NORM,
                    target_kl=TARGET_KL,
                    clip_value_loss=True,
                    value_clip_eps=VALUE_CLIP_EPS,
                )

                elapsed = time.perf_counter() - update_started
                log_metrics({"event": "update", "update": update,
                             "global_step": global_step, "episode": episode,
                             "elapsed_seconds": elapsed,
                             "reset_seconds": update_reset_seconds,
                             "reset_retries": update_reset_retries, **(stats or {})})
                throughput = len(buffer) / max(elapsed, 1e-6)
                if stats is not None:
                    if monitor is not None:
                        monitor.add_update(stats)
                    print(
                        f"upd {update:5d} | step {global_step:9d} | ep {episode:5d} | "
                        f"{throughput:5.1f} step/s | reset {update_reset_seconds:.1f}s/"
                        f"{update_reset_retries} retries | pi {stats['policy_loss']:+.4f} | "
                        f"v {stats['value_loss']:.4f} | H {stats['entropy']:.3f} | "
                        f"kl {stats['approx_kl']:.4f} | ev {stats['explained_variance']:+.2f}"
                    )

                if update % SAVE_EVERY_UPDATES == 0:
                    save_checkpoint(update)
                if args.eval_every > 0 and update % args.eval_every == 0:
                    run_greedy_eval(update)
                if (args.stop_below_hits > 0 and len(recent_hits) == recent_hits.maxlen
                        and np.mean(recent_hits) < args.stop_below_hits):
                    mean_hits = float(np.mean(recent_hits))
                    log_metrics({"event": "regression_stop", "update": update,
                                 "recent_mean_hits": mean_hits,
                                 "threshold": args.stop_below_hits})
                    print(f"stopped: last 20 fights averaged {mean_hits:.2f} hits "
                          f"(< {args.stop_below_hits:g}); checkpoint saved", flush=True)
                    break
        finally:
            pool.close()

    if monitor is None:
        train_loop()
    else:
        def monitored_train_loop():
            try:
                train_loop()
            except Exception as exc:
                error_path = os.path.join(run_dir, "training_error.log")
                details = traceback.format_exc()
                with open(error_path, "w", encoding="utf-8") as stream:
                    stream.write(details)
                monitor.training_error = str(exc).splitlines()[0]
                monitor.training_status = "failed"
                print(details, flush=True)
            else:
                monitor.training_status = "complete"

        threading.Thread(target=monitored_train_loop, daemon=True).start()
        monitor.start()


if __name__ == "__main__":
    mp.freeze_support()
    main()
