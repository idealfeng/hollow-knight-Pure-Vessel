"""Fixed-weight live policy evaluation with action and reward diagnostics.

The script plays live episodes but never updates model weights or normalizer
statistics. Use a separate --output path for each policy being compared.
"""

import argparse
from collections import Counter
import json
import os
import random
import statistics
import time


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--policy", choices=("checkpoint", "random", "approach",
                                             "approach_aligned", "evasive"),
                        required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--deterministic", action="store_true",
                        help="use the highest-probability checkpoint action instead of sampling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--game-speed", type=float, default=2.0)
    parser.add_argument("--expected-boss-hp", type=int, default=1600,
                        help="fail before evaluating a different boss difficulty")
    parser.add_argument("--output", required=True)
    parser.add_argument("--save-demos", default="",
                        help="optional NPZ of synchronized stacked observations and actions")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    if args.policy == "checkpoint" and not args.checkpoint:
        parser.error("--checkpoint is required for checkpoint policy")
    return args


def main():
    args = parse_args()
    os.environ["HOLLOWRL_PORT"] = str(args.port)

    import torch
    import numpy as np

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    import controls
    import game_state
    from agent.action import MACRO_ACTION_NAMES, MacroAction, boss_phase, is_attack_action
    from agent.calc import box_center, box_distance
    from hollow_knight_env import HollowKnightEnv
    from main_multi import CHECKPOINT_FORMAT, prepare_fight
    from obs_norm import RunningNorm
    from ppo_model import ActorCritic
    from agent.scripted_policy import approach_action, evasive_action

    def set_speed(target):
        deadline = time.monotonic() + 6.0
        next_send = 0.0
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_send:
                controls.set_speed(target)
                next_send = now + 0.25
            if game_state.connected and abs(game_state.game_speed - target) < 0.05:
                return
            time.sleep(0.03)
        raise RuntimeError(
            f"port {args.port}: cannot set speed {target}; "
            f"observed {game_state.game_speed}"
        )

    prepare_fight(args.port)
    env = HollowKnightEnv(reuse_live_fight=True)
    reward_parts = Counter()
    last_breakdown = []
    original_transition = env.reward_tracker.transition

    def record_transition(*arguments, **kwargs):
        nonlocal last_breakdown
        value, last_breakdown = original_transition(
            *arguments, **kwargs, return_breakdown=True
        )
        return value

    env.reward_tracker.transition = record_transition

    model = normalizer = None
    if args.policy == "checkpoint":
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state_size = env.observation_space.shape[0]
        if (checkpoint.get("format_version") != CHECKPOINT_FORMAT
                or checkpoint.get("observation_size") != state_size
                or checkpoint.get("action_names") != list(MACRO_ACTION_NAMES)):
            raise RuntimeError("checkpoint observation/action format does not match; "
                               f"expected format {CHECKPOINT_FORMAT}")
        model = ActorCritic(state_size, [len(MACRO_ACTION_NAMES)],
                            hidden=int(checkpoint["model_hidden"]))
        model.load_state_dict(checkpoint["model"])
        model.eval()
        normalizer = RunningNorm(state_size)
        normalizer.load_state_dict(checkpoint["obs_norm"])

    action_counts = Counter()
    attack_distances = []
    attack_center_distances = []
    attack_dx = []
    attack_dy = []
    attack_phases = Counter()
    wrong_horizontal = 0
    horizontal_requests = 0
    horizontal_facing_confirmed = 0
    horizontal_facing_observed = 0
    move_requests = 0
    move_direction_confirmed = 0
    attack_result_observed = 0
    attack_requests_observed = 0
    episodes = []
    step_wall_seconds = []
    action_wall_seconds = []
    demo_observations = []
    demo_actions = []
    demo_episode_ids = []
    demo_rewards = []
    demo_dones = []

    def check_boss_hp():
        observed = env.state["boss_health"]
        if observed != args.expected_boss_hp:
            raise RuntimeError(
                f"port {args.port}: initial boss HP is {observed}, expected "
                f"{args.expected_boss_hp}; align statue difficulty before comparison"
            )

    try:
        observation, _ = env.reset(options={
            "reuse_live_fight": True,
            "require_live_on_first_reset": True,
        })
        check_boss_hp()
        set_speed(args.game_speed)

        for episode_index in range(args.episodes):
            episode_parts = Counter()
            episode_actions = Counter()
            episode_reward = 0.0
            initial_boss_hp = env.state["boss_health"]
            while True:
                if args.policy == "random":
                    action = random.randrange(len(MACRO_ACTION_NAMES))
                elif args.policy in ("approach", "approach_aligned"):
                    action = approach_action(
                        env.state, require_facing=args.policy == "approach_aligned"
                    )
                elif args.policy == "evasive":
                    action = evasive_action(env.state)
                else:
                    normalized = normalizer.normalize(observation)
                    action = (model.greedy_act(normalized)[0] if args.deterministic
                              else model.act(normalized)[0][0])

                previous_state = env.state
                previous_knight_box = previous_state["knight_box"]
                if args.save_demos:
                    demo_observations.append(np.asarray(observation, dtype=np.float32).copy())
                    demo_actions.append(int(action))
                    demo_episode_ids.append(episode_index)
                action_counts[action] += 1
                episode_actions[action] += 1
                if is_attack_action(action):
                    state = previous_state
                    knight_box = state["knight_box"]
                    boss_box = state["boss_box"]
                    attack_phases[boss_phase(state).name.lower()] += 1
                    if knight_box is not None and state["boss_health"] > 0:
                        knight_x, knight_y = box_center(knight_box)
                        boss_x, boss_y = (box_center(boss_box) if boss_box is not None
                                          else state["boss_position"])
                        dx, dy = boss_x - knight_x, boss_y - knight_y
                        attack_dx.append(dx)
                        attack_dy.append(dy)
                        attack_center_distances.append((dx * dx + dy * dy) ** 0.5)
                        if boss_box is not None:
                            attack_distances.append(box_distance(knight_box, boss_box))
                        if action in (MacroAction.ATTACK_LEFT, MacroAction.ATTACK_RIGHT):
                            horizontal_requests += 1
                            wrong_horizontal += int(
                                (action == MacroAction.ATTACK_LEFT and boss_x > knight_x + 0.5)
                                or (action == MacroAction.ATTACK_RIGHT and boss_x < knight_x - 0.5)
                            )

                observation, reward, terminated, truncated, info = env.step(action)
                step_wall_seconds.append(info["step_wall_seconds"])
                action_wall_seconds.append(info["action_wall_seconds"])
                if args.save_demos:
                    demo_rewards.append(float(reward))
                    demo_dones.append(bool(terminated or truncated))
                current_state = env.state
                if is_attack_action(action):
                    attack_requests_observed += 1
                    attack_result_observed += int(
                        current_state["attacking"]
                        or current_state["hit_count"] > previous_state["hit_count"]
                    )
                if action in (MacroAction.ATTACK_LEFT, MacroAction.ATTACK_RIGHT):
                    horizontal_facing_observed += 1
                    horizontal_facing_confirmed += int(
                        current_state["knight_facing_right"]
                        == (action == MacroAction.ATTACK_RIGHT)
                    )
                if action in (MacroAction.MOVE_LEFT, MacroAction.MOVE_RIGHT):
                    if previous_knight_box is not None and current_state["knight_box"] is not None:
                        old_x, _ = box_center(previous_knight_box)
                        new_x, _ = box_center(current_state["knight_box"])
                        move_requests += 1
                        move_direction_confirmed += int(
                            (new_x - old_x) > 0.05 if action == MacroAction.MOVE_RIGHT
                            else (old_x - new_x) > 0.05
                        )
                episode_reward += reward
                for label, amount in last_breakdown:
                    reward_parts[label] += amount
                    episode_parts[label] += amount
                if terminated or truncated:
                    episode = {
                        "reward": round(episode_reward, 3),
                        "steps": env.episode_steps,
                        "hits": info["hits"],
                        "damage": info["damage"],
                        "misses": info["misses"],
                        "staggers": info["staggers"],
                        "min_boss_hp": info["min_boss_health"],
                        "initial_boss_hp": initial_boss_hp,
                        "won": info["won"],
                        "terminated": bool(terminated),
                        "truncated_reason": info["truncated_reason"],
                        "reward_parts": dict(episode_parts),
                        "actions": {MACRO_ACTION_NAMES[k]: v for k, v in episode_actions.items()},
                    }
                    episodes.append(episode)
                    print(f"episode {episode_index + 1}/{args.episodes}: "
                          f"reward={episode_reward:+.2f} hits={info['hits']} "
                          f"misses={info['misses']} boss_hp={info['min_boss_health']}",
                          flush=True)
                    break

            if episode_index + 1 < args.episodes:
                set_speed(1.0)
                observation, _ = env.reset(options={"reuse_live_fight": True})
                check_boss_hp()
                set_speed(args.game_speed)
    finally:
        controls.set_speed(1.0)
        env.close()

    total = sum(action_counts.values())
    result = {
        "policy": args.policy,
        "port": args.port,
        "checkpoint": args.checkpoint if model is not None else None,
        "deterministic": args.deterministic,
        "seed": args.seed,
        "game_speed": args.game_speed,
        "step_wall_seconds_median": statistics.median(step_wall_seconds),
        "step_wall_seconds_p95": float(np.percentile(step_wall_seconds, 95)),
        "action_wall_seconds_median": statistics.median(action_wall_seconds),
        "episodes": episodes,
        "total_steps": total,
        "action_counts": {name: action_counts[i]
                          for i, name in enumerate(MACRO_ACTION_NAMES)},
        "action_percent": {name: round(100 * action_counts[i] / total, 2)
                           for i, name in enumerate(MACRO_ACTION_NAMES)},
        "attack_requests": sum(action_counts[i] for i in range(len(MACRO_ACTION_NAMES))
                               if is_attack_action(i)),
        "attack_distance_median": (statistics.median(attack_distances)
                                   if attack_distances else None),
        "attack_distance_over_5": sum(d > 5 for d in attack_distances),
        "attack_distance_over_9": sum(d > 9 for d in attack_distances),
        "attack_distance_samples": len(attack_distances),
        "attack_center_distance_median": (statistics.median(attack_center_distances)
                                          if attack_center_distances else None),
        "attack_center_distance_over_5": sum(d > 5 for d in attack_center_distances),
        "attack_center_distance_over_9": sum(d > 9 for d in attack_center_distances),
        "attack_center_distance_samples": len(attack_center_distances),
        "attack_abs_dx_median": (statistics.median(map(abs, attack_dx))
                                 if attack_dx else None),
        "attack_abs_dy_median": (statistics.median(map(abs, attack_dy))
                                 if attack_dy else None),
        "attack_phases": dict(attack_phases),
        "wrong_horizontal_requests": wrong_horizontal,
        "horizontal_attack_requests": horizontal_requests,
        "horizontal_facing_confirmed": horizontal_facing_confirmed,
        "horizontal_facing_observed": horizontal_facing_observed,
        "move_direction_confirmed": move_direction_confirmed,
        "move_requests_observed": move_requests,
        "attack_result_observed": attack_result_observed,
        "attack_requests_observed": attack_requests_observed,
        "reward_parts": dict(reward_parts),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(f"diagnostics written: {args.output}", flush=True)
    if args.save_demos:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_demos)), exist_ok=True)
        np.savez_compressed(
            args.save_demos,
            observations=np.stack(demo_observations),
            actions=np.asarray(demo_actions, dtype=np.int64),
            episode_ids=np.asarray(demo_episode_ids, dtype=np.int32),
            rewards=np.asarray(demo_rewards, dtype=np.float32),
            dones=np.asarray(demo_dones, dtype=bool),
            action_names=np.asarray(MACRO_ACTION_NAMES),
            observation_size=np.int32(env.observation_space.shape[0]),
            format_version=np.int32(CHECKPOINT_FORMAT),
            policy=np.str_(args.policy),
        )
        print(f"demonstrations written: {args.save_demos}", flush=True)


if __name__ == "__main__":
    main()
