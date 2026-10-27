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
    parser.add_argument("--policy", choices=("checkpoint", "random"), required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--game-speed", type=float, default=2.0)
    parser.add_argument("--output", required=True)
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

    import controls
    import game_state
    from agent.action import MACRO_ACTION_NAMES, MacroAction, boss_phase, is_attack_action
    from agent.calc import box_center, box_distance
    from hollow_knight_env import HollowKnightEnv
    from main_multi import prepare_fight
    from obs_norm import RunningNorm
    from ppo_model import ActorCritic

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
        if (checkpoint.get("observation_size") != state_size
                or checkpoint.get("action_names") != list(MACRO_ACTION_NAMES)):
            raise RuntimeError("checkpoint observation/action format does not match")
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
    episodes = []

    try:
        observation, _ = env.reset(options={
            "reuse_live_fight": True,
            "require_live_on_first_reset": True,
        })
        set_speed(args.game_speed)

        for episode_index in range(args.episodes):
            episode_parts = Counter()
            episode_actions = Counter()
            episode_reward = 0.0
            while True:
                if model is None:
                    action = random.randrange(len(MACRO_ACTION_NAMES))
                else:
                    action = model.act(normalizer.normalize(observation))[0][0]

                action_counts[action] += 1
                episode_actions[action] += 1
                if is_attack_action(action):
                    state = env.state
                    knight_box = state["knight_box"]
                    boss_box = state["boss_box"]
                    attack_phases[boss_phase(state).name.lower()] += 1
                    if knight_box is not None and state["boss_health"] > 0:
                        knight_x, knight_y = box_center(knight_box)
                        boss_x, boss_y = state["boss_position"]
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
                        "min_boss_hp": info["min_boss_health"],
                        "won": info["won"],
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
                set_speed(args.game_speed)
    finally:
        controls.set_speed(1.0)
        env.close()

    total = sum(action_counts.values())
    result = {
        "policy": args.policy,
        "port": args.port,
        "checkpoint": args.checkpoint if model is not None else None,
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
        "reward_parts": dict(reward_parts),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(f"diagnostics written: {args.output}", flush=True)


if __name__ == "__main__":
    main()
