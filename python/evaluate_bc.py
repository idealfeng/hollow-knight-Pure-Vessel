"""Play a saved button policy without updating its weights."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    # The socket modules read their port at import time.
    os.environ["HOLLOWRL_PORT"] = str(args.port)
    import torch
    import controls
    from agent.action import BUTTON_ORDER, NUM_BUTTONS
    from hollow_knight_env import HollowKnightEnv
    from main_multi import CHECKPOINT_FORMAT, prepare_fight
    from obs_norm import RunningNorm
    from ppo_model import ActorCritic

    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (checkpoint.get("format_version") != CHECKPOINT_FORMAT
            or checkpoint.get("action_distribution") != "exclusive_directions_v1"
            or checkpoint.get("action_names") != list(BUTTON_ORDER)):
        raise ValueError("checkpoint does not match the current button policy")
    frame_stack = int(checkpoint["frame_stack"])
    model = ActorCritic(
        int(checkpoint["observation_size"]), [1] * NUM_BUTTONS,
        hidden=int(checkpoint["model_hidden"]), bernoulli=True,
        exclusive_directions=True,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    obs_norm = RunningNorm(int(checkpoint["observation_size"]))
    obs_norm.load_state_dict(checkpoint["obs_norm"])

    prepare_fight(args.port)
    env = HollowKnightEnv(control_hz=int(checkpoint.get("control_hz", 10)),
                          frame_stack=frame_stack, reuse_live_fight=True)
    results = []
    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(options={
                "reuse_live_fight": True,
                "require_live_on_first_reset": True,
            })
            if not controls.set_speed(1.0):
                raise RuntimeError("could not resume game speed after fight preparation")
            total_reward = 0.0
            choices = Counter()
            presses = Counter()
            previous = (0,) * NUM_BUTTONS
            started = time.perf_counter()
            while True:
                action = model.greedy_act(obs_norm.normalize(observation))
                for index, name in enumerate(BUTTON_ORDER):
                    if action[index]:
                        choices[name] += 1
                        if name == "attack" or not previous[index]:
                            presses[name] += 1
                previous = action
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    result = {
                        "episode": episode + 1, "port": args.port,
                        "reward": total_reward, "elapsed_seconds": time.perf_counter() - started,
                        "choices": dict(choices), "presses": dict(presses), **info,
                    }
                    results.append(result)
                    print(json.dumps(result, ensure_ascii=False), flush=True)
                    break
    finally:
        controls.set_speed(1.0)
        env.close()

    report = args.checkpoint.parent / "bc_eval.jsonl"
    with report.open("a", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"saved {len(results)} evaluation result(s) to {report}")


if __name__ == "__main__":
    main()
