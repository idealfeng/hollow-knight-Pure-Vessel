"""Exercise scene reset in one staged game; does not load or train a policy."""

import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="127.0.0.1:9001")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--natural-death", action="store_true",
                        help="let the boss kill an idle knight during the first cycle")
    parser.add_argument("--natural-death-each-cycle", action="store_true",
                        help="check damage and i-frames before and after repeated resets")
    parser.add_argument("--reset-after-first-hit", action="store_true",
                        help="reset the first cycle during damage invulnerability (requires >=2 cycles)")
    parser.add_argument("--disconnect-check", action="store_true",
                        help="also verify a dropped control connection resumes normal death")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error("--cycles must be positive")
    if args.reset_after_first_hit and (args.cycles < 2 or not args.natural_death_each_cycle):
        parser.error("--reset-after-first-hit requires --natural-death-each-cycle and --cycles >= 2")
    from endpoints import parse_endpoint
    host, port = parse_endpoint(args.endpoint)
    os.environ["HOLLOWRL_HOST"] = host
    os.environ["HOLLOWRL_PORT"] = str(port)
    import controls
    import game_state
    import fight_reset
    from hollow_knight_env import HollowKnightEnv
    from main_multi import prepare_fight

    prepare_fight(host, port)
    env = HollowKnightEnv(reuse_live_fight=True)
    successful = False
    results = []
    try:
        env.reset(options={"reuse_live_fight": True, "require_live_on_first_reset": True})
        if not env._fast_reset_supported:
            raise RuntimeError("This endpoint does not support RESET_FIGHT; install the new DLL")
        initial = {key: env.state[key] for key in ("knight_health", "mana", "boss_health")}
        print("captured", fight_reset.status(), "resources", initial, flush=True)
        for index in range(args.cycles):
            natural_death = args.natural_death_each_cycle or (args.natural_death and index == 0)
            forced_death = index % 2 == 0 and not natural_death
            damage_trace = []
            if natural_death:
                previous_hp = env.state["knight_health"]
                started = time.perf_counter()
                controls.set_speed(1.0)
                while True:
                    _, _, terminated, truncated, info = env.step([0] * 7)
                    hp = env.state["knight_health"]
                    if truncated:
                        raise RuntimeError(
                            f"Natural-death test ended without a boss kill: {info['truncated_reason']}; "
                            "check F8 debug health lock, pause state and connectivity"
                        )
                    if hp > previous_hp:
                        raise RuntimeError(
                            f"Idle knight unexpectedly healed: {previous_hp} -> {hp}; "
                            "check F8 debug health lock or other healing mods"
                        )
                    if hp < previous_hp:
                        event = {"seconds": round(time.perf_counter() - started, 3),
                                 "from_hp": previous_hp, "to_hp": hp}
                        damage_trace.append(event)
                        print("damage", event, flush=True)
                        if previous_hp - hp > 2:
                            raise RuntimeError(f"Unexpected Traitor Lord damage burst: {event}")
                        if len(damage_trace) > 1 and event["seconds"] - damage_trace[-2]["seconds"] < 0.3:
                            raise RuntimeError(f"Damage invulnerability missing: {damage_trace[-2:]}")
                        if args.reset_after_first_hit and index == 0 and hp > 0:
                            print("resetting during active damage invulnerability", flush=True)
                            break
                    previous_hp = hp
                    if terminated or truncated:
                        if not terminated or info["won"]:
                            raise RuntimeError(f"Expected a natural knight death: {info}")
                        break
                print("damage test complete", "boss_hp", game_state.boss_health,
                      "knight_hp", game_state.knight_health, flush=True)
            if forced_death:
                controls.kill()
                deadline = time.perf_counter() + 3.0
                while game_state.knight_health > 0 and time.perf_counter() < deadline:
                    time.sleep(0.02)
                if game_state.knight_health > 0 or game_state.boss_health <= 0:
                    raise RuntimeError("Training death was not held in the boss scene")
                print("death held", "boss_hp", game_state.boss_health,
                      "knight_hp", game_state.knight_health, flush=True)
            _, info = env.reset()
            actual = {key: env.state[key] for key in initial}
            if actual != initial or info["reset_mode"] != "scene":
                raise RuntimeError(f"Reset changed resources or used menus: {actual}, {info}")
            if env.state["hit_count"] != 0 or env.state["damage_count"] != 0:
                raise RuntimeError("New fight inherited old episode counters")
            controls.request("RESET_FIGHT VISUAL")
            time.sleep(0.05)  # diagnostics are sampled on Unity's main thread
            visual = controls.request("RESET_FIGHT VISUAL") or ""
            if "main=True" not in visual or "fade=Normal" not in visual:
                raise RuntimeError(f"Scene camera/fade did not recover: {visual}")
            result = {"cycle": index + 1, "forced_death": forced_death,
                      "natural_death": natural_death,
                      "reset_during_invulnerability": args.reset_after_first_hit and index == 0,
                      "reset_seconds": info["reset_seconds"], "mode": info["reset_mode"],
                      "resources": actual, "status": fight_reset.status(), "visual": visual,
                      "damage_trace": damage_trace}
            results.append(result)
            print(json.dumps(result), flush=True)
        if args.disconnect_check:
            from enviroment.reset import statue_position
            controls.kill()
            deadline = time.perf_counter() + 3.0
            while game_state.knight_health > 0 and time.perf_counter() < deadline:
                time.sleep(0.02)
            if game_state.knight_health > 0:
                raise RuntimeError("Disconnect test could not hold a terminal state")
            controls.close()  # exercise the server's socket-disconnect cleanup
            deadline = time.perf_counter() + 15.0
            while statue_position() is None and time.perf_counter() < deadline:
                time.sleep(0.05)
            if statue_position() is None:
                raise RuntimeError("Disconnected trainer left the game in a held death")
            print("disconnect recovery passed; knight returned to the statue", flush=True)
        successful = True
    finally:
        env.close()
        controls.set_speed(0.0 if successful and not args.disconnect_check else 1.0)
        controls.close()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"endpoint": args.endpoint, "results": results},
                                         indent=2), encoding="utf-8")
    print("fast reset validation passed; game " + (
        "returned to statue" if args.disconnect_check else "is frozen ready for training"), flush=True)


if __name__ == "__main__":
    main()
