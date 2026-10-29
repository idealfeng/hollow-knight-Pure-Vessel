"""Subprocess boundary for one live Hollow Knight instance.

The environment modules keep their socket connection and latest frame in
module globals, so process isolation is intentional: each process sets its
port before importing them and therefore owns exactly one game connection.
"""

import os
import time
import traceback


def env_worker_main(connection, host, port, control_hz, frame_stack,
                    episode_time_limit, game_speed):
    host = str(host)
    port = int(port)
    os.environ["HOLLOWRL_HOST"] = host
    os.environ["HOLLOWRL_PORT"] = str(port)

    # Windows spawn imports main_multi (and thus agent.action -> controls and
    # game_state) before this function runs. Correct already-imported module
    # constants before hollow_knight_env starts the state polling thread.
    import controls
    import game_state
    controls.HOST = host
    controls.PORT = port
    game_state.HOST = host
    game_state.PORT = port

    from hollow_knight_env import HollowKnightEnv

    env = None
    try:
        def set_game_speed(target):
            # A scene transition can overwrite Time.timeScale *after* the mod
            # acknowledges SPEED. Reissue it until a fresh state snapshot
            # stays at the requested speed, rather than killing the worker on
            # one transient 1x frame at a rollout boundary.
            deadline = time.monotonic() + 6.0
            next_send = 0.0
            stable_since = None
            sent_successfully = False
            while time.monotonic() < deadline:
                now = time.monotonic()
                if now >= next_send:
                    sent_successfully = controls.set_speed(target) or sent_successfully
                    next_send = now + 0.25
                if (sent_successfully and game_state.connected
                        and abs(game_state.game_speed - target) < 0.05):
                    if stable_since is None:
                        stable_since = now
                    elif now - stable_since >= 0.15:
                        return
                else:
                    stable_since = None
                time.sleep(0.03)
            raise RuntimeError(
                f"port {port}: game speed stayed at {game_state.game_speed} "
                f"instead of {target}"
            )

        env = HollowKnightEnv(
            control_hz=control_hz,
            frame_stack=frame_stack,
            episode_time_limit=episode_time_limit,
        )
        connection.send((True, {
            "event": "ready", "host": host, "port": port,
            "control_port": controls.PORT, "state_port": game_state.PORT,
        }))

        while True:
            command = connection.recv()
            op = command[0]
            if op == "close":
                break
            try:
                if op == "reset":
                    if env._has_reset_once and not env._fast_reset_supported:
                        # A terminal transition may have been frozen at the
                        # previous rollout boundary. Let its animation/menu
                        # progress at 1x, then restore training speed after
                        # the new fight has actually loaded.
                        set_game_speed(1.0)
                    observation, info = env.reset(options={
                        "reuse_live_fight": True,
                        "require_live_on_first_reset": True,
                    })
                    payload = (observation, info)
                elif op == "step":
                    payload = env.step(command[1])
                elif op == "start":
                    set_game_speed(game_speed)
                    payload = {"port": int(port), "game_speed": game_speed}
                elif op == "freeze":
                    set_game_speed(0.0)
                    payload = {"port": int(port), "game_speed": 0.0}
                else:
                    raise ValueError(f"unknown worker command {op!r}")
                connection.send((True, payload))
            except Exception:
                connection.send((False, traceback.format_exc()))
                break
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        try:
            connection.send((False, traceback.format_exc()))
        except (EOFError, BrokenPipeError):
            pass
    finally:
        if env is not None:
            try:
                controls.set_speed(1.0)
                env.close()
            except Exception:
                pass
        connection.close()
