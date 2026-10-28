"""Getting back into the fight after the knight dies.

reset_environment() drives the pantheon menu: wait out the death sequence,
step to the retry option, confirm, then wait for a live fight to appear. It
retries the menu part, because a single missed input otherwise leaves the
trainer feeding actions into a menu screen forever.
"""

import os
import time

import controls
import game_state as game_reader
from agent.action import is_intro_state

# Waits before the menu is reachable. Both are blind sleeps rather than waits
# on observed state: nothing the mod reports distinguishes "death animation
# playing" from "menu up", so there is no predicate to wait on yet.
DEATH_SETTLE_WAIT = float(os.environ.get("HOLLOWRL_DEATH_SETTLE_WAIT", "7.0"))
POST_LEFT_WAIT = float(os.environ.get("HOLLOWRL_POST_LEFT_WAIT", "2.0"))

MENU_STEP_WAIT = float(os.environ.get("HOLLOWRL_MENU_STEP_WAIT", "0.5"))
FIGHT_LOAD_TIMEOUT = float(os.environ.get("HOLLOWRL_FIGHT_LOAD_TIMEOUT", "3.0"))
RETRY_WAIT = float(os.environ.get("HOLLOWRL_RETRY_WAIT", "3.0"))

# How long the fight has to keep looking alive before we believe it. Guards
# against the one frame of "live" state that appears mid-transition.
STABLE_DURATION = 0.5


def game_seconds(seconds):
    """Real seconds to sleep in order to span `seconds` of game time. The
    in-game 2x toggle halves the wall-clock cost of everything below."""
    speed = game_reader.game_speed
    return seconds / (speed if speed > 0.1 else 1.0)


def tap(action, hold=0.06, after=0.0):
    """Press an action, then optionally wait. `hold` is real seconds - Unity
    polls input once per rendered frame and timeScale doesn't change the frame
    rate, so a hold has to be timed in real time. `after` is the caller's to
    scale (see game_seconds)."""
    controls.tap(action, hold_ms=hold * 1000)
    if after:
        time.sleep(after)


def wait_until(predicate, timeout=15.0, poll=0.05):
    """Poll `predicate` until it returns true. True if it did, False on timeout."""
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        if predicate():
            return True
        time.sleep(poll)
    return False


def state_is_live():
    """A fight is actually running: knight on screen, both sides alive."""
    return (game_reader.knight_box is not None
            and game_reader.boss_health > 0
            and game_reader.knight_health > 0)


def state_is_stable(duration=STABLE_DURATION, poll=0.05):
    """state_is_live() held continuously for `duration` seconds."""
    end = time.perf_counter() + duration
    while time.perf_counter() < end:
        if not state_is_live():
            return False
        time.sleep(poll)
    return True


def state_is_combat_ready():
    """The boss intro is over and a real body collider is available."""
    return (state_is_live()
            and bool(game_reader.boss_control_state)
            and not is_intro_state(game_reader.boss_control_state)
            and game_reader.boss_box is not None)


def reset_environment(state=None, attempts=4, return_details=False):
    """Restart the fight.

    ``return_details`` keeps the old boolean API intact while allowing the
    vector trainer to report how much wall time and how many menu attempts
    are being lost to resets.
    """
    started = time.perf_counter()
    controls.release_all()

    def success(attempt):
        controls.release_all()
        # Boss HP appears about 13 game seconds before the Traitor Lord intro
        # ends. Those frames cannot teach combat and previously consumed a
        # large fraction of every rollout as idle/missed attacks.
        if not wait_until(state_is_combat_ready, timeout=game_seconds(22.0)):
            print("*** RESET FAILED - boss intro did not reach combat ***")
            result = {
                "ok": False,
                "attempts": attempt,
                "seconds": time.perf_counter() - started,
            }
            return result if return_details else False
        game_reader.reset_episode_counters()
        result = {
            "ok": True,
            "attempts": attempt,
            "seconds": time.perf_counter() - started,
        }
        return result if return_details else True

    time.sleep(DEATH_SETTLE_WAIT)

    tap("left")
    time.sleep(POST_LEFT_WAIT)

    for attempt in range(1, attempts + 1):
        tap("up", after=game_seconds(MENU_STEP_WAIT))
        tap("submit", after=0)

        if (wait_until(state_is_live, timeout=game_seconds(FIGHT_LOAD_TIMEOUT))
                and state_is_stable()):
            return success(attempt)

        print(f"reset attempt {attempt}/{attempts} did not take, retrying")
        # A slow scene load can finish just after FIGHT_LOAD_TIMEOUT. Watch
        # state during the retry delay instead of blindly pressing the menu
        # again over a fight that has already begun.
        if (wait_until(state_is_live, timeout=game_seconds(RETRY_WAIT))
                and state_is_stable()):
            return success(attempt)

    # Focus is no longer a possible cause - input goes straight into the game
    # process now - so a failure here means the menu is not where we think.
    print("*** RESET FAILED - is the game sitting on the death/retry menu? ***")
    result = {
        "ok": False,
        "attempts": attempts,
        "seconds": time.perf_counter() - started,
    }
    return result if return_details else False
