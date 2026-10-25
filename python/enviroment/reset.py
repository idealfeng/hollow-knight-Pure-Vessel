"""Getting back into the fight after the knight dies.

reset_environment() drives the pantheon menu: wait out the death sequence,
step to the retry option, confirm, then wait for a live fight to appear. It
retries the menu part, because a single missed input otherwise leaves the
trainer feeding actions into a menu screen forever.
"""

import time

import controls
import game_state as game_reader

# Waits before the menu is reachable. Both are blind sleeps rather than waits
# on observed state: nothing the mod reports distinguishes "death animation
# playing" from "menu up", so there is no predicate to wait on yet.
DEATH_SETTLE_WAIT = 7.0      # death sequence + scene teardown
POST_LEFT_WAIT = 2.0         # menu settling after the initial "left"

MENU_STEP_WAIT = 0.5         # the tap("up") settle
FIGHT_LOAD_TIMEOUT = 3.0     # how long to give the fight to come up
RETRY_WAIT = 3.0             # between failed attempts

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


def reset_environment(state=None, attempts=4):
    """Restart the fight. Returns True if one actually started."""
    controls.release_all()

    time.sleep(DEATH_SETTLE_WAIT)

    tap("left")
    time.sleep(POST_LEFT_WAIT)

    for attempt in range(1, attempts + 1):
        tap("up", after=game_seconds(MENU_STEP_WAIT))
        tap("submit", after=0)

        if (wait_until(state_is_live, timeout=game_seconds(FIGHT_LOAD_TIMEOUT))
                and state_is_stable()):
            controls.release_all()
            game_reader.reset_episode_counters()
            return True

        print(f"reset attempt {attempt}/{attempts} did not take, retrying")
        time.sleep(game_seconds(RETRY_WAIT))

    # Focus is no longer a possible cause - input goes straight into the game
    # process now - so a failure here means the menu is not where we think.
    print("*** RESET FAILED - is the game sitting on the death/retry menu? ***")
    return False
