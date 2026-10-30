"""Gymnasium environment around the HollowRL TCP/mod interface.

One Python process controls one game instance. For future vector training,
start each worker with a distinct ``HOLLOWRL_PORT`` before importing this
module; the matching game process/mod must use the same port.
"""

from collections import deque
import math
import os
import time

import gymnasium as gym
import numpy as np

import controls
import game_state
import fight_reset
from agent.action import NUM_BUTTONS, execute_buttons
from agent.reward import RewardTracker, get_reward_config
from injury_context import combat_context
from combat_trace import CombatTrace
from enviroment import OBSERVATION_SIZE, get_state, is_done, reset_environment, state_to_vector
from enviroment.reset import state_is_combat_ready

DEFAULT_CONTROL_HZ = 10
DEFAULT_FRAME_STACK = 4
DEFAULT_TIME_LIMIT = 60.0


class FrameStacker:
    def __init__(self, width, count):
        if count < 1:
            raise ValueError("frame stack must be >= 1")
        self.width = int(width)
        self.count = int(count)
        self.frames = deque(maxlen=self.count)

    @property
    def output_size(self):
        return self.width * self.count

    def reset(self, observation):
        frame = np.asarray(observation, dtype=np.float32)
        self.frames.clear()
        for _ in range(self.count):
            self.frames.append(frame.copy())
        return self.value()

    def append(self, observation):
        self.frames.append(np.asarray(observation, dtype=np.float32))
        return self.value()

    def value(self):
        if len(self.frames) != self.count:
            raise RuntimeError("frame stack has not been reset")
        return np.concatenate(tuple(self.frames), dtype=np.float32)


class HollowKnightEnv(gym.Env):
    """A fixed-rate, vector-observation boss fight environment."""

    metadata = {"render_modes": []}

    def __init__(self, control_hz=DEFAULT_CONTROL_HZ,
                 frame_stack=DEFAULT_FRAME_STACK,
                 episode_time_limit=DEFAULT_TIME_LIMIT,
                 reuse_live_fight=False, reward_profile="victory_v1", boss_level=None,
                 observation_time_limit=None):
        super().__init__()
        self.control_hz = int(control_hz)
        if self.control_hz <= 0:
            raise ValueError("control_hz must be positive")
        self.step_dt = 1.0 / self.control_hz
        self.episode_time_limit = float(episode_time_limit)
        if not math.isfinite(self.episode_time_limit) or self.episode_time_limit <= 0:
            raise ValueError("episode time limit must be finite and positive")
        # Keep a transferred policy's elapsed-time input on its original scale
        # when extending the actual fight budget. None retains legacy behavior.
        self.observation_time_limit = float(
            self.episode_time_limit if observation_time_limit is None else observation_time_limit
        )
        if not math.isfinite(self.observation_time_limit) or self.observation_time_limit <= 0:
            raise ValueError("observation time limit must be finite and positive")
        self.reuse_live_fight = bool(reuse_live_fight)
        self.boss_level = boss_level
        if boss_level not in (None, 0, 1, 2):
            raise ValueError("boss level must be 0, 1, or 2")

        self.stacker = FrameStacker(OBSERVATION_SIZE, int(frame_stack))
        self.action_space = gym.spaces.MultiBinary(NUM_BUTTONS)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.stacker.output_size,),
            dtype=np.float32,
        )

        self.reward_profile = reward_profile
        self.reward_tracker = RewardTracker(reward_config=get_reward_config(reward_profile))
        self.damage_events = []
        self.recent_context = deque(maxlen=5)
        self.state = None
        self.raw_observation = None
        self.prev_buttons = None
        self.episode_steps = 0
        self.min_boss_hp = None
        self._has_reset_once = False
        self.last_reset_seconds = 0.0
        self.last_reset_attempts = 0
        self.total_reset_seconds = 0.0
        self.total_reset_retries = 0
        self.reset_count = 0
        self.last_reset_mode = "reuse"
        self._fast_reset_checked = False
        self._fast_reset_supported = False
        self.combat_trace = CombatTrace.from_environment(self.step_dt)

    def _time_fraction(self, steps=None):
        steps = self.episode_steps if steps is None else steps
        return min(steps * self.step_dt / self.observation_time_limit, 1.0)

    def _read(self, steps=None, retries=40, delay=0.02):
        for _ in range(retries):
            death = game_state.episode_death_frame
            if death is not None and self.state is not None:
                # Keep the final counters/HP even if Unity has already removed
                # the hero collider or restored HP during the return to Godhome.
                state = dict(self.state)
                state.update(
                    knight_health=0,
                    boss_health=(death.boss_health if death.boss_health >= 0
                                 else self.state["boss_health"]),
                    hit_count=death.hit_count - game_state.hit_count_baseline,
                    damage_count=death.damage_count - game_state.damage_count_baseline,
                    attacking=False, attack_box=None,
                )
                vector = state_to_vector(state, self._time_fraction(steps))
                return state, np.asarray(vector, dtype=np.float32)
            state = get_state()
            try:
                vector = state_to_vector(state, self._time_fraction(steps))
            except (IndexError, TypeError, ValueError):
                vector = None
            if vector is not None:
                return state, np.asarray(vector, dtype=np.float32)
            time.sleep(delay)
        return None, None

    @staticmethod
    def _live(state):
        return (state is not None
                and state.get("knight_box") is not None
                and state.get("knight_health", 0) > 0
                and state.get("boss_health", -1) > 0)

    def _wait_while_paused(self):
        waited = False
        if game_state.training_paused:
            controls.release_all()
        while game_state.training_paused:
            waited = True
            time.sleep(0.1)
        return waited

    def _info(self, won=False, truncated_reason=None, include_damage_events=False):
        state = self.state or {}
        info = {
            "port": int(os.environ.get("HOLLOWRL_PORT", "9000")),
            "hits": int(state.get("hit_count", 0)),
            "damage": int(state.get("damage_count", 0)),
            "episode_damage": int(self.reward_tracker.episode_damage),
            "damage_tracking_valid": bool(self.reward_tracker.damage_tracking_valid),
            "boss_health": int(state.get("boss_health", -1)),
            "min_boss_health": self.min_boss_hp,
            "misses": self.reward_tracker.misses,
            "staggers": self.reward_tracker.staggers,
            "dodges": self.reward_tracker.dodges,
            "reward_parts": dict(self.reward_tracker.last_breakdown),
            "won": bool(won),
            "truncated_reason": truncated_reason,
            "reset_seconds": self.last_reset_seconds,
            "reset_attempts": self.last_reset_attempts,
            "total_reset_seconds": self.total_reset_seconds,
            "total_reset_retries": self.total_reset_retries,
            "reset_count": self.reset_count,
            "reset_mode": self.last_reset_mode,
            "boss_level": self.boss_level,
        }
        if include_damage_events:
            info["damage_events"] = list(self.damage_events)
        return info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        reset_started = time.perf_counter()
        options = options or {}
        # A fresh Windows worker starts its polling thread in parallel with
        # this reset. Give the first connection time to publish a frame.
        first_reset = not self._has_reset_once
        current, _ = self._read(
            retries=200 if first_reset else 5,
            delay=0.05 if first_reset else 0.02,
        )
        reuse = options.get("reuse_live_fight", self.reuse_live_fight)
        if not self._fast_reset_checked:
            self._fast_reset_supported = fight_reset.enable()
            self._fast_reset_checked = True
        if self.boss_level is not None:
            if not self._fast_reset_supported:
                raise RuntimeError("Explicit boss difficulty requires the tier-switch fast-reset mod")
            fight_reset.set_level(self.boss_level)
            reuse = False

        if (options.get("require_live_on_first_reset", False)
                and not self._has_reset_once and not self._live(current)):
            port = os.environ.get("HOLLOWRL_PORT", "9000")
            hp = None if current is None else current.get("boss_health")
            raise RuntimeError(
                f"port {port}: first reset requires an active boss fight "
                f"(boss_health={hp}). Enter the Traitor Lord arena before "
                "starting training."
            )

        fast_details = None
        if self._fast_reset_supported and (self._has_reset_once or not reuse):
            fast_details = fight_reset.reset(state_is_combat_ready)
            if fast_details is not None and not fast_details["ok"]:
                if self.boss_level is not None:
                    raise RuntimeError(f"Difficulty-controlled reset failed; menu fallback disabled: {fast_details}")
                print(f"scene reset failed; falling back to statue navigation: {fast_details}")
                fight_reset.disable()
                self._fast_reset_supported = False
                # We are already resetting. Inspect the live stream rather
                # than the previous episode's deliberately latched death.
                current = get_state()

        if fast_details is not None and fast_details["ok"]:
            self.last_reset_attempts = 1
            self.last_reset_mode = "scene"
            game_state.reset_episode_counters()
        elif self._live(current) and (self._has_reset_once or not reuse):
            controls.kill()
            end = time.perf_counter() + 5.0
            while time.perf_counter() < end and game_state.knight_health > 0:
                time.sleep(0.05)
            current = None

        if not (fast_details is not None and fast_details["ok"]) and not self._live(current):
            reset_details = reset_environment(current, return_details=True)
            self.last_reset_attempts = int(reset_details["attempts"])
            self.total_reset_retries += max(self.last_reset_attempts - 1, 0)
            if not reset_details["ok"]:
                raise RuntimeError(f"could not reset boss fight: {reset_details}")
            self.last_reset_mode = "menu"
        elif not (fast_details is not None and fast_details["ok"]):
            self.last_reset_attempts = 0
            self.last_reset_mode = "reuse"
            # Reusing a fight on worker startup skips reset_environment(),
            # which normally establishes the per-episode counter baselines.
            # Without this, the first episode reports lifetime hits/damage
            # accumulated by earlier runs of the same game process.
            game_state.reset_episode_counters()

        state, raw = self._read(steps=0)
        if not self._live(state) or raw is None:
            raise RuntimeError("reset completed but no live observation arrived")
        if self.boss_level is not None:
            deadline = time.monotonic() + 2.0
            observed = fight_reset.target()
            while observed is not None and observed["observed_level"] != self.boss_level and time.monotonic() < deadline:
                time.sleep(0.05)
                observed = fight_reset.target()
            if observed is None or observed["target_level"] != self.boss_level or observed["observed_level"] != self.boss_level:
                raise RuntimeError(f"Boss difficulty was not applied: expected {self.boss_level}, observed {observed}")
            expected_health = 800 if self.boss_level == 0 else 1300
            if state["boss_health"] != expected_health:
                raise RuntimeError(f"Fresh fight has wrong boss HP: {state['boss_health']} != {expected_health}")

        self.state = state
        self.raw_observation = raw
        self.episode_steps = 0
        self.min_boss_hp = state["boss_health"]
        self.reward_tracker.reset()
        self.prev_buttons = None
        self._has_reset_once = True
        self.damage_events.clear()
        self.recent_context.clear()
        self.recent_context.append(combat_context(state, 0))
        self.last_reset_seconds = time.perf_counter() - reset_started
        self.total_reset_seconds += self.last_reset_seconds
        self.reset_count += 1
        if self.combat_trace is not None:
            self.combat_trace.begin(state, self.reset_count, boss_level=self.boss_level,
                                    reward_profile=self.reward_profile)
        return self.stacker.reset(raw), self._info()

    def step(self, action):
        if self.state is None:
            raise RuntimeError("call reset() before step()")

        if self._wait_while_paused():
            state, raw = self._read()
            if state is None:
                raise RuntimeError("lost game state while paused")
            self.state, self.raw_observation = state, raw

        previous = self.state
        previous_buttons = self.prev_buttons
        self.recent_context.append(combat_context(previous, self.episode_steps, action))
        step_started = time.perf_counter()
        self.prev_buttons = execute_buttons(action, self.prev_buttons)
        action_seconds = time.perf_counter() - step_started

        next_step = self.episode_steps + 1
        timed_out = next_step * self.step_dt >= self.episode_time_limit
        if timed_out:
            controls.kill()

        speed = game_state.game_speed if game_state.game_speed > 0.1 else 1.0
        time.sleep(self.step_dt / speed)

        current, raw = self._read(steps=next_step)
        disconnected = raw is None

        # A victory may remove HeroController in the same transition. Preserve
        # the last valid geometry while applying the mod's latched hp=0.
        if disconnected and game_state.boss_health == 0:
            current = dict(previous)
            current["boss_health"] = 0
            current["damage_count"] = max(previous.get("damage_count", 0),
                                          game_state.damage_count - game_state.damage_count_baseline)
            current["knight_health"] = game_state.knight_health
            current["boss_facing_left"] = None
            current["boss_control_state"] = ""
            raw = np.asarray(
                state_to_vector(current, self._time_fraction(next_step)),
                dtype=np.float32,
            )
            disconnected = False
        elif disconnected:
            current, raw = previous, self.raw_observation

        won = current["boss_health"] == 0
        fight_lost = (not disconnected and not won
                      and previous.get("boss_health", -1) > 0
                      and current.get("boss_health", -1) < 0)
        terminated = bool(won or (current["knight_health"] <= 0 and not timed_out))
        truncated = bool((timed_out and not won) or disconnected or fight_lost)
        done_for_reward = terminated or truncated
        masks_lost = max(previous["knight_health"] - current["knight_health"],
                         current.get("damage_count", 0) - previous.get("damage_count", 0), 0)
        if masks_lost > 0 and not timed_out:
            self.damage_events.append({"masks_lost": int(masks_lost),
                                       "history": list(self.recent_context),
                                       "after": combat_context(current, next_step)})
        if done_for_reward:
            controls.release_all()
            self.prev_buttons = None

        reward = self.reward_tracker.transition(
            previous,
            current,
            done_for_reward,
            timed_out=timed_out,
            step_dt=self.step_dt,
            truncated=truncated,
        )

        self.episode_steps = next_step
        self.state = current
        self.raw_observation = raw
        if current.get("boss_health", -1) >= 0:
            self.min_boss_hp = min(self.min_boss_hp, current["boss_health"])

        stacked = self.stacker.append(raw)
        reason = ("timeout" if timed_out else "disconnect" if disconnected
                  else "fight_lost" if fight_lost else None)
        info = self._info(won, reason, include_damage_events=done_for_reward)
        info["action_wall_seconds"] = action_seconds
        info["step_wall_seconds"] = time.perf_counter() - step_started
        if self.combat_trace is not None:
            self.combat_trace.transition(
                previous, current, action, previous_buttons, step=next_step, reward=reward,
                reward_parts=self.reward_tracker.last_breakdown, terminated=terminated,
                truncated=truncated, won=won, truncated_reason=reason)
        return stacked, reward, terminated, truncated, info

    def close(self):
        if self.combat_trace is not None:
            self.combat_trace.close()
        if self._fast_reset_supported:
            fight_reset.disable()
        controls.release_all()
        controls.close()
