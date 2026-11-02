"""Reward shaping for the Traitor Lord environment.

``calculate_reward`` scores one state transition. ``RewardTracker`` adds the
small amount of history required to decide whether a nail swing really missed
and whether an ACTIVE boss phase ended without damage.
"""

from dataclasses import dataclass, field
import math

from agent.action import BossMove, BossPhase, boss_move, boss_phase, is_intro_state
from agent.calc import box_distance
from wave_clearance import WaveClearanceTracker, WaveFrontCriteria
from combat_potential import PotentialConfig, potential_reward

NAIL_DAMAGE_REFERENCE = 33.0
HIT_REWARD_PER_NAIL = 1.0

BOSS_DEFEATED = 100.0
CLEAN_DODGE = 0.05
EMPTY_ATTACK = -0.03

# Time and distance are deliberately weaker than a landed hit. Distance is
# ignored while an attack is dangerous so making space remains legal.
TIME_PENALTY_PER_SECOND = 0.10
FAR_DISTANCE = 9.0
FAR_FULL_DISTANCE = 18.0
FAR_PENALTY_PER_SECOND = 0.05

# Victory curriculum: trades remain negative, while failed episodes no longer
# all collapse to almost exactly the same -76 return.
DAMAGE_PER_MASK = 3.0
DEATH = -20.0


@dataclass(frozen=True)
class RewardConfig:
    """A checkpointable objective; the default retains the victory curriculum."""

    profile: str = "victory_v1"
    nail_damage_reference: float = NAIL_DAMAGE_REFERENCE
    hit_reward_per_nail: float = HIT_REWARD_PER_NAIL
    boss_defeated: float = BOSS_DEFEATED
    clean_dodge: float = CLEAN_DODGE
    empty_attack: float = EMPTY_ATTACK
    time_penalty_per_second: float = TIME_PENALTY_PER_SECOND
    far_distance: float = FAR_DISTANCE
    far_full_distance: float = FAR_FULL_DISTANCE
    far_penalty_per_second: float = FAR_PENALTY_PER_SECOND
    # Optional safety gate for distance pressure after a boss animation ends.
    # Legacy profiles keep their original phase-only behavior.
    distance_hazard_exemption: float = 0.0
    damage_per_mask: float = DAMAGE_PER_MASK
    death: float = DEATH
    flawless_win_bonus: float = 0.0
    shockwave_spacing_bonus: float = 0.0
    shockwave_min_gap: float = 3.0
    shockwave_max_gap: float = 6.0
    shockwave_clear_bonus: float = 0.0
    shockwave_clear_max_rewards: int = 2
    shockwave_clear_version: str = "observed_front_crossing_v1"
    shockwave_clear_min_height: float = 8.0
    shockwave_clear_max_width: float = 4.0
    shockwave_clear_max_spawn_gap: float = 4.0
    shockwave_clear_min_horizontal_speed: float = 0.0
    shockwave_clear_max_vertical_speed: float = 1000000.0
    shockwave_clear_threat_gap: float = 6.0
    shockwave_clear_gap: float = 1.0
    shockwave_clear_confirmation_steps: int = 2
    shockwave_clear_min_horizontal_travel: float = 0.5


@dataclass(frozen=True)
class PhaseDodgeRewardConfig(RewardConfig):
    """Experimental FSM completion objective, separate from hazard clearance.

    Keeping these fields in a subclass leaves existing checkpoint objectives
    byte-for-byte equivalent when serialized with ``asdict``.
    """

    clean_dodge_max_rewards: int | None = None
    clean_dodge_version: str = "observed_move_completion_v1"


@dataclass(frozen=True)
class PotentialRewardConfig(PhaseDodgeRewardConfig):
    potential: PotentialConfig = field(default_factory=PotentialConfig)


_REWARD_CONFIGS = {
    "victory_v1": RewardConfig(),
    "nohit_v1": RewardConfig(profile="nohit_v1", damage_per_mask=6.0,
                              flawless_win_bonus=25.0),
    "radiant_spacing_v1": RewardConfig(profile="radiant_spacing_v1",
                                        shockwave_spacing_bonus=0.20),
    "radiant_wave_clear_v1": RewardConfig(profile="radiant_wave_clear_v1",
                                          shockwave_clear_bonus=1.0),
    # Experimental starting coefficients, not an established optimal range.
    # Actual HP loss rewards nail and spell damage identically; no bonus is
    # paid merely for spending SOUL, standing close or finishing an FSM phase.
    "radiant_spells_v1": RewardConfig(profile="radiant_spells_v1",
                                      nail_damage_reference=21.0,
                                      clean_dodge=0.0, empty_attack=0.0,
                                      far_distance=5.0, far_full_distance=12.0,
                                      far_penalty_per_second=0.15,
                                      distance_hazard_exemption=6.0),
    # Small +0.1 trial for every clean completed move, with no episode cap.
    # A confirmed empty nail swing costs only 0.005 (1/200 of a landed hit).
    # Completion means the observed boss animation ended cleanly, not that a
    # travelling wave or sickle disappeared or would otherwise hit the knight.
    "radiant_spells_dodge_v1": PhaseDodgeRewardConfig(
        profile="radiant_spells_dodge_v1", nail_damage_reference=21.0,
        clean_dodge=0.1, empty_attack=-0.005,
        far_distance=5.0, far_full_distance=12.0,
        far_penalty_per_second=0.15, distance_hazard_exemption=6.0),
    "radiant_potential_v1": PotentialRewardConfig(
        profile="radiant_potential_v1", nail_damage_reference=21.0,
        clean_dodge=0.1, empty_attack=-0.005,
        far_distance=5.0, far_full_distance=12.0,
        # The range potential replaces the old always-far distance cost.
        far_penalty_per_second=0.0, distance_hazard_exemption=6.0),
}


def get_reward_config(profile="victory_v1"):
    try:
        return _REWARD_CONFIGS[profile]
    except (KeyError, TypeError):
        raise ValueError(f"unknown reward profile: {profile!r}") from None

_APPROACH_PHASES = {
    BossPhase.NEUTRAL,
    BossPhase.RECOVERY,
}


def _boss_distance(state):
    knight_box = state.get("knight_box")
    boss_box = state.get("boss_box")
    if knight_box is None or boss_box is None or state.get("boss_health", -1) < 0:
        return None
    return box_distance(knight_box, boss_box)


def _horizontal_boss_gap(state):
    """Edge separation along the floor; jumping cannot increase this gap."""
    knight_box = state.get("knight_box")
    boss_box = state.get("boss_box")
    if knight_box is None or boss_box is None or state.get("boss_health", -1) <= 0:
        return None
    return max(boss_box[0] - knight_box[2], knight_box[0] - boss_box[2], 0.0)


def _distance_pressure_safe(state, exemption):
    """Avoid urging the knight toward a boss while detached hazards remain.

    A neutral/recovery FSM phase does not establish that travelling waves have
    disappeared. Unknown attachment flags count as detached; missing/invalid
    geometry suppresses pressure rather than assuming it is safe to approach.
    This gate only removes a small cost and never awards positive reward.
    """
    if exemption <= 0:
        return True

    def valid_box(box):
        try:
            return (len(box) == 4 and all(math.isfinite(v) for v in box)
                    and box[0] <= box[2] and box[1] <= box[3])
        except (TypeError, ValueError, IndexError):
            return False

    knight = state.get("knight_box")
    hazards = state.get("hazards")
    if (state.get("knight_health", 0) <= 0 or not valid_box(knight)
            or not isinstance(hazards, (list, tuple))):
        return False
    for hazard in hazards:
        attached = (hazard.get("boss_attached") if isinstance(hazard, dict)
                    else getattr(hazard, "boss_attached", None))
        if attached is True:
            continue
        box = (hazard.get("box") if isinstance(hazard, dict)
               else getattr(hazard, "box", None))
        if not valid_box(box) or box_distance(knight, box) <= exemption:
            return False
    return True


def calculate_reward(previous_state, current_state, done, timed_out=False,
                     return_breakdown=False, step_dt=1.0 / 10.0,
                     missed_attack=False, successful_dodge=False, *,
                     reward_config=None, episode_damage=None, truncated=False,
                     successful_wave_spacing=False, successful_wave_clear=False):
    """Score one transition.

    Delayed event flags normally come from :class:`RewardTracker`. They default
    to false so offline transition tests can still call this function directly.
    """
    config = reward_config or get_reward_config()

    reward = 0.0
    breakdown = []

    def add(amount, label):
        nonlocal reward
        reward += amount
        breakdown.append((label, amount))

    prev_hp = previous_state["boss_health"]
    cur_hp = current_state["boss_health"]
    hp_valid = cur_hp >= 0 and prev_hp >= 0

    if hp_valid and cur_hp < prev_hp:
        damage_dealt = prev_hp - cur_hp
        add(damage_dealt / config.nail_damage_reference * config.hit_reward_per_nail,
            "boss damage")

    if done and prev_hp > 0 and cur_hp == 0:
        add(config.boss_defeated, "boss defeated")

    # Whole-fight history must be supplied explicitly; one clean terminal
    # transition alone cannot establish that the episode was flawless.
    if (config.flawless_win_bonus and done and not timed_out and not truncated
            and prev_hp > 0 and cur_hp == 0
            and current_state["knight_health"] > 0
            and episode_damage == 0
            and current_state["knight_health"] >= previous_state["knight_health"]
            and current_state.get("damage_count", 0) <= previous_state.get("damage_count", 0)):
        add(config.flawless_win_bonus, "flawless victory")

    phase_dodge_safe = (
        not isinstance(config, PhaseDodgeRewardConfig)
        or (not done and not timed_out and not truncated
            and prev_hp > 0 and cur_hp > 0
            and current_state["knight_health"] > 0
            and current_state["knight_health"] >= previous_state["knight_health"]
            and "damage_count" in previous_state and "damage_count" in current_state
            and previous_state["damage_count"] >= 0
            and math.isfinite(previous_state["damage_count"])
            and current_state["damage_count"] == previous_state["damage_count"]
            and boss_phase(current_state) in (
                BossPhase.RECOVERY, BossPhase.NEUTRAL, BossPhase.WINDUP)
            and not is_intro_state(current_state.get("boss_control_state", "")))
    )
    if successful_dodge and config.clean_dodge and phase_dodge_safe:
        add(config.clean_dodge, "clean dodge")

    if (successful_wave_spacing and config.shockwave_spacing_bonus
            and not done and not timed_out and not truncated
            and current_state["knight_health"] > 0 and cur_hp > 0):
        add(config.shockwave_spacing_bonus, "shockwave spacing")

    if (successful_wave_clear and config.shockwave_clear_bonus
            and not done and not timed_out and not truncated
            and current_state["knight_health"] > 0 and cur_hp > 0
            and current_state["knight_health"] >= previous_state["knight_health"]
            and current_state.get("damage_count", 0) == previous_state.get("damage_count", 0)):
        add(config.shockwave_clear_bonus, "shockwave clear")

    if missed_attack and config.empty_attack:
        add(config.empty_attack, "empty attack")

    if not timed_out:
        dt = max(step_dt, 0.0)
        add(-config.time_penalty_per_second * dt, "time")

        if (boss_phase(current_state) in _APPROACH_PHASES
                and _distance_pressure_safe(current_state, config.distance_hazard_exemption)):
            distance = _boss_distance(current_state)
            if distance is not None and distance > config.far_distance:
                span = max(config.far_full_distance - config.far_distance, 1e-6)
                severity = min((distance - config.far_distance) / span, 1.0)
                add(-config.far_penalty_per_second * dt * severity,
                    "far from boss")

        masks_lost = previous_state["knight_health"] - current_state["knight_health"]
        if masks_lost > 0:
            add(-masks_lost * config.damage_per_mask, "took damage")

    if (done
            and previous_state["knight_health"] > 0
            and current_state["knight_health"] <= 0
            and not timed_out):
        add(config.death, "die")

    if return_breakdown:
        return reward, breakdown
    return reward


class RewardTracker:
    """Recognise delayed combat events and expose episode diagnostics."""

    def __init__(self, reward_config=None):
        self.reward_config = reward_config or get_reward_config()
        self.reset()

    def reset(self):
        self._swing_hits = None
        self._swing_hp = None
        self._active_damage = None
        self._phase_dodge = None
        self.dodge_rewards = 0
        self.potential_applied_steps = 0
        self.potential_invalid_steps = 0
        self.misses = 0
        self.staggers = 0
        self.dodges = 0
        self.last_breakdown = []
        self.episode_damage = 0
        self.damage_tracking_valid = True
        self._damage_counter_last = None
        self._episode_ended = False
        self._wave_spacing = None
        self.wave_spacing_rewards = 0
        config = self.reward_config
        self._wave_clearance = (WaveClearanceTracker(
            WaveFrontCriteria(config.shockwave_clear_min_height,
                              config.shockwave_clear_max_width,
                              config.shockwave_clear_min_horizontal_speed,
                              config.shockwave_clear_max_vertical_speed,
                              config.shockwave_clear_max_spawn_gap),
            threat_gap=config.shockwave_clear_threat_gap,
            clear_gap=config.shockwave_clear_gap,
            confirmation_steps=config.shockwave_clear_confirmation_steps,
            maximum_rewards=config.shockwave_clear_max_rewards,
            minimum_horizontal_travel=config.shockwave_clear_min_horizontal_travel,
        ) if config.shockwave_clear_bonus else None)
        self.wave_clear_rewards = 0

    def _completed_phase_dodge(self, previous_state, current_state, *,
                               done, timed_out, truncated,
                               masks_lost, counter_delta):
        """Count an observed clean move once; detached hazards are not checked.

        The dive's Jump -> DSlash Antic -> DSlash is one move despite its
        intermediate WINDUP. Other new antic states can begin the next move
        when the sampling cadence missed the intervening recovery.
        """
        previous_phase = boss_phase(previous_state)
        current_phase = boss_phase(current_state)
        previous_move = boss_move(previous_state)
        current_move = boss_move(current_state)
        if (done or timed_out or truncated or self._episode_ended
                or not self.damage_tracking_valid
                or previous_state.get("knight_health", 0) <= 0
                or current_state.get("knight_health", 0) <= 0
                or previous_state.get("boss_health", -1) <= 0
                or current_state.get("boss_health", -1) <= 0
                or current_phase == BossPhase.UNKNOWN
                or "damage_count" not in previous_state
                or "damage_count" not in current_state
                or not math.isfinite(previous_state["damage_count"])
                or not math.isfinite(current_state["damage_count"])
                or is_intro_state(current_state.get("boss_control_state", ""))):
            self._phase_dodge = None
            return False

        clean_step = masks_lost == 0 and counter_delta == 0
        cycle = self._phase_dodge
        if cycle is None:
            if (previous_phase == BossPhase.WINDUP and previous_move != BossMove.NONE
                    and previous_state.get("boss_control_state") != "DSlash Antic"):
                cycle = {"move": previous_move, "active_seen": False,
                         "clean": clean_step}
            elif (previous_phase in (BossPhase.NEUTRAL, BossPhase.RECOVERY)
                  and current_phase == BossPhase.ACTIVE
                  and current_move != BossMove.NONE):
                # A known non-active -> active edge is usable even if the
                # short antic fell between samples. Starting in ACTIVE is not.
                cycle = {"move": current_move, "active_seen": False,
                         "clean": clean_step}

        successful = False
        if cycle is not None:
            cycle["clean"] = cycle["clean"] and clean_step
            if current_phase == BossPhase.ACTIVE:
                if current_move == cycle["move"]:
                    cycle["active_seen"] = True
                else:
                    cycle = None
            elif (current_phase == BossPhase.WINDUP
                  and current_move == BossMove.DIVE
                  and cycle["move"] == BossMove.DIVE
                  and current_state.get("boss_control_state") == "DSlash Antic"):
                # This is preparation for the second active part of one dive.
                pass
            elif current_phase in (BossPhase.RECOVERY, BossPhase.NEUTRAL,
                                    BossPhase.WINDUP):
                move_matches = (current_phase != BossPhase.RECOVERY
                                or current_move in (BossMove.NONE, cycle["move"]))
                if cycle["active_seen"] and move_matches:
                    if cycle["clean"]:
                        self.dodges += 1
                        maximum = self.reward_config.clean_dodge_max_rewards
                        if maximum is None or self.dodge_rewards < maximum:
                            self.dodge_rewards += 1
                            successful = True
                    cycle = None
                elif (current_phase != BossPhase.WINDUP
                      or current_move != cycle["move"]):
                    cycle = None

        if cycle is None and current_phase == BossPhase.WINDUP and current_move != BossMove.NONE:
            # DSlash Antic belongs to the middle of a dive. It can continue
            # an existing cycle but cannot establish a complete one itself.
            if current_state.get("boss_control_state") != "DSlash Antic":
                cycle = {"move": current_move, "active_seen": False,
                         "clean": clean_step}
        self._phase_dodge = cycle
        return successful

    def transition(self, previous_state, current_state, done,
                   timed_out=False, step_dt=1.0 / 10.0,
                   return_breakdown=False, truncated=False):
        missed_attack = False
        successful_dodge = False

        previous_damage = previous_state.get("damage_count", 0)
        current_damage = current_state.get("damage_count", previous_damage)
        if self._damage_counter_last is None:
            # A positive episode-relative initial counter means damage was
            # already observed before the first policy transition.
            self.episode_damage += max(previous_damage, 0)
        elif previous_damage < self._damage_counter_last:
            self.damage_tracking_valid = False
        if min(previous_damage, current_damage) < 0 or current_damage < previous_damage:
            # Never convert a counter/baseline discontinuity into a free bonus.
            self.damage_tracking_valid = False
        counter_delta = max(current_damage - previous_damage, 0)
        if self._damage_counter_last is not None:
            counter_delta += max(previous_damage - self._damage_counter_last, 0)
        masks_lost = max(previous_state["knight_health"] - current_state["knight_health"], 0)
        self.episode_damage += max(masks_lost, counter_delta)
        self._damage_counter_last = current_damage
        successful_wave_clear = False
        if self._wave_clearance is not None:
            event = self._wave_clearance.transition(
                previous_state, current_state,
                done=done or self._episode_ended or not self.damage_tracking_valid,
                timed_out=timed_out, truncated=truncated,
            )
            successful_wave_clear = bool(event and event["reward_eligible"])
            if successful_wave_clear:
                self.wave_clear_rewards += 1

        # Only a game-confirmed attack animation starts a swing. An attack
        # request rejected by cooldown is not incorrectly labelled a whiff.
        if (self._swing_hits is None
                and not previous_state.get("attacking", False)
                and current_state.get("attacking", False)):
            self._swing_hits = previous_state.get("hit_count", 0)
            self._swing_hp = previous_state.get("boss_health", -1)

        if self._swing_hits is not None:
            hit_landed = current_state.get("hit_count", 0) > self._swing_hits
            hp_dropped = (self._swing_hp >= 0
                          and 0 <= current_state.get("boss_health", -1) < self._swing_hp)
            if hit_landed or hp_dropped:
                self._swing_hits = self._swing_hp = None
            elif (previous_state.get("attacking", False)
                  and not current_state.get("attacking", False)):
                missed_attack = True
                self.misses += 1
                self._swing_hits = self._swing_hp = None

        previous_phase = boss_phase(previous_state)
        current_phase = boss_phase(current_state)
        previous_wave_active = (boss_move(previous_state) == BossMove.SHOCKWAVE
                                and previous_phase == BossPhase.ACTIVE)
        current_wave_active = (boss_move(current_state) == BossMove.SHOCKWAVE
                               and current_phase == BossPhase.ACTIVE)
        successful_wave_spacing = False
        if self.reward_config.shockwave_spacing_bonus:
            if current_wave_active and not previous_wave_active:
                gap = _horizontal_boss_gap(current_state)
                self._wave_spacing = {
                    "eligible_gap": (gap is not None
                                     and self.reward_config.shockwave_min_gap <= gap
                                     <= self.reward_config.shockwave_max_gap),
                    "damage_at_start": self.episode_damage,
                    "clean": masks_lost == 0 and counter_delta == 0,
                }
            wave = self._wave_spacing
            if wave is not None:
                if (masks_lost > 0 or counter_delta > 0
                        or self.episode_damage != wave["damage_at_start"]
                        or not self.damage_tracking_valid):
                    wave["clean"] = False
                if done or timed_out or truncated or current_state["knight_health"] <= 0:
                    self._wave_spacing = None
                elif previous_wave_active and not current_wave_active:
                    # A known live phase after ACTIVE establishes completion;
                    # scene disappearance, unknown state or a kill does not.
                    successful_wave_spacing = bool(
                        wave["eligible_gap"] and wave["clean"]
                        and self.damage_tracking_valid and not self._episode_ended
                        and current_state.get("boss_health", -1) > 0
                        and current_phase in (BossPhase.RECOVERY, BossPhase.NEUTRAL,
                                              BossPhase.WINDUP))
                    self._wave_spacing = None
                    if successful_wave_spacing:
                        self.wave_spacing_rewards += 1
        if isinstance(self.reward_config, PhaseDodgeRewardConfig):
            successful_dodge = self._completed_phase_dodge(
                previous_state, current_state, done=done, timed_out=timed_out,
                truncated=truncated, masks_lost=masks_lost,
                counter_delta=counter_delta)
        else:
            # Preserve historical objectives and their checkpoint semantics.
            if self._active_damage is None:
                if (previous_phase != BossPhase.ACTIVE
                        and current_phase == BossPhase.ACTIVE):
                    self._active_damage = previous_state.get("damage_count", 0)
            elif current_state.get("damage_count", 0) > self._active_damage:
                self._active_damage = -1

            if (self._active_damage is not None
                    and previous_phase == BossPhase.ACTIVE
                    and current_phase != BossPhase.ACTIVE):
                if self._active_damage >= 0:
                    successful_dodge = True
                    self.dodges += 1
                self._active_damage = None

        reward, self.last_breakdown = calculate_reward(
            previous_state,
            current_state,
            done,
            timed_out=timed_out,
            return_breakdown=True,
            step_dt=step_dt,
            missed_attack=missed_attack,
            successful_dodge=successful_dodge,
            reward_config=self.reward_config,
            episode_damage=(self.episode_damage if self.damage_tracking_valid
                            and not self._episode_ended else None),
            truncated=truncated,
            successful_wave_spacing=successful_wave_spacing,
            successful_wave_clear=successful_wave_clear,
        )
        if isinstance(self.reward_config, PotentialRewardConfig) and not self._episode_ended:
            shaping = potential_reward(previous_state, current_state,
                                       terminal=bool(done or timed_out or truncated),
                                       config=self.reward_config.potential)
            if shaping is None:
                self.potential_invalid_steps += 1
                raise RuntimeError("Potential training received invalid combat geometry/telemetry; refusing to drop a shaping transition")
            else:
                self.potential_applied_steps += 1
                reward += shaping
                self.last_breakdown.append(("potential shaping", shaping))
        if (done or truncated
                or (isinstance(self.reward_config, PhaseDodgeRewardConfig) and timed_out)):
            self._episode_ended = True
        return (reward, self.last_breakdown) if return_breakdown else reward
