"""Bounded state potentials for recorded geometry, never an action controller.

TTC uses constant-velocity axis-aligned boxes over a short horizon. It is a
risk estimate: accelerating attacks, newly spawned hazards and future input
changes are not predicted. No reward is paid for a ready cooldown or a button.
"""

from dataclasses import dataclass
import math

@dataclass(frozen=True)
class PotentialConfig:
    gamma: float = 0.9955
    eta: float = 0.1
    range_weight: float = 0.25
    safety_weight: float = 1.0
    # Initial interval informed by recorded uncharmed nail hits, not an exact
    # contact-distance calibration: the controller observes only at 10 Hz.
    range_min_gap: float = 2.0
    range_max_gap: float = 3.2
    range_distance_scale: float = 6.0
    ttc_horizon: float = 0.5
    attack_phase_range_multiplier: float = 0.0
    protection_time_margin: float = 0.02
    require_dash_telemetry: bool = True

    def __post_init__(self):
        values = (self.gamma, self.eta, self.range_weight, self.safety_weight,
                  self.range_min_gap, self.range_max_gap, self.range_distance_scale,
                  self.ttc_horizon, self.attack_phase_range_multiplier,
                  self.protection_time_margin)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Potential coefficients must be finite")
        if not 0 < self.gamma <= 1 or min(self.eta, self.range_weight, self.safety_weight) < 0:
            raise ValueError("Invalid discount or potential scale")
        if (self.range_min_gap < 0 or self.range_max_gap < self.range_min_gap
                or self.range_distance_scale <= 0 or self.ttc_horizon <= 0
                or not 0 <= self.attack_phase_range_multiplier <= 1
                or self.protection_time_margin < 0):
            raise ValueError("Invalid potential geometry or horizon")


@dataclass(frozen=True)
class PotentialResult:
    value: float | None
    range_value: float
    safety_value: float
    valid: bool
    earliest_unprotected_ttc: float | None = None
    reason: str = ""


def _box(value):
    try:
        if (len(value) != 4 or not all(math.isfinite(v) for v in value)
                or value[0] >= value[2] or value[1] >= value[3]):
            return None
        return tuple(float(v) for v in value)
    except (TypeError, ValueError):
        return None


def _velocity(value):
    try:
        return (tuple(float(v) for v in value)
                if len(value) == 2 and all(math.isfinite(v) for v in value) else None)
    except (TypeError, ValueError):
        return None


def swept_aabb_interval(knight_box, hazard_box, relative_velocity):
    """Return future contact [entry, exit], or None if no contact is predicted.

    ``relative_velocity`` is hazard minus knight velocity. Invalid arguments
    raise ValueError rather than conflating missing data with a safe state.
    The exit can be infinite for persistent stationary overlap.
    """
    knight, hazard, velocity = _box(knight_box), _box(hazard_box), _velocity(relative_velocity)
    if knight is None or hazard is None or velocity is None:
        raise ValueError("TTC requires finite nonempty boxes and a two-axis velocity")
    entry, exit_time = -math.inf, math.inf
    for axis in (0, 1):
        low, high = axis, axis + 2
        speed = velocity[axis]
        if speed == 0:
            if hazard[high] < knight[low] or hazard[low] > knight[high]:
                return None
            continue
        a = (knight[low] - hazard[high]) / speed
        b = (knight[high] - hazard[low]) / speed
        entry = max(entry, min(a, b))
        exit_time = min(exit_time, max(a, b))
    if exit_time < max(entry, 0.0):
        return None
    return max(entry, 0.0), exit_time


def _protected_until(state, hazard_type, config):
    """Use only confirmed current protection with a known remaining duration.

    ``can_shade_cloak_now`` deliberately has no role. Unknown protection
    duration receives no exemption, preserving conservative geometry risk.
    """
    durations = []
    remaining_keys = []
    if state.get("is_shadow_dashing") and hazard_type == 1:
        # The game's shadow dash exempts enemy damage (hazardType 1), not
        # terrain/spikes or an unknown collider category.
        remaining_keys.append("shadow_dash_remaining")
    if state.get("is_invuln") and state.get("invulnerability_time_valid") is True:
        remaining_keys.append("invulnerability_remaining")
    for remaining_key in remaining_keys:
        duration = state.get(remaining_key)
        if isinstance(duration, (int, float)) and math.isfinite(duration) and duration >= 0:
            durations.append(duration)
    # Leave one physics-step margin rather than treating countdown equality
    # as guaranteed coverage. Wall contact or action cancellation can end a
    # dash sooner still; constant-velocity TTC remains an approximation.
    return max(max(durations, default=0.0) - config.protection_time_margin, 0.0)


def _invalid(reason):
    return PotentialResult(None, 0.0, 0.0, False, reason=reason)


def potential(state, config=None):
    """Evaluate one fixed bounded potential, including state-dependent weights.

    Safe states have safety potential 0; dangerous states lie in [-1, 0].
    Taking a transition out of danger therefore preserves its full score
    increase instead of turning shaping off on the safe side of the edge.
    """
    # agent.__init__ re-exports rewards, which import this pure module. Delay
    # the classification import until evaluation to avoid that import cycle.
    from agent.action import BossPhase, boss_phase

    config = config or PotentialConfig()
    if config.require_dash_telemetry and state.get("dash_state_valid") is not True:
        return _invalid("missing dash telemetry")
    phase = boss_phase(state)
    if state.get("knight_health", 0) <= 0 or state.get("boss_health", -1) <= 0:
        return _invalid("non-live state; caller must use the terminal boundary")
    knight, boss = _box(state.get("knight_box")), _box(state.get("boss_box"))
    knight_velocity, boss_velocity = _velocity(state.get("knight_velocity")), _velocity(state.get("boss_velocity"))
    hazards = state.get("hazards")
    if (knight is None or boss is None or knight_velocity is None or boss_velocity is None
            or not isinstance(hazards, (tuple, list))):
        return _invalid("missing or invalid geometry/velocity")
    geometry = set()
    has_attached_body = False
    for hazard in hazards:
        hazard_box = _box(hazard.get("box") if isinstance(hazard, dict) else getattr(hazard, "box", None))
        hazard_velocity = _velocity(hazard.get("velocity") if isinstance(hazard, dict)
                                    else getattr(hazard, "velocity", None))
        if hazard_box is None or hazard_velocity is None:
            return _invalid("missing or invalid hazard geometry/velocity")
        attached = (hazard.get("boss_attached") if isinstance(hazard, dict)
                    else getattr(hazard, "boss_attached", False))
        has_attached_body = has_attached_body or attached is True
        hazard_type = (hazard.get("hazard_type") if isinstance(hazard, dict)
                       else getattr(hazard, "hazard_type", None))
        if not isinstance(hazard_type, int):
            hazard_type = None
        geometry.add((hazard_box, hazard_velocity, hazard_type))
    if not has_attached_body:
        # Keep body contact represented even during a missing hazard snapshot,
        # but never guess its damage category to grant shadow protection.
        geometry.add((boss, boss_velocity, None))
    earliest = math.inf
    for hazard_box, hazard_velocity, hazard_type in geometry:
        relative = tuple(hazard_velocity[i] - knight_velocity[i] for i in (0, 1))
        interval = swept_aabb_interval(knight, hazard_box, relative)
        if interval is None:
            continue
        entry, exit_time = interval
        protection = _protected_until(state, hazard_type, config)
        # Leaving the contact box before protection ends resolves this threat.
        # Remaining inside at expiry is a future unprotected collision.
        if protection > 0 and exit_time <= protection:
            continue
        exposure = max(entry, protection)
        if exposure <= config.ttc_horizon:
            earliest = min(earliest, exposure)
    safety = -(1.0 - min(earliest / config.ttc_horizon, 1.0))

    horizontal_gap = max(knight[0] - boss[2], boss[0] - knight[2], 0.0)
    deviation = max(config.range_min_gap - horizontal_gap,
                    horizontal_gap - config.range_max_gap, 0.0)
    range_value = -min(deviation / config.range_distance_scale, 1.0)
    # A horizontal attack-distance heuristic is not meaningful while jumping
    # clearly above/below the target. Current safety also softens approach
    # pressure while the knight is at imminent collision risk.
    vertical_overlap = min(knight[3], boss[3]) - max(knight[1], boss[1])
    phase_multiplier = (0.0 if phase == BossPhase.UNKNOWN else
                        1.0 if phase in (BossPhase.NEUTRAL, BossPhase.RECOVERY)
                        else config.attack_phase_range_multiplier)
    range_multiplier = phase_multiplier * max(0.0, 1.0 + safety) if vertical_overlap > 0 else 0.0
    value = config.range_weight * range_multiplier * range_value + config.safety_weight * safety
    return PotentialResult(value, range_value, safety, True,
                           None if math.isinf(earliest) else earliest)


def potential_reward(previous_state, current_state, terminal=False, config=None):
    """Return eta*(gamma*Phi(next)-Phi(previous)), or None for missing data.

    True task endings (victory, death and the configured hard cutoff) have
    next potential 0. A partial rollout is not a terminal. Negative shaping
    terms are intentionally preserved; clipping them would break telescoping.
    """
    config = config or PotentialConfig()
    previous = potential(previous_state, config)
    if not previous.valid:
        return None
    if terminal:
        next_value = 0.0
    else:
        current = potential(current_state, config)
        if not current.valid:
            return None
        next_value = current.value
    return config.eta * (config.gamma * next_value - previous.value)
