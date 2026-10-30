"""Conservative, observable wave-front crossing events.

This module never chooses an action. A caller supplies a classifier calibrated
against captured game trajectories; an arbitrary detached hazard is not assumed
to be a Traitor Lord wave. Disappearance and animation completion alone cannot
establish a successful pass.
"""
from dataclasses import dataclass
import math

from agent.action import BossMove, BossPhase, boss_move, boss_phase


def _field(hazard, name):
    return hazard.get(name) if isinstance(hazard, dict) else getattr(hazard, name)


def _valid_box(box):
    return (box is not None and len(box) == 4
            and all(math.isfinite(float(value)) for value in box)
            and box[0] < box[2] and box[1] < box[3])


def _vertical_overlap(first, second):
    return min(first[3], second[3]) > max(first[1], second[1])


def _side_gap(knight, hazard):
    if hazard[0] > knight[2]:
        return 1, hazard[0] - knight[2]
    if hazard[2] < knight[0]:
        return -1, knight[0] - hazard[2]
    return 0, 0.0


@dataclass(frozen=True)
class WaveFrontCriteria:
    """All bounds are explicit, to prevent using unverified default geometry."""

    minimum_height: float
    maximum_width: float
    minimum_horizontal_speed: float
    maximum_vertical_speed: float
    maximum_spawn_gap: float

    def __call__(self, hazard, state):
        if _field(hazard, "boss_attached"):
            return False
        box, boss = _field(hazard, "box"), state.get("boss_box")
        velocity = _field(hazard, "velocity")
        if not _valid_box(box) or not _valid_box(boss) or velocity is None:
            return False
        if len(velocity) != 2 or not all(math.isfinite(float(v)) for v in velocity):
            return False
        spawn_gap = max(box[0] - boss[2], boss[0] - box[2], 0.0)
        return (box[3] - box[1] >= self.minimum_height
                and box[2] - box[0] <= self.maximum_width
                and abs(velocity[0]) >= self.minimum_horizontal_speed
                and abs(velocity[1]) <= self.maximum_vertical_speed
                and spawn_gap <= self.maximum_spawn_gap
                and _vertical_overlap(box, boss))


@dataclass
class _Front:
    instance_id: int
    initial_side: int
    last_gap: float
    initial_center: float
    motion_verified: bool = False
    currently_closing: bool = False
    samples: int = 0
    approaching: bool = False
    threatened: bool = False
    cleared: bool = False
    missing: bool = False


class WaveClearanceTracker:
    """Observe one safe crossing per move, with a per-episode reward cap.

    ``transition`` returns a diagnostic event only when a pass is confirmed.
    ``last_diagnostic`` exposes pending fronts and rejected cycles for trace
    audits. Call ``reset`` whenever a fresh episode/scene starts.
    """

    def __init__(self, classifier, *, threat_gap, clear_gap, confirmation_steps=2,
                 maximum_rewards=2, minimum_horizontal_travel=0.5):
        if not callable(classifier):
            raise TypeError("A trajectory-calibrated wave classifier is required")
        if (threat_gap <= 0 or clear_gap < 0 or confirmation_steps < 1
                or maximum_rewards < 1 or minimum_horizontal_travel < 0):
            raise ValueError("Invalid wave clearance limits")
        self.classifier = classifier
        self.threat_gap = float(threat_gap)
        self.clear_gap = float(clear_gap)
        self.confirmation_steps = int(confirmation_steps)
        self.maximum_rewards = int(maximum_rewards)
        self.minimum_horizontal_travel = float(minimum_horizontal_travel)
        self.reset()

    def reset(self):
        self.reward_count = 0
        self.completed_passes = 0
        self.cycle_count = 0
        self._cycle = None
        self._in_wave = False
        self._step = 0
        self.last_diagnostic = {"event": "reset"}

    @staticmethod
    def _wave_window(state):
        return (boss_move(state) == BossMove.SHOCKWAVE
                and boss_phase(state) in (BossPhase.WINDUP, BossPhase.ACTIVE))

    def _diagnostic(self, event, **values):
        cycle = self._cycle
        self.last_diagnostic = {
            "event": event, "step": self._step,
            "cycle": cycle["number"] if cycle else None,
            "reward_count": self.reward_count,
            "fronts": [vars(front).copy() for front in cycle["fronts"].values()] if cycle else [],
            **values,
        }
        return self.last_diagnostic

    def transition(self, previous, current, *, done=False, timed_out=False, truncated=False):
        self._step += 1
        current_window = self._wave_window(current)
        previous_window = self._wave_window(previous)
        fresh_window = (previous_window or current_window) and not self._in_wave
        self._in_wave = current_window

        invalid = (done or timed_out or truncated
                   or current.get("knight_health", 0) <= 0
                   or current.get("boss_health", -1) <= 0
                   or not _valid_box(previous.get("knight_box"))
                   or not _valid_box(current.get("knight_box")))
        if invalid:
            self._diagnostic("discarded_terminal_or_missing_state")
            self._cycle = None
            return None

        damaged = (current["knight_health"] < previous["knight_health"]
                   or current.get("damage_count", 0) != previous.get("damage_count", 0))
        if fresh_window:
            self.cycle_count += 1
            self._cycle = {
                "number": self.cycle_count,
                "seen": {_field(h, "instance_id") for h in previous.get("hazards", [])},
                "fronts": {}, "admission": {}, "invalid": False, "confirmation": 0,
            }
        cycle = self._cycle
        if cycle is None:
            self._diagnostic("idle")
            return None
        if damaged:
            cycle["invalid"] = True
        if cycle["invalid"]:
            self._diagnostic("discarded_damage_or_ambiguous_front")
            if not current_window:
                self._cycle = None
            return None

        hazards = {_field(h, "instance_id"): h for h in current.get("hazards", [])}
        knight = current["knight_box"]
        fronts = cycle["fronts"]
        can_spawn = current.get("boss_control_state") in ("Slamming", "Waves")
        for identifier, hazard in hazards.items():
            if identifier not in cycle["seen"] and can_spawn:
                cycle["admission"][identifier] = self._step
            born = cycle["admission"].get(identifier)
            if (identifier not in fronts and born is not None and self._step - born <= 3
                    and can_spawn and self.classifier(hazard, current)):
                side, gap = _side_gap(knight, _field(hazard, "box"))
                # A collider first seen already overlapping has no observable
                # approach; it may be logged, but cannot earn a crossing bonus.
                box = _field(hazard, "box")
                fronts[identifier] = _Front(identifier, side, gap, (box[0] + box[2]) / 2)
            cycle["seen"].add(identifier)

        for identifier, front in fronts.items():
            hazard = hazards.get(identifier)
            if hazard is None:
                front.missing = True
                front.currently_closing = False
                if front.threatened and not front.cleared:
                    cycle["invalid"] = True
                continue
            box = _field(hazard, "box")
            if not _valid_box(box):
                cycle["invalid"] = True
                continue
            side, gap = _side_gap(knight, box)
            center = (box[0] + box[2]) / 2
            if abs(center - front.initial_center) >= self.minimum_horizontal_travel:
                front.motion_verified = True
            vx = _field(hazard, "velocity")[0] - current.get("knight_velocity", (0, 0))[0]
            vertical = _vertical_overlap(knight, box)
            closing = (side == front.initial_side
                       and (gap < front.last_gap - 1e-6 if front.samples else side * vx < 0))
            front.currently_closing = bool(closing and vertical)
            if front.initial_side and vertical and closing:
                front.approaching = True
                if gap <= self.threat_gap:
                    front.threatened = True
            if (front.initial_side and front.approaching and vertical
                    and side != front.initial_side):
                # A fast front can cross the entire near interval between two
                # 10 Hz samples. Same-ID side crossing still demonstrates the
                # threat; demanding an overlapping sample would miss it.
                front.threatened = True
            # Re-evaluate live geometry: the knight can reverse after a pass.
            # A historical cleared latch must not award while the same front
            # is overlapping again during dash protection.
            front.cleared = bool(front.initial_side and front.approaching
                                 and side == -front.initial_side and gap >= self.clear_gap)
            front.last_gap = gap
            front.samples += 1

        if cycle["invalid"]:
            self._diagnostic("discarded_damage_or_ambiguous_front")
            return None
        meaningful = [front for front in fronts.values() if front.threatened and front.motion_verified]
        # A symmetric front moving away may briefly get closer because the
        # knight dashes after it. It must not remain a pending threat forever
        # once it is receding without ever entering the near interval.
        pending = [front for front in fronts.values() if not front.cleared
                   and (front.threatened or front.currently_closing)]
        known_post_phase = (not current_window
                            and boss_phase(current) in (BossPhase.NEUTRAL, BossPhase.WINDUP,
                                                        BossPhase.RECOVERY))
        safe_pass = bool(meaningful and all(front.cleared for front in meaningful) and not pending)
        if safe_pass:
            cycle["confirmation"] += 1
        else:
            cycle["confirmation"] = 0
        if safe_pass and known_post_phase and cycle["confirmation"] >= self.confirmation_steps:
            self.completed_passes += 1
            reward_eligible = self.reward_count < self.maximum_rewards
            if reward_eligible:
                self.reward_count += 1
            result = self._diagnostic("clean_wave_pass", reward_eligible=reward_eligible,
                                      passed_ids=[front.instance_id for front in meaningful],
                                      confirmation_steps=cycle["confirmation"])
            self._cycle = None
            return result
        self._diagnostic("pending", confirmation_steps=cycle["confirmation"])
        return None
