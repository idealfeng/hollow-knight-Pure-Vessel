"""Reward shaping for the Pure Vessel environment.

``calculate_reward`` scores one state transition. ``RewardTracker`` adds the
small amount of history required to decide whether a nail swing really missed
and whether an ACTIVE boss phase ended without damage.
"""

from agent.action import BossMove, BossPhase, boss_move, boss_phase
from agent.calc import box_distance

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


def calculate_reward(previous_state, current_state, done, timed_out=False,
                     return_breakdown=False, step_dt=1.0 / 10.0,
                     missed_attack=False, successful_dodge=False):
    """Score one transition.

    Delayed event flags normally come from :class:`RewardTracker`. They default
    to false so offline transition tests can still call this function directly.
    """
    current_move = boss_move(current_state)
    previous_move = boss_move(previous_state)

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
        add(damage_dealt / NAIL_DAMAGE_REFERENCE * HIT_REWARD_PER_NAIL,
            "boss damage")

    if done and prev_hp > 0 and cur_hp == 0:
        add(BOSS_DEFEATED, "boss defeated")

    if successful_dodge:
        add(CLEAN_DODGE, "clean dodge")

    if missed_attack:
        add(EMPTY_ATTACK, "empty attack")

    if not timed_out:
        dt = max(step_dt, 0.0)
        add(-TIME_PENALTY_PER_SECOND * dt, "time")

        if boss_phase(current_state) in _APPROACH_PHASES:
            distance = _boss_distance(current_state)
            if distance is not None and distance > FAR_DISTANCE:
                span = max(FAR_FULL_DISTANCE - FAR_DISTANCE, 1e-6)
                severity = min((distance - FAR_DISTANCE) / span, 1.0)
                add(-FAR_PENALTY_PER_SECOND * dt * severity,
                    "far from boss")

        masks_lost = previous_state["knight_health"] - current_state["knight_health"]
        if masks_lost > 0:
            add(-masks_lost * DAMAGE_PER_MASK, "took damage")

    if (done
            and previous_state["knight_health"] > 0
            and current_state["knight_health"] <= 0
            and not timed_out):
        add(DEATH, "die")

    if return_breakdown:
        return reward, breakdown
    return reward


class RewardTracker:
    """Recognise delayed combat events and expose episode diagnostics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._swing_hits = None
        self._swing_hp = None
        self._active_damage = None
        self.misses = 0
        self.staggers = 0
        self.dodges = 0
        self.last_breakdown = []

    def transition(self, previous_state, current_state, done,
                   timed_out=False, step_dt=1.0 / 10.0,
                   return_breakdown=False):
        missed_attack = False
        successful_dodge = False

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
        )
        return (reward, self.last_breakdown) if return_breakdown else reward
