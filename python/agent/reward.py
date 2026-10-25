"""The reward function: one number per control step, from the transition
between two consecutive game states.

Everything the agent is being asked to do lives here. Pass
return_breakdown=True to also get the list of (label, amount) terms that
fired - test_gui_client.py uses it to show which term produced what.
"""

from agent.action import boss_move, PureVesselMove

# ---- landing hits ----
# A hit remains useful shaping, but it must not outweigh taking a two-mask
# Pure Vessel hit. A clean streak grows from 1.0 to at most 2.0 per nail hit.
HIT_BASE = 1.0
HIT_STREAK_STEP = 0.25
HIT_STREAK_CAP = 4

# ---- staggers ----
# Reward the observed Control FSM entering its real stun state rather than
# duplicating hundreds of points at approximate HP thresholds.
STAGGER = 10.0

BOSS_DEFEATED = 100.0

# ---- costs ----
TIME_PENALTY_PER_SECOND = 0.15
DAMAGE_PER_MASK = 4.0
DEATH = -40.0


def calculate_reward(previous_state, current_state, done, timed_out=False,
                     return_breakdown=False, step_dt=1.0 / 15.0):
    current_move = boss_move(current_state)
    previous_move = boss_move(previous_state)

    reward = 0
    breakdown = []

    def add(amount, label):
        nonlocal reward
        reward += amount
        breakdown.append((label, amount))

    # ==================================================================
    # REWARDS
    # ==================================================================

    prev_hp = previous_state["boss_health"]
    cur_hp = current_state["boss_health"]
    # hp == -1 means "boss missing" (not defeated). Guard every boss-hp-based
    # term so a transiently absent boss can't fake a hit, stagger or defeat.
    hp_valid = cur_hp >= 0 and prev_hp >= 0

    # Damage dealt to the boss, scaled by the damage-free hit streak.
    if hp_valid and cur_hp < prev_hp:
        streak_steps = min(max(current_state["hit_streak"] - 1, 0), HIT_STREAK_CAP)
        add(HIT_BASE + streak_steps * HIT_STREAK_STEP, "hit landed")

    if (hp_valid
            and current_move == PureVesselMove.STUN
            and previous_move != PureVesselMove.STUN):
        add(STAGGER, "boss staggered")

    # Defeat is a confirmed hp of exactly 0 - not a missing boss object.
    if done and prev_hp > 0 and cur_hp == 0:
        add(BOSS_DEFEATED, "boss defeated")

    # ==================================================================
    # PENALTIES
    # ==================================================================

    # Standing cost of time. Not charged on the step the cap ends the episode,
    # so a scheduled ending stays free of surprise charges.
    if not timed_out:
        add(-TIME_PENALTY_PER_SECOND * max(step_dt, 0.0), "time")

    if not timed_out:
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
