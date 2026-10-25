"""The reward function: one number per control step, from the transition
between two consecutive game states.

Everything the agent is being asked to do lives here. Pass
return_breakdown=True to also get the list of (label, amount) terms that
fired - test_gui_client.py uses it to show which term produced what.
"""

from agent.action import boss_attack

# ---- landing hits ----
# A hit is worth more the longer the current damage-free streak, up to a cap:
# 4.0 for a lone hit, 10.0 for the fifth and beyond of an unbroken chain.
HIT_BASE = 4.0
HIT_STREAK_STEP = 1.5
HIT_STREAK_CAP = 4          # max bonus steps -> peak 4.0 + 4*1.5 = 10.0 per hit

# ---- staggers ----
# Boss health thresholds where the fight's scripted stagger phases trigger.
# Worth far more than the hits themselves: they are the checkpoints that mark
# real progress through the fight.
FIRST_STAGGER_HEALTH = 537
SECOND_STAGGER_HEALTH = 339
FIRST_STAGGER = 60
SECOND_STAGGER = 120

BOSS_DEFEATED = 100
DODGE = 2

# ---- costs ----
TIME_PENALTY = 0.02         # charged every step, to discourage stalling
DAMAGE_PER_MASK = 0.3       # scaled by masks lost
NON_ATTACK_HIT = -6         # taking damage with no boss attack out at all
DEATH = -20


def calculate_reward(previous_state, current_state, done, timed_out=False,
                     return_breakdown=False):
    current_attack = boss_attack(current_state)
    previous_attack = boss_attack(previous_state)

    reward = 0
    breakdown = []

    def add(amount, label):
        nonlocal reward
        reward += amount
        breakdown.append((label, amount))

    # ==================================================================
    # REWARDS
    # ==================================================================

    # Damage dealt to the boss, scaled by the damage-free hit streak.
    if current_state["boss_health"] < previous_state["boss_health"]:
        streak_steps = min(max(current_state["hit_streak"] - 1, 0), HIT_STREAK_CAP)
        add(HIT_BASE + streak_steps * HIT_STREAK_STEP, "hit landed")

    # Staggers - awarded on the step the threshold is crossed downwards.
    if previous_state["boss_health"] > FIRST_STAGGER_HEALTH >= current_state["boss_health"]:
        add(FIRST_STAGGER, "first stagger")

    if previous_state["boss_health"] > SECOND_STAGGER_HEALTH >= current_state["boss_health"]:
        add(SECOND_STAGGER, "second stagger")

    if done and current_state["boss_health"] <= 0:
        add(BOSS_DEFEATED, "boss defeated")

    # An attack that was live last step is over now and cost us no health -
    # i.e. the knight got out of the way rather than tanking it.
    if (not current_attack.is_active
            and previous_attack.is_active
            and current_state["knight_health"] == previous_state["knight_health"]):
        add(DODGE, "dodge an attack")

    # ==================================================================
    # PENALTIES
    # ==================================================================

    # Standing cost of time. Not charged on the step the cap ends the episode,
    # so a scheduled ending stays free of surprise charges.
    if not timed_out:
        add(-TIME_PENALTY, "time")

    if not timed_out:
        masks_lost = previous_state["knight_health"] - current_state["knight_health"]
        if masks_lost > 0:
            add(-masks_lost * DAMAGE_PER_MASK, "took damage")

            # Hit while no attack was out either before or after: not a
            # mistimed dodge, just standing in something avoidable.
            if not current_attack.is_active and not previous_attack.is_active:
                add(NON_ATTACK_HIT, "hit by non-attack")

    if done and current_state["knight_health"] <= 0 and not timed_out:
        add(DEATH, "die")

    if return_breakdown:
        return reward, breakdown

    return reward
