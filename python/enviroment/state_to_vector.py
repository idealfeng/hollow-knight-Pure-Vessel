"""Reading the game: the raw state dict, and the flat observation vector the
network actually consumes.

get_state() snapshots the globals the game_state polling thread keeps up to
date. state_to_vector() flattens that snapshot into the fixed-width list of
numbers the policy sees - its length is STATE_SIZE in main.py, and its field
order is mirrored by FEATURE_NAMES in state_vector_viewer.py.
"""

import game_state as game_reader
from agent.action import boss_attack
from agent.calc import box_center, nearest_hazard_distance, relative_offsets

game_reader.start()


# Hazard and attack box slots are zero-filled when the game reports nothing
# there, so the vector keeps a fixed width whatever is on screen.
EMPTY_BOX = (0.0, 0.0, 0.0, 0.0)

# How many hazard boxes get their own slots in the observation. The boss can
# briefly have more; the extras are visible to nearest_hazard_distance() but
# not to the per-box features.
TRACKED_HAZARDS = 2

# The arena floor. The knight's box sits at exactly this y while grounded,
# which is cheaper (and matches the mod's numbers better) than inferring it
# from the velocity.
GROUND_Y = 27.015


class HazardSide:
    """Which side of the nearest hazard the knight is on. These values go
    straight into the observation vector, so don't renumber them."""

    NONE = -1
    RIGHT = 0
    LEFT = 1
    SAME = 2


def attack_box_features(attack_box):
    """The knight's attack hitbox, or a zero box when it isn't swinging."""
    return list(attack_box) if attack_box else list(EMPTY_BOX)


def hazard_box_features(hazards):
    """The first TRACKED_HAZARDS hazard boxes flattened into one list,
    zero-padded so the width is always TRACKED_HAZARDS * 4."""
    features = []
    for i in range(TRACKED_HAZARDS):
        features.extend(hazards[i] if i < len(hazards) else EMPTY_BOX)
    return features


def boss_facing_flag(boss_facing_left):
    """1 when the boss faces left, 0 when it faces right OR when no boss was
    found at all (game_state uses None for that)."""
    return int(boss_facing_left is True)


def on_ground(state):
    return state["knight_box"][1] == GROUND_Y


def relative_horizontal_position(state):
    """Which side of the first hazard the knight is on - see HazardSide."""
    if not state["hazard_boxes"]:
        return HazardSide.NONE

    knight_x, _ = box_center(state["knight_box"])
    hazard_x, _ = box_center(state["hazard_boxes"][0])

    if knight_x < hazard_x:
        return HazardSide.LEFT
    if knight_x > hazard_x:
        return HazardSide.RIGHT
    return HazardSide.SAME


def get_state():
    return {
        "knight_box": game_reader.knight_box,
        "attacking": game_reader.attacking,
        "attack_box": game_reader.attack_box,
        # Per-episode, not lifetime - the mod's raw counters never reset on
        # their own (see game_state.reset_episode_counters()).
        "hit_count": game_reader.hit_count - game_reader.hit_count_baseline,
        "damage_count": game_reader.damage_count - game_reader.damage_count_baseline,
        # Consecutive hits without taking damage. Reward-only - deliberately
        # NOT added to state_to_vector(), which would change STATE_SIZE.
        "hit_streak": game_reader.hit_streak,
        "hazard_boxes": game_reader.hazard_boxes,
        "knight_velocity": game_reader.knight_velocity,
        "knight_facing_right": game_reader.knight_facing_right,
        "boss_health": game_reader.boss_health,
        "boss_facing_left": game_reader.boss_facing_left,
        "boss_velocity": game_reader.boss_velocity,
        "knight_health": game_reader.knight_health,
        "mana": game_reader.mana,
        "is_dashing": game_reader.is_dashing,
        "is_jumping": game_reader.is_jumping,
        "is_falling": game_reader.is_falling,
        "is_recoiling": game_reader.is_recoiling,
        "is_touching_wall_left": game_reader.is_touching_wall_left,
        "is_touching_wall_right": game_reader.is_touching_wall_right,
        "can_shade_cloak_now": game_reader.can_shade_cloak_now,
        "is_invuln": game_reader.is_invuln,
    }


def state_to_vector(state, time_frac=0.0):
    """Flatten a state dict into the network's input vector, or None if the
    game has not reported a knight yet.

    `time_frac` is how far the current episode has run against its time
    limit: 0.0 at the start, 1.0 at the cap. Without it the episode's forced
    ending is invisible to the agent, so the critic cannot tell a state at
    t=5s from an identical-looking one at t=59s even though their true
    returns differ enormously. Callers pass their own clock (see main.py) -
    this function has no episode notion of its own.

    Field order is mirrored by FEATURE_NAMES in state_vector_viewer.py; keep
    the two in step.
    """
    knight_box = state["knight_box"]
    if knight_box is None:
        return None

    return [
        # --- knight ---
        *knight_box,
        state["knight_health"],
        state["mana"],
        int(state["knight_facing_right"]),
        *state["knight_velocity"],
        *attack_box_features(state["attack_box"]),
        int(state["attacking"]),

        # --- hazards / boss ---
        *hazard_box_features(state["hazard_boxes"]),
        state["boss_health"],
        boss_facing_flag(state["boss_facing_left"]),
        *state["boss_velocity"],

        # --- fight progress ---
        state["hit_count"],
        state["damage_count"],

        # --- derived features ---
        # Everything above is absolute world coordinates; these give the
        # network the relationships it would otherwise have to reconstruct
        # from separately normalised absolutes.
        boss_attack(state),
        nearest_hazard_distance(knight_box, state["hazard_boxes"]),
        relative_horizontal_position(state),

        # --- knight movement/action state (debug-mod style flags) ---
        int(state["is_dashing"]),
        int(state["is_jumping"]),
        int(state["is_falling"]),
        int(state["is_recoiling"]),
        int(state["is_touching_wall_left"]),
        int(state["is_touching_wall_right"]),
        int(state["can_shade_cloak_now"]),
        int(state["is_invuln"]),

        # --- episode clock ---
        time_frac,

        # --- knight-relative offsets to hazard 1 and hazard 2 ---
        *relative_offsets(knight_box, state["hazard_boxes"]),
    ]


def is_done():
    return game_reader.boss_health <= 0 or game_reader.knight_health <= 0
