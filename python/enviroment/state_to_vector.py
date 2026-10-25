"""Raw game-state snapshots and Pure Vessel's relative observation vector."""

import game_state as game_reader
from agent.action import boss_attack, boss_move, PureVesselMove
from agent.calc import box_center, box_distance

game_reader.start()


TRACKED_HAZARDS = 8
HAZARD_FEATURES = 8
BASE_FEATURES = 55
OBSERVATION_SIZE = BASE_FEATURES + TRACKED_HAZARDS * HAZARD_FEATURES
MAX_OFFSET = 50.0
NO_HAZARD_DISTANCE = 50.0


class HazardSide:
    NONE = -1
    RIGHT = 0
    LEFT = 1
    SAME = 2


def _clip(value, limit=MAX_OFFSET):
    return max(-limit, min(limit, value))


def _move_one_hot(move):
    one_hot = [0] * len(PureVesselMove)
    one_hot[int(move)] = 1
    return one_hot


def _relative_box_features(reference_box, target_box):
    """Target centre offset and size, plus a presence bit."""
    if reference_box is None or target_box is None:
        return [0.0, 0.0, 0.0, 0.0, 0]
    rx, ry = box_center(reference_box)
    tx, ty = box_center(target_box)
    min_x, min_y, max_x, max_y = target_box
    return [
        _clip(tx - rx),
        _clip(ty - ry),
        max_x - min_x,
        max_y - min_y,
        1,
    ]


def _sorted_hazards(knight_box, hazards):
    # Edge distance, rather than first-seen order or centre distance, makes
    # large sword pillars rank by how soon they can actually touch the knight.
    return sorted(hazards, key=lambda hazard: box_distance(knight_box, hazard.box))


def _hazard_features(knight_box, knight_velocity, hazards):
    kx, ky = box_center(knight_box)
    kvx, kvy = knight_velocity
    features = []
    ordered = _sorted_hazards(knight_box, hazards)

    for index in range(TRACKED_HAZARDS):
        if index >= len(ordered):
            features.extend((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0))
            continue

        hazard = ordered[index]
        hx, hy = box_center(hazard.box)
        min_x, min_y, max_x, max_y = hazard.box
        hvx, hvy = hazard.velocity
        features.extend((
            _clip(hx - kx),
            _clip(hy - ky),
            max_x - min_x,
            max_y - min_y,
            _clip(hvx - kvx),
            _clip(hvy - kvy),
            int(hazard.boss_attached),
            1,
        ))

    return features


def _directional_hazard_gaps(knight_box, hazards):
    """Nearest edge gap on the knight's left, right, above and below."""
    if not hazards:
        return [NO_HAZARD_DISTANCE] * 4

    kx, ky = box_center(knight_box)
    left = right = above = below = NO_HAZARD_DISTANCE
    kmin_x, kmin_y, kmax_x, kmax_y = knight_box

    for hazard in hazards:
        hmin_x, hmin_y, hmax_x, hmax_y = hazard.box
        hx, hy = box_center(hazard.box)
        if hx <= kx:
            left = min(left, max(kmin_x - hmax_x, 0.0))
        if hx >= kx:
            right = min(right, max(hmin_x - kmax_x, 0.0))
        if hy >= ky:
            above = min(above, max(hmin_y - kmax_y, 0.0))
        if hy <= ky:
            below = min(below, max(kmin_y - hmax_y, 0.0))

    return [left, right, above, below]


def nearest_hazard_distance(knight_box, hazard_boxes):
    if not hazard_boxes:
        return NO_HAZARD_DISTANCE
    return min(box_distance(knight_box, hazard_box) for hazard_box in hazard_boxes)


def relative_horizontal_position(state):
    if not state["hazards"]:
        return HazardSide.NONE
    nearest = _sorted_hazards(state["knight_box"], state["hazards"])[0]
    knight_x, _ = box_center(state["knight_box"])
    hazard_x, _ = box_center(nearest.box)
    if knight_x < hazard_x:
        return HazardSide.LEFT
    if knight_x > hazard_x:
        return HazardSide.RIGHT
    return HazardSide.SAME


def on_ground(state):
    return state["is_on_ground"]


def get_state():
    return {
        "knight_box": game_reader.knight_box,
        "attacking": game_reader.attacking,
        "attack_box": game_reader.attack_box,
        "hit_count": game_reader.hit_count - game_reader.hit_count_baseline,
        "damage_count": game_reader.damage_count - game_reader.damage_count_baseline,
        "hit_streak": game_reader.hit_streak,
        "hazards": game_reader.hazards,
        "hazard_boxes": game_reader.hazard_boxes,
        "knight_velocity": game_reader.knight_velocity,
        "knight_facing_right": game_reader.knight_facing_right,
        "boss_health": game_reader.boss_health,
        "boss_facing_left": game_reader.boss_facing_left,
        "boss_velocity": game_reader.boss_velocity,
        "boss_position": game_reader.boss_position,
        "boss_box": game_reader.boss_box,
        "boss_control_state": game_reader.boss_control_state,
        "wall_distance_left": game_reader.wall_distance_left,
        "wall_distance_right": game_reader.wall_distance_right,
        "knight_health": game_reader.knight_health,
        "mana": game_reader.mana,
        "is_dashing": game_reader.is_dashing,
        "is_jumping": game_reader.is_jumping,
        "is_falling": game_reader.is_falling,
        "is_on_ground": game_reader.is_on_ground,
        "is_recoiling": game_reader.is_recoiling,
        "is_touching_wall_left": game_reader.is_touching_wall_left,
        "is_touching_wall_right": game_reader.is_touching_wall_right,
        "can_shade_cloak_now": game_reader.can_shade_cloak_now,
        "is_invuln": game_reader.is_invuln,
    }


def state_to_vector(state, time_frac=0.0):
    """Build the 119-feature, mostly knight-relative Pure Vessel state."""
    knight_box = state["knight_box"]
    if knight_box is None:
        return None

    knight_velocity = state["knight_velocity"]
    kvx, kvy = knight_velocity
    boss_present = state["boss_facing_left"] is not None
    kx, ky = box_center(knight_box)

    if boss_present:
        boss_hp_obs = state["boss_health"] if state["boss_health"] >= 0 else 0
        bx, by = state["boss_position"]
        boss_dx, boss_dy = _clip(bx - kx), _clip(by - ky)
        bvx, bvy = state["boss_velocity"]
        boss_rel_vx, boss_rel_vy = _clip(bvx - kvx), _clip(bvy - kvy)
        if state["boss_box"] is not None:
            bmin_x, bmin_y, bmax_x, bmax_y = state["boss_box"]
            boss_width, boss_height = bmax_x - bmin_x, bmax_y - bmin_y
        else:
            boss_width = boss_height = 0.0
        boss_facing_left = int(state["boss_facing_left"])
        boss_faces_knight = int((boss_dx < 0 and boss_facing_left)
                                or (boss_dx > 0 and not boss_facing_left))
        knight_faces_boss = int((boss_dx > 0 and state["knight_facing_right"])
                                or (boss_dx < 0 and not state["knight_facing_right"]))
    else:
        boss_hp_obs = 0
        boss_dx = boss_dy = boss_rel_vx = boss_rel_vy = 0.0
        boss_width = boss_height = 0.0
        boss_facing_left = boss_faces_knight = knight_faces_boss = 0

    hazards = state["hazards"]
    nearest = (min(box_distance(knight_box, hazard.box) for hazard in hazards)
               if hazards else NO_HAZARD_DISTANCE)

    vector = [
        # Knight state; no absolute world position.
        state["knight_health"], state["mana"], int(state["knight_facing_right"]),
        kvx, kvy, int(state["attacking"]), int(state["is_dashing"]),
        int(state["is_jumping"]), int(state["is_falling"]),
        int(state["is_recoiling"]), int(state["is_on_ground"]),
        int(state["is_touching_wall_left"]), int(state["is_touching_wall_right"]),
        min(max(state["wall_distance_left"], 0.0), MAX_OFFSET),
        min(max(state["wall_distance_right"], 0.0), MAX_OFFSET),
        int(state["can_shade_cloak_now"]), int(state["is_invuln"]),

        # Nail box relative to the knight.
        *_relative_box_features(knight_box, state["attack_box"]),

        # Boss relative to the knight.
        int(boss_present), boss_hp_obs, boss_dx, boss_dy,
        boss_rel_vx, boss_rel_vy, boss_width, boss_height,
        boss_facing_left, boss_faces_knight, knight_faces_boss,

        # Fight progress.
        state["hit_count"], state["damage_count"], time_frac,

        # Global hazard summary.
        len(hazards), nearest,
        *_directional_hazard_gaps(knight_box, hazards),
        int(boss_attack(state)),

        # Move one-hot (PureVesselMove, 12 dims).
        *_move_one_hot(boss_move(state)),

        # Eight closest hazards, each in knight-relative coordinates.
        *_hazard_features(knight_box, knight_velocity, hazards),
    ]

    if len(vector) != OBSERVATION_SIZE:
        raise RuntimeError(
            f"observation width {len(vector)} != expected {OBSERVATION_SIZE}"
        )
    return vector


def is_done():
    # Boss is "defeated" only at an exact hp of 0. -1 means "no boss present"
    # right now and must NOT end the episode.
    return game_reader.boss_health == 0 or game_reader.knight_health <= 0
