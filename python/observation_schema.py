"""Explicit observation layouts; old policies keep their original 118 columns."""

LEGACY_SCHEMA = "legacy_v1"
COMBAT_SCHEMA = "combat_v2"
LEGACY_FRAME_WIDTH = 118
DASH_FEATURES = (
    "dash_state_valid", "can_dash_now", "is_shadow_dashing",
    "shadow_cooldown_remaining", "shadow_dash_remaining", "dash_remaining",
    "invulnerability_remaining", "invulnerability_time_valid",
)
COMBAT_EXTRA_FEATURES = DASH_FEATURES + ("knight_width", "knight_height")
COMBAT_FRAME_WIDTH = LEGACY_FRAME_WIDTH + len(COMBAT_EXTRA_FEATURES)


def observation_width(schema=LEGACY_SCHEMA):
    if schema == LEGACY_SCHEMA:
        return LEGACY_FRAME_WIDTH
    if schema == COMBAT_SCHEMA:
        return COMBAT_FRAME_WIDTH
    raise ValueError(f"unknown observation schema: {schema!r}")
