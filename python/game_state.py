"""Background reader for the game's state.

start() spins up a thread that polls the mod over TCP and republishes each
frame into this module's globals. Everything else in the project reads those
globals directly (see enviroment.get_state), so there is exactly one socket
and one parser no matter how many consumers there are.
"""

import socket
import threading
import time
from dataclasses import dataclass

HOST = "127.0.0.1"
PORT = 9000
RECONNECT_DELAY = 0.5   # seconds to wait before retrying a failed connect
SOCKET_TIMEOUT = 2.0    # give up on a silent server rather than blocking forever

# The loop below had no throttling at all - it round-tripped sendall/recv on
# a loopback socket as fast as the OS and the mod's lock could return one,
# almost certainly several hundred Hz or more. Nothing that reads these
# globals needs anywhere close to that: main.py's control loop runs at 15 Hz
# (CONTROL_HZ), the fastest UI consumer (test_gui_client.py) ticks every
# 30 ms (~33 Hz). 100 Hz is comfortably above every real consumer while
# cutting out most of the wasted round trips - each one also taking a lock
# on the mod's side that Update() needs every rendered frame, so hammering
# this needlessly competes for CPU with the actual game, which is the real
# bottleneck (measured elsewhere in this project at ~98% of wall-clock time).
POLL_INTERVAL = 0.01


# ======================================================================
# PUBLISHED STATE
#
# Rewritten wholesale by the polling thread every frame. Readers take them as
# a best-effort snapshot - there is no lock, because every value is a single
# assignment of an immutable object and consumers sample far slower than the
# writer.

connected = False
knight_box = None
attacking = False
attack_box = None
hit_count = 0
damage_count = 0
hazard_boxes = []

# hit_count/damage_count from the mod never reset - they count for the whole
# BepInEx session, across every episode. These snapshot the raw counters at
# the start of each episode so callers can read a per-episode delta instead
# (see reset_episode_counters() and enviroment.get_state()).
hit_count_baseline = 0
damage_count_baseline = 0

# Consecutive hits landed without taking damage. Reset by a damage tick or by
# a new episode. Tracked here (in the polling thread, which sees every socket
# frame) rather than in the reward function, which only samples at the control
# rate and would miss hit/damage transitions that fall between its steps.
hit_streak = 0

# Previous raw counter readings, for detecting increments. None means "no
# valid previous frame" (first connect, or after a drop) - without this guard
# the first frame after connecting would read the mod's whole lifetime count
# as one enormous streak.
_prev_hits = None
_prev_damages = None

knight_velocity = (0.0, 0.0)
knight_facing_right = True
boss_health = 0
boss_facing_left = None  # None means no boss found
boss_velocity = (0.0, 0.0)
knight_health = 0
mana = 0
game_speed = 1.0
training_paused = False

is_dashing = False
is_jumping = False
is_falling = False
is_recoiling = False
is_touching_wall_left = False
is_touching_wall_right = False
can_shade_cloak_now = False
is_invuln = False

_thread = None


@dataclass(frozen=True)
class Frame:
    """One parsed frame from the mod. Purely a carrier between _parse_frame()
    and _publish() so neither has to name all two dozen fields positionally."""

    knight_box: tuple
    attacking: bool
    attack_box: tuple            # None while the knight isn't swinging
    hit_count: int
    damage_count: int
    knight_velocity: tuple
    knight_facing_right: bool
    boss_health: int
    boss_facing_left: bool       # None when no boss was found
    boss_velocity: tuple
    knight_health: int
    mana: int
    game_speed: float
    training_paused: bool
    hazard_boxes: list
    is_dashing: bool
    is_jumping: bool
    is_falling: bool
    is_recoiling: bool
    is_touching_wall_left: bool
    is_touching_wall_right: bool
    can_shade_cloak_now: bool
    is_invuln: bool


def _parse_frame(response):
    """Parse one `pos` reply. See connection.cs for the writing side.

    Wire format - semicolon-separated sections, each comma-separated:
        0: knight box            min_x,min_y,max_x,max_y
        1: attacking             "1"/"0"
        2: attack box            same as 0, or empty when not swinging
        3: counters              hits,damages   (lifetime, never reset)
        4: scalars               see the unpack below
        5: flags                 nine "1"/"0"s
        6+: one hazard box each  same as 0
    """
    parts = response.split(";")

    # The title screen and scene transitions have no HeroController. The mod
    # represents that by leaving the knight-box field empty so consumers do
    # not mistake Bounds' all-zero default for a real observation.
    box = tuple(map(float, parts[0].split(","))) if parts[0] else None
    is_attacking = parts[1] == "1"
    atk_box = tuple(map(float, parts[2].split(","))) if parts[2] else None
    hits, damages = map(int, parts[3].split(","))

    (vel_x, vel_y, facing_right, health, boss_facing,
     boss_vel_x, boss_vel_y, hero_health, mp, speed) = parts[4].split(",")

    (dashing, jumping, falling, recoiling,
     touching_wall_left, touching_wall_right,
     shade_cloak_now, invuln, paused) = (flag == "1" for flag in parts[5].split(","))

    return Frame(
        knight_box=box,
        attacking=is_attacking,
        attack_box=atk_box,
        hit_count=hits,
        damage_count=damages,
        knight_velocity=(float(vel_x), float(vel_y)),
        knight_facing_right=facing_right == "1",
        boss_health=int(health),
        # The mod sends the boss's facing as a tri-state, with -1 for "no
        # boss in the scene at all" - which is not the same as facing right.
        boss_facing_left={"1": False, "0": True, "-1": None}[boss_facing],
        boss_velocity=(float(boss_vel_x), float(boss_vel_y)),
        knight_health=int(hero_health),
        mana=int(mp),
        game_speed=float(speed),
        training_paused=paused,
        hazard_boxes=[tuple(map(float, p.split(","))) for p in parts[6:] if p],
        is_dashing=dashing,
        is_jumping=jumping,
        is_falling=falling,
        is_recoiling=recoiling,
        is_touching_wall_left=touching_wall_left,
        is_touching_wall_right=touching_wall_right,
        can_shade_cloak_now=shade_cloak_now,
        is_invuln=invuln,
    )


def _update_hit_streak(frame):
    """Extend or break the damage-free streak, from the change in the raw
    counters since the last frame. Must run before they are overwritten."""
    global hit_streak, _prev_hits, _prev_damages

    if _prev_hits is not None:
        if frame.damage_count > _prev_damages:
            hit_streak = 0
        elif frame.hit_count > _prev_hits:
            hit_streak += frame.hit_count - _prev_hits

    _prev_hits = frame.hit_count
    _prev_damages = frame.damage_count


def _publish(frame):
    """Copy a parsed frame into the module globals readers watch."""
    global knight_box, attacking, attack_box, hit_count, damage_count, hazard_boxes
    global knight_velocity, knight_facing_right, boss_health, boss_facing_left
    global boss_velocity, knight_health, mana, game_speed, training_paused
    global is_dashing, is_jumping, is_falling, is_recoiling
    global is_touching_wall_left, is_touching_wall_right
    global can_shade_cloak_now, is_invuln

    knight_box = frame.knight_box
    attacking = frame.attacking
    attack_box = frame.attack_box
    hit_count = frame.hit_count
    damage_count = frame.damage_count
    hazard_boxes = frame.hazard_boxes
    knight_velocity = frame.knight_velocity
    knight_facing_right = frame.knight_facing_right
    boss_health = frame.boss_health
    boss_facing_left = frame.boss_facing_left
    boss_velocity = frame.boss_velocity
    knight_health = frame.knight_health
    mana = frame.mana
    game_speed = frame.game_speed
    training_paused = frame.training_paused

    is_dashing = frame.is_dashing
    is_jumping = frame.is_jumping
    is_falling = frame.is_falling
    is_recoiling = frame.is_recoiling
    is_touching_wall_left = frame.is_touching_wall_left
    is_touching_wall_right = frame.is_touching_wall_right
    can_shade_cloak_now = frame.can_shade_cloak_now
    is_invuln = frame.is_invuln


def _on_disconnect():
    """Blank out anything that would be a lie while we have no feed."""
    global connected, knight_box, attacking, attack_box, hazard_boxes
    global hit_streak, _prev_hits, _prev_damages

    connected = False
    knight_box = None
    attacking = False
    attack_box = None
    hazard_boxes = []
    # Don't bridge a streak across a dropped connection.
    hit_streak = 0
    _prev_hits = None
    _prev_damages = None


def _poll():
    global connected

    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                # Without a timeout, readline() blocks forever against a
                # listener that accepts but never replies. Time out and
                # reconnect instead of hanging on a stale mod instance.
                sock.settimeout(SOCKET_TIMEOUT)
                sock.connect((HOST, PORT))
                connected = True

                # TCP is a byte stream, not a message protocol. One recv()
                # may contain half a state line or several state lines. The
                # old recv(4096) implementation therefore intermittently fed
                # fragments such as "9.61\r\n48.16" into _parse_frame(). A
                # buffered file preserves surplus bytes and returns exactly
                # one newline-terminated reply for each `pos` request.
                with sock.makefile("rb") as reader:
                    while True:
                        sock.sendall(b"pos\n")
                        payload = reader.readline()
                        if not payload:
                            break

                        try:
                            response = payload.decode("utf-8").rstrip("\r\n")
                            frame = _parse_frame(response)
                        except (UnicodeError, ValueError, IndexError, KeyError) as exc:
                            # A genuinely malformed or version-mismatched line
                            # should not kill the polling thread. Framing errors
                            # are prevented above; this guard remains for bad
                            # fields or an incompatible mod build.
                            print(f"game_state: skipping malformed frame ({exc})")
                            continue

                        _update_hit_streak(frame)
                        _publish(frame)

                        time.sleep(POLL_INTERVAL)

        except (ConnectionError, OSError):
            _on_disconnect()
            # A refused connection on loopback fails in ~1 ms, so retrying
            # immediately pins a core at 100% for as long as the game is not
            # up - stealing CPU from the game itself, which is the actual
            # bottleneck. Back off instead; nothing can arrive in the meantime.
            time.sleep(RECONNECT_DELAY)


def start():
    global _thread
    if _thread is None:
        _thread = threading.Thread(target=_poll, daemon=True)
        _thread.start()


def reset_episode_counters():
    global hit_count_baseline, damage_count_baseline, hit_streak
    hit_count_baseline = hit_count
    damage_count_baseline = damage_count
    hit_streak = 0
