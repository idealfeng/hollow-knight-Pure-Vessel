# HollowRL

A reinforcement learning agent that fights a Hollow Knight boss — Pure Vessel —
trained with PPO against the live game.

There is no emulator, no screen scraping and no fake keystrokes. A BepInEx mod
loaded inside the game exposes the fight as a structured state feed and accepts
input commands over a loopback TCP socket; a Python trainer on the other end
reads that state at 15 Hz, samples an action from a policy network, and sends the
resulting button presses back into the game process.

```
        Hollow Knight process                      Python trainer
  ┌───────────────────────────────┐          ┌──────────────────────────┐
  │  HollowRL.dll (BepInEx)       │          │  main.py                 │
  │                               │          │                          │
  │  KnightPos.cs   reads Unity   │  "pos"   │  game_state.py  poll     │
  │                 state ────────┼─────────►│                 @100 Hz  │
  │  connection.cs  TCP :9000     │◄─────────┤                          │
  │                               │  state   │  state_to_vector()       │
  │                               │   line   │        │                 │
  │                               │          │        ▼                 │
  │  InputInjector.cs             │  KD/KU/  │  ActorCritic (PPO)       │
  │    Harmony-patches InControl  │◄─ TAP ───┤        │                 │
  │    getters, per frame         │  CLEAR/  │        ▼                 │
  │                               │   KILL   │  controls.py             │
  └───────────────────────────────┘          └──────────────────────────┘
```

Input is injected *underneath* the game's own input layer rather than through the
OS, so the game window does not need focus, nothing leaks into whatever else you
have open.

**Contents:** [Repository layout](#repository-layout) ·
[Setup](#setup) · [Running](#running) ·
[How an episode runs](#how-an-episode-runs) ·
[Observations](#observation-space--44-features) ·
[Actions](#action-space--five-independent-heads) ·
[Reward](#reward-function) · [PPO configuration](#ppo-configuration) ·
[Wire protocol](#wire-protocol) · [Files a run produces](#files-a-run-produces) ·
[Known rough edges](#known-rough-edges)

---

## Repository layout

```
unity_mod/HollowRL/
  connection.cs        BepInEx plugin: TCP server, per-frame state snapshot, hotkeys
  KnightPos.cs         Reads knight/boss/hazard state out of Unity (some via reflection)
  InputInjector.cs     Harmony patches on InControl getters — how input actually gets in
  HollowRL.csproj      SDK-style, builds against the game's own assemblies

python/
  main.py              Training entry point: rollout loop + PPO updates + monitor
  game_state.py        Background thread polling the mod, exposed as module globals
  controls.py          Input command client (KD / KU / TAP / CLEAR / KILL)
  ppo_model.py         Shared-trunk actor-critic with five independent action heads
  ppo_training.py      GAE + clipped surrogate objective + diagnostics
  rollout_buffer.py    Fixed-size rollout storage and GAE computation
  obs_norm.py          Running mean/variance observation normaliser
  monitor.py           Live 3x3 training dashboard (Tk + matplotlib)
  test_gui_client.py   Live game/reward/policy inspector, usable while you play
  state_vector_viewer.py  Bare listing of the 44 observation features, live
  agent/               action.py (action groups), reward.py, calc.py (box geometry)
  enviroment/          state_to_vector.py (observation + is_done), reset.py (episode reset)

BepInEx/              Bundled BepInEx 5 (x64) — copy its contents into the game folder
HK_save/              The save file the agent trains against (slot 1)
```

`enviroment` is spelled that way in the code; it is the real package name, not a
typo in this document.

---

## Setup

Four steps, in order: install BepInEx into the game, build and copy the mod,
install the save file, create the Python environment.

**Video walkthrough of installing this mod:**
<https://youtu.be/qeBDdZFYwac?si=gkyqiGJht_eLDToL>

### Requirements

- Hollow Knight (Steam). On Linux it runs under Proton; the game files live in a
  normal directory and its managed assemblies are ordinary .NET, so the mod
  builds with the Linux SDK even though the game itself runs through Wine.
- [BepInEx 5](https://github.com/BepInEx/BepInEx/releases), x64 — the mod loader
  the plugin runs under. A copy is bundled in `BepInEx/`; the first step below
  installs it. BepInEx 6 is a different loader and is not what this targets.
- .NET SDK (`dotnet`) — the project targets `net472` via the
  `Microsoft.NETFramework.ReferenceAssemblies` package, so Visual Studio and its
  targeting packs are not needed.
- Python 3.11+ with `numpy`, `torch`, `matplotlib`.
- Tk, for every GUI in this project. It is not a pip package: on Arch/CachyOS
  `sudo pacman -S tk`, on Debian/Ubuntu `sudo apt install python3-tk`.

### Install BepInEx

Copy the *contents* of this repo's `BepInEx/` folder into the Hollow Knight
folder, so that `winhttp.dll`, `doorstop_config.ini` and `BepInEx/` end up next
to the game executable — not in a nested subfolder:

```sh
cp -r BepInEx/. ~/.local/share/Steam/steamapps/common/"Hollow Knight"/
```

Then launch the game once so BepInEx unpacks `BepInEx/core` and writes
`BepInEx/LogOutput.log`.

**On Linux this also needs a Steam launch option** (right-click the game →
Properties → Launch Options):

```sh
WINEDLLOVERRIDES="winhttp=n,b" %command%
```

BepInEx 5 loads by hijacking `winhttp.dll`, and Wine ignores a local one in
favour of its builtin without this. If it is missing there is no
`BepInEx/LogOutput.log`, nothing listens on port 9000, and Python reports no
input channel — even though the game is running and the DLL is in `plugins/`.
Suspect this before suspecting the code.

### Build and install the mod

#### Windows

Follow this walkthrough for setting up BepInEx and dropping a plugin DLL into
Hollow Knight: <https://youtu.be/QMiWbu0il6w?si=XOCxaPSrAJe1jEH0>. Build the DLL
with the same `dotnet build` command shown below, then copy it into
`Hollow Knight\BepInEx\plugins\`.

#### Linux

```sh
dotnet build unity_mod/HollowRL.slnx -c Release
cp unity_mod/HollowRL/bin/Release/HollowRL.dll \
   ~/.local/share/Steam/steamapps/common/"Hollow Knight"/BepInEx/plugins/
```

The build resolves the game's assemblies from `$(HKDir)`, which defaults to the
standard Steam path above. If yours differs:

```sh
dotnet build unity_mod/HollowRL.slnx -c Release -p:HKDir="/path/to/Hollow Knight"
```

A missing game directory fails the build with an explicit message rather than a
wall of unresolved-reference errors.

### Save file

`HK_save/user1.dat` is the save this project is developed against: slot 1, parked
at the boss the agent trains on. The reset routine and the arena constants assume
it, so use it rather than one of your own.

> **Back up your own saves first.** Copying this file over `user1.dat` overwrites
> save slot 1.

#### Windows

Press `Win`+`R`, paste `%AppData%\..\LocalLow\Team Cherry\Hollow Knight`, press
Enter, then copy `HK_save/user1.dat` into the folder that opens.

#### Linux

```sh
cp HK_save/user1.dat \
   ~/.local/share/Steam/steamapps/compatdata/367520/pfx/drive_c/users/steamuser/AppData/LocalLow/"Team Cherry"/"Hollow Knight"/
```

### Python environment

```sh
python3 -m venv python/.venv
python/.venv/bin/pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv python\.venv
python\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## Running

Start the game with the mod loaded, load save slot 1, and stand at the boss
statue. Then, from `python/`:

```sh
.venv/bin/python main.py
```

On Windows, run this from the repository root (the script also selects the
correct working directory so checkpoints stay under `python/`):

```powershell
.\run_windows.ps1
```

This opens the training monitor and starts collecting rollouts. Training resumes
automatically from `ppo_checkpoint.pth` if one exists; the monitor's header shows
whether the run is fresh or resumed and from which update. Checkpoints are local
experiment artifacts and are intentionally ignored by Git, so a fresh clone starts
from scratch. Move a checkpoint into `python/` only when you deliberately want to
resume that exact environment and model architecture.

Two read-only tools can run alongside it, or on their own while you play the
fight yourself:

```sh
.venv/bin/python test_gui_client.py       # game view, live reward breakdown, policy probabilities
.venv/bin/python state_vector_viewer.py   # every observation feature and its current value
```

### In-game hotkeys

These are real keyboard keys, handled by the mod on the main thread — they are
for the human watching, not for the agent.

| Key | Effect |
|-----|--------|
| `O` | Toggle 2x game speed (`Time.timeScale`). The trainer scales its sleeps to match, so a step stays the same amount of *game* time. |
| `P` | Pause training. The trainer releases all held keys and blocks until you unpause. |
| `Y` | Kill the knight instantly, forcing an episode reset. Bypasses i-frames. |

### Replay capture (optional)

`main.py` can fire an external recorder every 100 episodes and on every new best
episode reward. It shells out to whatever `HOLLOWRL_REPLAY_CMD` is set to:

```sh
export HOLLOWRL_REPLAY_CMD='obs-cli replaybuffer save'                 # OBS replay buffer
export HOLLOWRL_REPLAY_CMD='pkill -SIGUSR1 -x gpu-screen-recorder'     # gpu-screen-recorder
```

Unset, replays are skipped and a single note is printed. Training does not depend
on it.

---

## How an episode runs

1. `reset_environment()` releases every key, waits out the death sequence, then
   navigates the retry menu (`left`, `up`, `submit`) and waits for a live, stable
   fight state — retrying up to four times before giving up with a message.
2. The control loop runs at **15 Hz**. Each step reads the current state, samples
   one action from each of the five heads, dispatches them, sleeps `1/15 s`
   scaled by game speed, then reads the next state and computes the reward.
3. An episode ends when the boss's health reaches 0, when the knight dies, or at
   the **60 s cap** — at which point the trainer sends `KILL` rather than letting
   the fight run on. A capped ending is flagged as `timed_out`, which suppresses
   the death and time penalties so a scheduled ending costs nothing.
4. Every **2048 steps** (~136 s of play, roughly two full-length episodes) the
   rollout closes, keys are released, and PPO runs for 4 epochs over the batch.

Losing contact with the state feed mid-rollout is treated as a truncation, so the
advantage chain is cut cleanly instead of splicing two unrelated moments together.

---

## Observation space — 119 features

Built by `enviroment.state_to_vector()`, normalised by a running mean/variance
estimator (`obs_norm.py`) that clips to ±10 sigma. All boxes are
`(min_x, min_y, max_x, max_y)` — two opposite corners, **not** `(x, y, w, h)`.

| Group | Features |
|-------|----------|
| Knight body | health/mana, facing, velocity, action flags, grounded/wall contact, and left/right wall distance (17) |
| Nail hitbox | `attack_dx/dy/width/height`, `attack_present` — knight-relative (5) |
| Boss | `boss_present`, `boss_health`, `boss_dx/dy`, `boss_rel_vel_x/y`, `boss_width/height`, `boss_facing_left`, `boss_faces_knight`, `knight_faces_boss` (11) |
| Fight progress | `hit_count`, `damage_count`, `time_frac` (3) |
| Hazard summary | `hazard_count`, `nearest_edge_distance`, 4 directional gaps, `attack_active` (7) |
| Move one-hot | 12 Pure Vessel moves from the Control FSM state (12) |
| Hazards | 8 closest hazards, each `dx/dy/width/height/rel_vel_x/rel_vel_y/boss_attached/present` (64) |

`time_frac` matters more than it looks: without it the critic cannot distinguish a
state at t=5 s from an identical-looking one at t=59 s, even though their true
returns differ enormously.

`state_vector_viewer.py`'s `FEATURE_NAMES` list must stay in the same order as
`state_to_vector()`'s return, and `STATE_SIZE = OBSERVATION_SIZE` (119) appears in both `main.py` and
`test_gui_client.py`. Changing the observation means touching all four places.

---

## Action space — five independent heads

Each head is sampled independently every step, so the agent can (for example)
move left, look up and swing at once.

| Head | Size | Actions |
|------|------|---------|
| move | 6 | left, right, stop, tap-left, tap-right, nothing |
| look | 4 | up, down, stop looking, nothing |
| attack | 2 | attack, nothing |
| dash | 2 | dash, nothing |
| jump | 3 | short hop (0.1 s), long jump (0.4 s), nothing |

Joint log-probability is the sum across heads. Per-head log-probs are stored in
the rollout buffer as well, which is what lets `ppo_update` report a genuine
per-head KL, clip fraction, entropy and gradient norm instead of only the blended
joint figures — a calm head and a wild one average out to a perfectly
normal-looking joint KL.

Because KL is additive over heads, `TARGET_KL` is set to **0.1**, not the usual
single-head 0.01–0.03. Across five heads that works out to a conventional ~0.02
per-head budget; the textbook value stops nearly every update after one epoch.

Held directions (`left`/`right`/`up`/`down`) stay down until released. Attacks,
dashes and jumps are timed taps whose release is scheduled *by the mod*, not by a
Python timer — a Python-side timer could be stuck behind the GIL during a PPO
update, silently stretching a 0.1 s hop into something much longer at exactly the
moments hardest to notice.

---

## Reward function

Defined in `agent/reward.py`. Logged raw; scaled by `REWARD_SCALE = 0.05` before
it reaches the buffer.

| Event | Reward |
|-------|--------|
| Damage dealt to the boss | +1.0, plus 0.25 per consecutive clean hit, capped at +2.0 |
| Boss Control FSM enters a real stun | +10 |
| Boss defeated | +100 |
| Time | −0.15 per second |
| Damage taken | −4 per mask |
| Death | −40 |

The hit streak is tracked in the polling thread rather than in the reward
function, because the poll sees every socket frame while the reward function only
samples at 15 Hz and would miss hit/damage transitions falling between steps.

The stagger signal and move classifier are read directly from Pure Vessel's
PlayMaker Control FSM. `agent/action.py:boss_move()` reads its
`ActiveStateName` and maps it to a 12-class `PureVesselMove` one-hot, so a move is
recognised during its windup ("* Antic") rather than only once its hazard collider
appears.

---

## PPO configuration

All in `main.py`.

| Parameter | Value | Note |
|-----------|-------|------|
| `CONTROL_HZ` | 15 | 900 steps per capped episode |
| `ROLLOUT_STEPS` | 2048 | ~2 episodes per update |
| `GAMMA` | 0.997 | 333-step / 22 s horizon. At 0.99 a +100 defeat bonus discounts to 0.012 across an episode — the terminal rewards may as well not exist |
| `GAE_LAMBDA` | 0.95 | |
| `CLIP_EPS` | 0.2 | policy ratio clip |
| `VALUE_CLIP_EPS` | 1.0 | deliberately decoupled from `CLIP_EPS` — one bounds a dimensionless probability ratio, the other a return-scaled value |
| `LR` | 3e-4 | Adam, `eps=1e-5` |
| `ENT_COEF` | 0.01 | replaces epsilon-greedy exploration |
| `VF_COEF` | 0.5 | |
| `MAX_GRAD_NORM` | 0.5 | |
| `TARGET_KL` | 0.1 | joint budget across five heads |
| `EPOCHS` / `MINIBATCH_SIZE` | 4 / 128 | 16 gradient steps per update |

The network is a shared 119→128→128 tanh trunk with orthogonal initialisation,
five categorical heads (gain 0.01) and one critic head (gain 1.0).

### Reading the monitor

`monitor.py` plots nine panels: reward per episode, episode length, hits vs damage
taken, policy/value loss, entropy, approx-KL and clip fraction, explained
variance, reward per step, and a text summary.

Explained variance is the single most useful number there — ~1.0 means the critic
predicts returns well, ~0.0 means it is no better than guessing the mean, and
negative means it is actively worse than useless. Value-loss spikes paired with
an explained-variance dip point at the critic or the shared trunk, never at an
action head: the value gradient structurally never reaches head weights, which is
why `grad_norm_trunk` and `grad_norm_critic` are logged separately.

---

## Wire protocol

One TCP listener on `127.0.0.1:9000`, one thread per client. The trainer keeps
two connections open (state poll and input), and the dashboards add more.

**State request.** Send `pos\n`, receive one semicolon-separated line:

```
knight_box ; attacking ; attack_box ; hits,damages ; scalars ; flags ; hazard_box… 
```

| Field | Contents |
|-------|----------|
| `knight_box` | `min_x,min_y,max_x,max_y` |
| `attacking` | `0` / `1` |
| `attack_box` | same corner format, **empty** when not attacking |
| counters | `hits,damages` — lifetime totals for the BepInEx session |
| scalars | `vel_x,vel_y,facing_right,boss_health,boss_facing,boss_vel_x,boss_vel_y,knight_health,mana,game_speed,boss_x,boss_y,wall_left_distance,wall_right_distance` |
| flags | `dashing,jumping,falling,recoiling,wall_left,wall_right,shade_cloak,invuln,paused,on_ground` |
| hazards | zero or more `id,box,velocity,boss_attached` records; Python keeps the nearest eight plus summaries |

`boss_facing` is `1` for right, `0` for left, `-1` when no boss is present.
Python stores it inverted, as `boss_facing_left`.

**Input commands.** Every command gets a one-line reply (`ok`, or `err …`).

| Command | Effect |
|---------|--------|
| `KD <action>` | Hold until released. Idempotent. |
| `KU <action>` | Release. Safe on an action that is not held. |
| `TAP <action> <ms>` | Hold for `ms`, released by the mod. |
| `CLEAR` | Release everything. |
| `KILL` | Instant death, to force an episode reset. |

Actions: `left`, `right`, `up`, `down`, `jump`, `attack`, `dash`, `cast`,
`quick_cast`, `super_dash`, `dream_nail`, `submit`, `cancel`, `pause`. These are
the trainer's vocabulary, deliberately not the game's field names, so Python
never has to know the in-game key bindings.

Commands arrive on socket threads and are only ever *applied* on the main thread
in `InputInjector.Tick()`, so the game sees a consistent input state per frame.
Actions the trainer is not currently driving pass straight through to the real
hardware — you can pick up a controller and play with the mod loaded. When a
client disconnects, everything it held is released, so a crashed trainer cannot
leave the knight running into a wall for the rest of the session.

---

## Files a run produces

| Path | Contents |
|------|----------|
| `python/ppo_checkpoint.pth` | Model, optimiser, observation-normaliser stats, update/step/episode counters. Written every 5 updates. |
| `python/monitor_history.npz` | Every plotted series, saved at most once per 5 s. Reloaded on startup so curves survive a restart. |
| `python/runs_archive/` | Archived checkpoint + history pairs from earlier runs. |

A checkpoint whose architecture no longer matches (different action heads, for
instance) exits with an explicit message rather than a raw shape error. Because
the two files are independent, replacing one without the other gives you a model
plotted against someone else's history — archive them as a pair.

---

## Known rough edges

- **Move classification depends on the Control FSM state.** `boss_move()` keys off
  Pure Vessel's PlayMaker `Control` FSM `ActiveStateName`; a different boss needs
  its own state-to-move map (see `agent/action.py:_FSM_TO_MOVE`).
- **Reset costs ~9 s of fixed sleep per episode**, on top of the menu navigation.
  The surrounding comments describe a state-driven wait that the code does not
  currently do.
- **`STATE_SIZE` is duplicated** in `main.py` and `test_gui_client.py`, and the
  feature-name list in `state_vector_viewer.py` is maintained by hand.
- **The reward's `hit_streak`** is read from the live state, so it is deliberately
  absent from the observation vector — the agent cannot see its own streak.
- **Single environment, no parallelism.** The game is the bottleneck (~98% of
  wall-clock time), which is why the state poll is throttled to 100 Hz and the
  reconnect loop backs off rather than spinning.
