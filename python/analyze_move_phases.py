"""Read-only: analyse Control FSM state sequences in the diagnostic TSVs.

Goal: can we distinguish Windup / Active / Recovery of each move from the actual
FSM state alone? Outputs per-state durations, predecessor/successor transitions,
hazard co-occurrence and the dominant animation clip - without guessing.
"""
import collections
import csv
import glob
import os

DIAG_DIR = r"D:\Program Files\Steam\steamapps\common\Hollow Knight\BepInEx\HollowRL_Diagnostics"


def control_state(fsms):
    for item in fsms.split("|"):
        if item.startswith("Control:"):
            return item.split(":", 1)[1]
    return ""


def first_anim(anims):
    for a in anims.split("|"):
        if a.strip():
            return a.strip()
    return ""


boss_runs = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(DIAG_DIR, "*.tsv"))):
    with open(f, encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) < 18:
                continue
            try:
                t = float(row[0])
                boss_id = row[2]
                state = control_state(row[14])
                anim = first_anim(row[15])
                hazard_count = int(row[16])
                bvx = float(row[6])
                bvy = float(row[7])
            except (ValueError, IndexError):
                continue
            if state:
                boss_runs[boss_id].append((t, state, hazard_count, bvx, bvy, anim))

state_dur = collections.defaultdict(list)          # state -> list of (frames, seconds)
transitions = collections.defaultdict(collections.Counter)  # state -> next state
state_hazard = collections.defaultdict(lambda: [0, 0])       # state -> [frames with hazard, total]
state_anim = collections.defaultdict(collections.Counter)    # state -> anim
state_speed = collections.defaultdict(list)         # state -> list of |v|

for boss_id, seq in boss_runs.items():
    seq.sort(key=lambda x: x[0])
    i, n = 0, len(seq)
    while i < n:
        t0, state = seq[i][0], seq[i][1]
        j = i
        while j < n and seq[j][1] == state:
            j += 1
        t1 = seq[j - 1][0]
        state_dur[state].append((j - i, t1 - t0))
        for k in range(i, j):
            if seq[k][2] > 0:
                state_hazard[state][0] += 1
            state_hazard[state][1] += 1
            state_anim[state][seq[k][5]] += 1
            state_speed[state].append((seq[k][3] ** 2 + seq[k][4] ** 2) ** 0.5)
        if j < n:
            transitions[state][seq[j][1]] += 1
        i = j

print("=== per-state: count / median duration / hazard% / median speed / top anim / transitions ===")
for state in sorted(state_dur):
    runs = state_dur[state]
    frames = [r[0] for r in runs]
    secs = [r[1] for r in runs]
    hz = state_hazard[state]
    hazard_pct = 100.0 * hz[0] / hz[1] if hz[1] else 0.0
    speeds = state_speed[state]
    med_speed = sorted(speeds)[len(speeds) // 2] if speeds else 0.0
    top_anim = state_anim[state].most_common(2)
    nxt = transitions[state].most_common(3)
    print("\n[%s]  runs=%d  med_frames=%.0f  med_s=%.2f" % (state, len(runs), sorted(frames)[len(frames)//2], sorted(secs)[len(secs)//2]))
    print("   hazard%%=%.0f  med_speed=%.2f  anim=%s" % (hazard_pct, med_speed, top_anim))
    print("   -> %s" % nxt)
