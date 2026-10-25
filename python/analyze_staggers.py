"""Find Pure Vessel stagger thresholds from boss_hp plateaus in a TSV."""
import csv
import glob
import os
import sys

DIAG_DIR = os.environ.get(
    "HOLLOWRL_DIAG_DIR",
    os.path.join(os.path.dirname(__file__), "diagnostics"),
)


def control_state(fsms):
    for item in fsms.split("|"):
        if item.startswith("Control:"):
            return item.split(":", 1)[1]
    return ""


def analyze(path):
    with open(path, encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)
        stun_runs = []
        i = 0
        rows = []
        for row in reader:
            if len(row) < 18:
                continue
            try:
                rows.append((float(row[0]), int(row[3]), control_state(row[14])))
            except ValueError:
                continue

    # Continuous runs where Control FSM is in a Stun state.
    n = len(rows)
    i = 0
    while i < n:
        if rows[i][2].startswith("Stun"):
            t0, hp0, _ = rows[i]
            j = i
            while j < n and rows[j][2].startswith("Stun"):
                j += 1
            t1, hp1, _ = rows[j - 1]
            dur = t1 - t0
            if dur >= 0.5:
                stun_runs.append((hp0, hp1, round(dur, 2), rows[i][2], rows[j - 1][2]))
            i = j
        else:
            i += 1

    print("FILE: %s" % os.path.basename(path))
    print("  Stun runs (Control FSM in Stun*, >=0.5s):")
    for hp0, hp1, dur, s0, s1 in stun_runs:
        print("    hp %4d -> %4d  for %.2fs  (%s -> %s)" % (hp0, hp1, dur, s0, s1))


targets = sys.argv[1:] or [DIAG_DIR]
files = []
for target in targets:
    if os.path.isdir(target):
        files.extend(glob.glob(os.path.join(target, "*.tsv")))
    elif os.path.isfile(target):
        files.append(target)
for f in sorted(set(files)):
    analyze(f)
