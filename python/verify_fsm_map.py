"""Compare Control FSM states seen in diagnostic TSVs against _FSM_TO_MOVE.

Lists states that fall through to NONE so the move map can be audited against
what the game actually reports.
"""
import collections
import csv
import glob
import os

from agent.action import _FSM_TO_MOVE, PureVesselMove

DIAG_DIR = r"D:\Program Files\Steam\steamapps\common\Hollow Knight\BepInEx\HollowRL_Diagnostics"


def control_state(fsms):
    for item in fsms.split("|"):
        if item.startswith("Control:"):
            return item.split(":", 1)[1]
    return ""


counter = collections.Counter()
files = sorted(glob.glob(os.path.join(DIAG_DIR, "*.tsv")))
for f in files:
    with open(f, encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) < 15:
                continue
            s = control_state(row[14])
            if s:
                counter[s] += 1

print("TSV files: %d" % len(files))
print("distinct Control FSM states: %d" % len(counter))
print()

mapped = []
unmapped = []
for state, count in counter.most_common():
    move = _FSM_TO_MOVE.get(state)
    if move is None:
        unmapped.append((state, count))
    else:
        mapped.append((state, count, move))

print("=== MAPPED (%d states) ===" % len(mapped))
for state, count, move in mapped:
    print("  %6d  %-20s -> %s" % (count, state, move.name))

print()
print("=== UNMAPPED -> NONE (%d states) ===" % len(unmapped))
for state, count in unmapped:
    print("  %6d  %s" % (count, state))

# How often boss_move returns NONE overall
total = sum(counter.values())
unmapped_total = sum(c for _, c in unmapped)
print()
print("unmapped share: %d / %d = %.1f%%" % (unmapped_total, total, 100.0 * unmapped_total / total))
