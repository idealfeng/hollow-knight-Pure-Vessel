"""Map Pure Vessel moves -> FSM/animation/hazard from diagnostic TSVs."""
import collections
import csv
import glob
import os
import sys

DIAG_DIR = os.environ.get(
    "HOLLOWRL_DIAG_DIR",
    os.path.join(os.path.dirname(__file__), "diagnostics"),
)
NOISE_FSM = {"FSM", "Control", "corpse", "Destroy Self", "Stun Control",
             "Dreamnail Reject", "White Wave", "Spawn"}


def parse_fsms(s):
    out = []
    for item in s.split("|"):
        if ":" in item:
            name, state = item.split(":", 1)
        else:
            name, state = item, ""
        out.append((name.strip(), state.strip()))
    return out


def parse_anims(s):
    return [a.strip() for a in s.split("|") if a.strip()]


def parse_hazards(s):
    if not s:
        return []
    out = []
    for entry in s.split("||"):
        if not entry:
            continue
        f = entry.split("~")
        if len(f) >= 5:
            out.append({"object": f[1], "root": f[2]})
    return out


def analyze(path):
    fsm_all = collections.Counter()
    anim_all = collections.Counter()
    haz_obj = collections.Counter()
    haz_root = collections.Counter()
    fsm_attack = collections.Counter()
    anim_attack = collections.Counter()
    obj_to_fsm = collections.defaultdict(collections.Counter)
    obj_to_anim = collections.defaultdict(collections.Counter)
    rows = 0
    attack_rows = 0
    with open(path, encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) < 18:
                continue
            rows += 1
            fsms = parse_fsms(row[14])
            anims = parse_anims(row[15])
            try:
                n_haz = int(row[16])
            except ValueError:
                n_haz = 0
            hazards = parse_hazards(row[17])
            for name, state in fsms:
                if state:
                    fsm_all["%s:%s" % (name, state)] += 1
            for a in anims:
                anim_all[a] += 1
            for h in hazards:
                haz_obj[h["object"]] += 1
                haz_root[h["root"]] += 1
            if n_haz > 0 and hazards:
                attack_rows += 1
                for name, state in fsms:
                    if state and name not in NOISE_FSM:
                        fsm_attack["%s:%s" % (name, state)] += 1
                for a in anims:
                    anim_attack[a] += 1
                for h in hazards:
                    for name, state in fsms:
                        if state and name not in NOISE_FSM:
                            obj_to_fsm[h["object"]]["%s:%s" % (name, state)] += 1
                    for a in anims:
                        obj_to_anim[h["object"]][a] += 1
    return {"rows": rows, "attack_rows": attack_rows, "fsm_all": fsm_all,
            "anim_all": anim_all, "haz_obj": haz_obj, "haz_root": haz_root,
            "fsm_attack": fsm_attack, "anim_attack": anim_attack,
            "obj_to_fsm": obj_to_fsm, "obj_to_anim": obj_to_anim}


def show(counter, title, limit=50):
    print("\n=== %s (top %d) ===" % (title, limit))
    for key, n in counter.most_common(limit):
        print("  %6d  %s" % (n, key))


def main():
    targets = sys.argv[1:] or [DIAG_DIR]
    files = []
    for target in targets:
        if os.path.isdir(target):
            files.extend(glob.glob(os.path.join(target, "*.tsv")))
        elif os.path.isfile(target):
            files.append(target)
    files = sorted(set(files))
    print("Found %d files" % len(files))
    agg = {"rows": 0, "attack_rows": 0,
           "fsm_all": collections.Counter(), "anim_all": collections.Counter(),
           "haz_obj": collections.Counter(), "haz_root": collections.Counter(),
           "fsm_attack": collections.Counter(), "anim_attack": collections.Counter(),
           "obj_to_fsm": collections.defaultdict(collections.Counter),
           "obj_to_anim": collections.defaultdict(collections.Counter)}
    for f in files:
        r = analyze(f)
        print("\n#### %s  rows=%d attack_rows=%d" % (os.path.basename(f), r["rows"], r["attack_rows"]))
        show(r["haz_obj"], "hazard ObjectName", 30)
        for k in ("fsm_all", "anim_all", "haz_obj", "haz_root",
                  "fsm_attack", "anim_attack"):
            agg[k] += r[k]
        agg["rows"] += r["rows"]
        agg["attack_rows"] += r["attack_rows"]
        for o, c in r["obj_to_fsm"].items():
            agg["obj_to_fsm"][o] += c
        for o, c in r["obj_to_anim"].items():
            agg["obj_to_anim"][o] += c

    print("\n\n@@@@@@@@ AGGREGATE rows=%d attack_rows=%d @@@@@@@@" % (agg["rows"], agg["attack_rows"]))
    show(agg["haz_obj"], "hazard ObjectName (aggregate)", 60)
    show(agg["haz_root"], "hazard RootName (aggregate)", 60)
    show(agg["fsm_all"], "ALL FSM states", 70)
    show(agg["anim_all"], "ALL animation clips", 70)
    show(agg["fsm_attack"], "FSM states during attack frames", 70)
    show(agg["anim_attack"], "animation clips during attack frames", 70)

    print("\n\n=== hazard object -> top FSM states ===")
    for o, c in sorted(agg["obj_to_fsm"].items(), key=lambda kv: -sum(kv[1].values())):
        print("\n  [%s]" % o)
        for state, n in c.most_common(8):
            print("      %6d  %s" % (n, state))

    print("\n\n=== hazard object -> top animation clips ===")
    for o, c in sorted(agg["obj_to_anim"].items(), key=lambda kv: -sum(kv[1].values())):
        print("\n  [%s]" % o)
        for clip, n in c.most_common(8):
            print("      %6d  %s" % (n, clip))


if __name__ == "__main__":
    main()

