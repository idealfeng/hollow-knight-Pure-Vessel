"""Physical fight metrics for model selection, independent of reward shaping."""


def is_flawless(info):
    return (bool(info.get("won")) and not info.get("truncated_reason")
            and info.get("damage_tracking_valid", True)
            and info.get("episode_damage", info.get("damage", -1)) == 0)


def nohit_selection_score(summary, minimum_win_rate=0.90):
    win_rate = float(summary["win_rate"])
    hp = float(summary["mean_min_boss_hp"])
    eligible = win_rate >= minimum_win_rate
    if eligible:
        return (1.0, float(summary["nohit_rate"]), -float(summary["mean_damage"]), win_rate, -hp)
    return (0.0, win_rate, -hp, -float(summary["mean_damage"]), 0.0)


def nohit_candidate_improved(candidate, baseline, minimum_win_rate=0.90):
    if candidate["win_rate"] < minimum_win_rate or candidate["win_rate"] < baseline["win_rate"] - 0.10 - 1e-9:
        return False
    extra_nohits = candidate["nohit_rate"] - baseline["nohit_rate"]
    less_damage = baseline["mean_damage"] - candidate["mean_damage"]
    return ((extra_nohits >= 0.05 - 1e-9 and less_damage >= 0)
            or (extra_nohits >= -1e-9 and less_damage >= 0.50 - 1e-9))
