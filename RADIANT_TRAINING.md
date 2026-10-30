# Native Radiant trial — 2026-10-30

## Continuous follow-up from the new update 700

`radiant_continued_20261030_084439` backs up the new update-700 best and latest
checkpoints from the 120-second trial. It independently evaluates the frozen
selected policy for 40 fights, then continues from update 701 with the same
Radiant tier, 120-second limit, 60-second observation time scale, rewards,
learning rate 0.0001 and entropy coefficient 0.005. Normalization stays frozen.
Every 10 updates it runs 20 greedy fights. Latest checkpoints are saved each
update; the strongest evaluation checkpoint is retained separately.

This follow-up is continuous: `continuation_updates` is null and the trainer
uses the normal 10000-update ceiling, rather than the completed 20-update trial's
701 ceiling. The active pointer and read-only monitor follow the new directory.
No new game Mod or B-machine deployment is needed.

## 120-second recovery trial

The follow-up `radiant_120s_20261030_040514` preserves complete update-680 and
update-700 checkpoints, including optimizer and observation-normalization state,
under `backup/` with SHA256 hashes. Python now reacquires socket-bound reset
ownership after an explicit owner-missing reply. This requires no DLL update on
either machine and no new B-machine archive.

The frozen update-680 policy is assessed for 40 native Radiant fights with a
120-second episode limit. Its observation's elapsed-time scale remains 60 seconds,
clamped at 1, so the input seen before the former cutoff remains unchanged.
Reward profile, learning rate, entropy coefficient and observation normalization
are retained. Native fatal damage still contributes -27 from nine lost masks and
-20 from death, totaling -47; these are not increased or removed in this trial.

After the assessment, a separate continuation starts at update 681 and completes
20 updates through update 700, with 20 greedy assessment fights every 10 updates.
It then stops automatically and saves its weights. The previous update-700 model
is a separate preserved checkpoint; matching update numbers do not identify the
same model. `progress.json` records evaluation, continuation and stopped phases.
`python/watch_training.py` can follow the active experiment without controlling
the games. A timeout still truncates and ends the fight; it is not a victory and
does not count as a natural injury for the reward. Timeout counts must be reviewed
alongside wins and remaining boss HP.

Active run: `python/runs/traitor_4env_iframes_fixed/experiments/radiant_20261030_004625`.
The ordinary learner was backed up at update 729 and stopped before replacing
the local DLLs. Both local games now use the tier-switch mod. B's games require
one update/restart with `artifacts/radiant_mod.zip`, then all four games must load
training save slot 1 and stand at the Traitor Lord statue.

The coordinator waits without pressing buttons until every endpoint has the new
protocol and a living loaded save in the known staging area. It then enters the
boss arena to capture resources, reloads with the game's native BossLevel=2,
checks target and observed tier plus 1300 initial boss HP, and validates that the
first natural recorded health loss is fatal. A 1300-HP Ascended fight does not
pass the tier check. Explicit-tier resets never fall back silently to the old
menu reset. The tier command changes runtime setup, not save unlock flags.

After validation, the frozen update-641 source plays two fights per endpoint
(8 total). Results go to `radiant_baseline_fights.jsonl` and its summary. Then the
separate Radiant learner starts with the same source and retains ordinary models.
It uses LR 0.0001, entropy coefficient 0.005, 10Hz control, four stacked frames,
512 transitions per game/update, frozen normalization and the existing 60-second
fight limit. Periodic greedy evaluation selects by wins and remaining boss HP
while Radiant victories are rare; no-hit wins become the priority above the
existing win-rate gate. Native Radiant victory itself requires no damage.

Reward profile `radiant_spacing_v1` retains nail damage/33 reward +1, kill +100,
death -20 and per-mask damage -3. It does not include the separate +25 flawless
bonus: all native Radiant victories are already flawless.

The only new shaping is a bounded +0.20 shockwave-spacing reward. At the first
ACTIVE frame of a shockwave move, the horizontal collision-box edge gap must be
3–6 game units. The dangerous FSM phase must then finish normally with no damage
and a living knight. It awards once per move, not once per step; greater distance,
longer waiting, jumping higher, death and truncated fights cannot multiply it.
This is an experimental spacing range, not a verified optimal dodge distance.
The completion window follows the boss FSM; distant off-screen wave colliders
may remain after this window. Existing time costs continue to discourage stalling.

The package DLL SHA256 is
`4E8190E4452893C73A6C8939523305494EDAA510BCA3A244FB44521F34B32ABD`.
`update_b_machine.ps1` detects the original B game paths and arguments from
ports 9003/9004, backs up installed DLLs, updates and restarts the games. Keep
those original two game processes running before invoking the updater.

The current status is in `progress.json`; validation and gameplay have not
occurred until that status advances beyond `waiting_for_upgraded_games`.
