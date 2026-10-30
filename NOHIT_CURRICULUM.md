# Traitor Lord: no-hit training, 2026-10-29

The no-hit branch starts from the preserved update-641 checkpoint. It recorded
39 wins and 4 flawless wins in the previous 40-fight confirmation, with mean
damage 2.75 masks. The current victory learner was saved through update 707 and
backed up separately, including optimizer state, normalization and configuration.

Active workflow:
`python/runs/traitor_4env_iframes_fixed/experiments/nohit_20261029_232913`.
`progress.json` records the active phase; `decision.json` is written when the
pilot comparison has finished. The root run's `active_experiment.json` points here.

## Objective

| Raw reward component | Victory profile | No-hit profile |
| --- | ---: | ---: |
| Each mask lost | -3 | -6 |
| Boss defeated | +100 | +100 |
| Entire fight flawless and victorious | 0 | +25 additional |
| Boss damage / 33 HP | +1 | +1 |
| Clean dodge | +0.05 | +0.05 |
| Missed swing | -0.03 | -0.03 |
| Death | -20 | -20 |
| Time penalty / second | -0.10 | -0.10 |

Flawless means no recorded damage anywhere in the episode, including its killing
transition. Healing does not erase prior damage. Timeout, truncated or invalid
counter histories, and simultaneous player/boss deaths cannot earn this bonus.
Raw rewards still enter PPO with the existing scale of 0.05. Scores from different
reward profiles must not be used as evidence of better play.

Learning rate stays 0.0001, entropy coefficient 0.005, control 10Hz, frame stack 4,
rollout 512 transitions per game, 4 games. Architecture, observations and frozen
normalization are retained, as are the remaining PPO settings and 60-second limit.
This phase uses the current 800-HP difficulty with normal health resources.
No change to 1300 HP, one-hit death, or action filtering has been applied.

## Selection and verification

The no-hit selection objective first requires at least 90% wins in an evaluation
batch. Eligible checkpoints are ranked by flawless-win rate, then lower mean
damage, then win rate. Below the gate, wins and remaining boss health determine
recovery progress. Evaluations use fixed greedy actions and complete fresh fights.

The workflow evaluates the frozen source for 40 games, trains the new reward
profile for 20 updates, then evaluates its frozen result for 40 games. A candidate
must meet the win-rate gate and avoid losing more than 10 percentage points of
win rate relative to its source. It must either improve flawless-win rate by at
least 5 percentage points without increasing damage, or keep flawless-win rate
and reduce mean damage by at least 0.5 masks. A promising candidate requires a
second fresh 40-game evaluation of both policies before adoption. These are
practical pilot criteria, not a claim of statistical significance.

After acceptance the selected candidate continues under the no-hit profile.
If criteria fail, or the experiment fails, the separately saved latest victory
policy resumes with the original reward profile. No checkpoint is overwritten by
the trial. Branch retries retain validated progress and preserve logs.

## Injury evidence

A read-only sample of A's two windows found many losses near the boss body;
shockwaves also occurred. Nearest-hazard attribution is approximate and cannot
establish the actual damage source or prove Shade Cloak cooldown misuse.

Episode logs now include the five recent decision states around health loss:
Shade Cloak availability, action buttons, dashing/invulnerability, boss FSM and
move phase, collision boxes and hazard movement. These are diagnostic records;
they do not expand the policy's inputs. Exact remaining Shade Cloak cooldown is
not yet added to observations. That change can be considered after the event
records show whether timing or lack of cooldown information is the bottleneck.

The subsequent curriculum is to establish reliable no-hit performance, then
validate a longer 1300-HP fight, and finally attempt Radiant. Those later stages
have not been activated by this pilot.
