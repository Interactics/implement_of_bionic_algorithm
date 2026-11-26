# 2026-11-25 · Stabilize RL training (ADR 0005)

**Blocks changed:** B4 Tracking env (reward, action space), B5 PPO trainer (rollouts, adaptive LR, best-model eval, `--best`, `--tag`) · see [architecture](../architecture.md) and the [control view](../control-architecture.md)

## What changed
- `scripts/tracking_env.py`:
  - **Tracking rewards** are now `w · exp(−err² / σ²)` with the σ values from ADR 0005. Body terms use the mean over bodies, so they don't scale with the body count.
  - **Penalty sum clipped at −10 per step**, below the +20 survival bonus.
  - **Action space ±3.** The robot still gets `clip(a, ±1)`, and a new `act_range` penalty `0.5 · Σ max(0, |a| − 1)²` pulls the policy back inside.
  - The self-check now also asserts that rewards stay within their bounds and that out-of-range actions are penalized.
- `scripts/train_tracking.py`:
  - **Rollouts of 8 × 1024** instead of 8 × 256, so 8k samples per update.
  - **`AdaptiveLR` callback**, which applies the paper's KL rule once per update: KL > 0.02 → lr ÷ 1.5; KL < 0.005 → lr × 1.5; lr kept in [1e-5, 1e-3], starting at 1e-4.
  - **`EvalCallback`**: every ~1M steps, 10 deterministic episodes from random start frames. On a new best it saves `best_model.zip` and `best_vecnormalize.pkl`.
  - **`--play --best`** watches the best model.
  - **`--tag NAME`** keeps runs in separate folders, so last night's collapsed run (`05_02_poses_0-4`) stays untouched next to the new one (`05_02_poses_0-4_v2`).

## Why
- Last night's run learned, then collapsed. See the [long-training devlog](2026-11-25-long-training.md) and [ADR 0005](../adr/0005-stabilize-rl-training.md) for the three causes: small batches, an unbounded reward, and silent action clipping.

## How it works (big picture)
- **Smaller, steadier steps:** 4× more data per update averages out the noise, and the learning rate shrinks automatically whenever an update changes the policy too much (KL too high).
- **No huge punishments:** each tracking term now lies between 0 and its weight, and the penalties together can take away at most 10 per step. Staying alive is always worth it, and no single step can dominate an update.
- **Range pressure:** leaving ±1 no longer goes unnoticed. The penalty grows with how far out the action is.
- **Safety net:** the best evaluated policy is saved separately, so a late collapse can't destroy it.

## How verified
- Env self-check passes: zero action survives 37 steps as before (the dynamics are unchanged), rewards are within their bounds, and action 2.0 gives `act_range` = −13.5.
- Smoke runs of 40k–70k steps: training, the adaptive LR (it moves between 2e-5 and 1e-4) and checkpoint saving work. The first smoke run crashed on save because the lr lambda captured the callback, which can't be pickled. Fixed by capturing a plain float.
- **1-hour run started 11:16:** `--tag v2`, same 4 s segment. It **crashed at the first evaluation (1M steps)**: SB3 copies the normalization stats into the eval env and requires the same wrapper chain, and the eval env lacked `VecMonitor`. Fixed (`VecNormalize(VecMonitor(DummyVecEnv))`), checked with a 4k-step run where evaluation ran every 500 steps (`best_model.zip` and `best_vecnormalize.pkl` were written), then resumed from the 1M checkpoint at 11:23 for 15M more steps. Before the crash, the first 1M steps already reached `ep_len_mean` 113 with clip fraction about 0.1. Results will be added below.

## Deviations from the paper
- Reward kernel: exp instead of squared (Tab. 1), plus the penalty clip and action-range penalty. Recorded in [ADR 0004](../adr/0004-rl-tracking-simplifications.md) and [ADR 0005](../adr/0005-stabilize-rl-training.md).
- The adaptive LR rule now matches the paper, but it's applied once per PPO update rather than per mini-batch.

## Next
- Compare `v2` with last night's run in TensorBoard (`.venv/bin/tensorboard --logdir outputs/tracking`).
- If it's stable: a long run on the full 9.4 s clip.

## Result (12:30): stable, and the robot no longer falls
![v2 curves](media/v2_curves.png)

- **Stable over the whole run:**
  - KL median 0.013, max 0.032 (last night: up to 100,000).
  - Clip fraction falls from 0.17 to 0.10 (last night: 0.5).
  - Policy std decreases smoothly from 0.37 to 0.30, so the policy gets more confident instead of more random.
  - Return keeps rising slowly.
  - The learning rate ended near its floor (1.3e-5), because the KL stayed slightly above 0.01.
- **Correction, I misread the episode-length metric:** "200 = success" is wrong for the training curve. With RSI each episode starts at a *random* frame, so even a perfect policy averages about 100 steps until the clip ends. The `ep_len_mean` of ~105 (and last night's ~119 peak) therefore means **almost no falls**, not "stuck halfway".
- **Proper test of the best model** (saved at 6M steps), deterministic, starting at every 10th frame (20 starts): **0 falls; every start reaches the end of the segment.**
- Tracking quality from the clip start (mean distance of the 16 bodies to the reference over the 4 s):

  | Policy | Falls | Body error mean / max | Torso helper force mean / max |
  |---|---|---|---|
  | v2 best (6M) | 0 | 6.6 cm / 34 cm | 39 N / 360 N |
  | last night 4M | 0 | 6.4 cm / 20 cm | 55 N / 185 N |

  The helper force averages about 10 % of body weight (40.8 kg ≈ 400 N). [GIF](media/policy_05_02_poses_0-4_v2_best.gif)
- **Reading:** the fixes worked for *stability* (no collapse, safe to leave unattended). Tracking accuracy is about the same as last night's best checkpoint, so the policy has plateaued at "doesn't fall, about 6–7 cm off".

## Next
- Measure properly: Tab. 3 metrics (tracking error, foot sliding, penetration) for IK vs RL. "Body error" above is a first version of this.
- Reduce reliance on the helper force (e.g. anneal its limit), since it's non-physical.
- The full 9.4 s clip.
