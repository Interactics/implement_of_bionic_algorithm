# 2026-11-25 · Long RL training run (dance, 0–4 s)

**Blocks changed:** none (B5 PPO trainer run, no code change) · see [architecture](../architecture.md)

## What changed
- Started a detached training run: `05_02` modern dance, segment 0–4 s, **70M steps**, 8 parallel envs.
- Started 00:30 CET. Speed is ~4,800 steps/s, so it should finish around 04:30–05:00.

## Why
- The 400k-step smoke test only proved the pipeline learns (episodes 30 → 105 steps). Tracking the whole segment needs tens of millions of steps. ReActor trains for far longer on a GPU, so ~70M on a CPU is a first real test.
- The 4 s segment rather than the full 9.4 s clip, so we get a clear yes or no on one hard piece (a pirouette) before scaling up.

## How it works
- Command (detached with `nohup`, `caffeinate -i` keeps the Mac awake while on AC power):
  `nohup caffeinate -i .venv/bin/python scripts/train_tracking.py --clip 05_02_poses --t0 0 --t1 4 --steps 70e6 > outputs/tracking/train_long.log 2>&1 &`
- Output: `outputs/tracking/05_02_poses_0-4/` holds `model_<n>_steps.zip` + VecNormalize stats every ~1M steps, and `progress.csv`. This replaces the smoke run in the same folder, whose 400k checkpoint is superseded after the first 1M steps.

## How to check
- Progress: `tail -f outputs/tracking/train_long.log`. Watch `ep_len_mean`. *Correction (12:30): episodes start at random frames (RSI), so even a perfect policy averages ~100 steps; ~100 means no falls, not 200.* `ep_rew_mean` should rise with it.
- Watch the latest checkpoint (works while training): `.venv/bin/mjpython scripts/train_tracking.py --clip 05_02_poses --t0 0 --t1 4 --play`
- Stop early: `pkill -f train_tracking.py`. Continue later: same command plus `--resume`.

## Deviations from the paper
- Same as [ADR 0004](../adr/0004-rl-tracking-simplifications.md). Nothing new.

## Next
- Check the result: does `ep_len_mean` reach ~200, and do the feet still slide?
- Tab. 3 metrics (B8), so IK and RL can be compared with numbers instead of by eye.

## Result (07:00): the run collapsed
| Steps | ep_len_mean (training) | ep_rew_mean |
|---|---|---|
| 4.5M (best) | 119 of 200 | +735 |
| 7–16M | 60–75 | +250…+400, noisy |
| 16.4M → 70M | ~7 | −300 |

- Learning peaked at about 4.5M steps, plateaued, then **collapsed at 16.4M** and never recovered.
- Warning signs before the collapse: the approximate KL between consecutive policies spiked to 100–1,000+ from about 12M steps on (target 0.02), and the clip fraction sat near 0.5. *Correction: an earlier version of this note called this the value loss; I had read the wrong CSV column. The value loss stays at 0.01–1.*
- Curves: ![training curves](2026-11-25-curves.png) (made with `scripts/plot_training.py`)
- Deterministic evaluation of checkpoints (10 episodes each, per-step reward terms):
  - 4M: mean |action| 0.59, 68 steps.
  - 15M: |action| 0.80, 33 steps.
  - 70M: **|action| 1.00, meaning every action is pinned at its bound**, 7 steps. The root angular-velocity term is −53 per step (the robot spins) and the root-wrench penalty is −13 per step.
- Diagnosis:
  - The policy mean drifted outside ±1. The env clips actions, so outside the bounds the gradient carries no information.
  - The squared tracking errors are unbounded below. One spinning step costs more than the +20 survival bonus, so single updates move the policy far (KL spikes). The policy jumps into a region where actions saturate, and it can't get back.
- Proposed fix (not done yet):
  - Use bounded rewards, either an exp(−k·err²) kernel as in DeepMimic or clipping each term.
  - Add a small penalty on actions outside ±1, or squash them with tanh.
  - Lower the learning rate.
  - Keep the best checkpoint by evaluation.

## Follow-up: TensorBoard
- `train_tracking.py` now logs to TensorBoard as well as CSV (`tensorboard` 2.21 added to the requirements). New runs write `events.out.tfevents.*` into the run folder.
- View live: `.venv/bin/tensorboard --logdir outputs/tracking`, then open http://localhost:6006. The collapsed run above predates this, so it only has `progress.csv` (use `scripts/plot_training.py` for it).
- Verified with a 4k-step run, which wrote the event file. That test folder was deleted afterwards.

## Follow-up: rendered result
- Each policy was rolled out deterministically from the first frame of the clip. Orange spheres show the reference bodies.
  - **4M checkpoint: tracks the full 4 s (199 of 199 steps) without falling.** The arms drift off the reference near the end. [GIF](media/policy_4M.gif) · ![strip](media/policy_4M_strip.png)
  - 70M checkpoint: falls after 7 steps. [GIF](media/policy_70M.gif)
- Starting from random frames (the earlier evaluation) the 4M policy averaged only 68 steps, so some parts of the segment are still hard for it.
