# 2026-11-25 · Full-clip training run (05_02, 0–9.3 s)

**Blocks changed:** none (B5 training run, no code change) · see [architecture](../architecture.md)

## What changed
- Started a detached run on the **whole dance clip** (`05_02`, 0–9.3 s, 465 control steps), 50M steps, with the stabilized setup from [ADR 0005](../adr/0005-stabilize-rl-training.md).
- Started 15:59 CET at ~3,650 steps/s, so expected to finish around 19:45.
- Command: `nohup caffeinate -i .venv/bin/python scripts/train_tracking.py --clip 05_02_poses --t0 0 --t1 9.3 --steps 50e6 --tag full > outputs/tracking/train_full.log 2>&1 &`
- Run folder: `outputs/tracking/05_02_poses_0-9.3_full/`.

## Why
- The 4 s run was stable and its best model never fell, so the next step is the full motion: pirouettes plus the rest of the phrase.
- 50M steps instead of 16M: the clip is 2.3× longer, and the 4 s run was still improving its return slowly.

## How to check
- Live: `.venv/bin/tensorboard --logdir outputs/tracking`. `ep_len_mean` of about 230 (half of 465) means no falls, because episodes start at random frames.
- Watch the best model: `.venv/bin/mjpython scripts/train_tracking.py --clip 05_02_poses --t1 9.3 --tag full --play --best`
- Keep the Mac on power with the lid open.

## Results
- 18:20 (about 34M steps): the current best model, rolled out deterministically from frame 0, completes the **whole 9.3 s clip without falling**. [GIF](media/policy_05_02_poses_0-9.3_full_best.gif)

## Result (finished ~19:50)
![full curves](media/full_curves.png)

- **Stable over all 50M steps:** KL max 0.028, clip fraction median 0.13, policy std down smoothly from 0.37 to 0.23, value loss falling. Episode length went to about 235 within ~1M steps (≈ half of 465 = almost no falls) and stayed there. Return still creeping up.
- **Best model** (eval at 38M steps), deterministic tests:
  - 20 start frames spread over the clip: **1 fall** (start at 3.68 s, fell at 4.06 s, around the pirouette).
  - From frame 0: **completes the full 9.3 s.** Mean body error **4.8 cm** (max 10.1 cm), torso helper force mean **15 N** (max 61 N). [GIF](media/policy_05_02_poses_0-9.3_full_best.gif)

| Policy | Clip | Falls (deterministic) | Body error mean / max | Helper force mean / max |
|---|---|---|---|---|
| 4 s v2 best | 0–4 s | 0 / 20 | 6.6 / 34 cm | 39 / 360 N |
| **full best (38M)** | **0–9.3 s** | **1 / 20** | **4.8 / 10 cm** | **15 / 61 N** |

- Reading: the longer clip and longer training gave a *more accurate* policy that needs *less* helper force. More varied motion and lower final exploration noise likely help. The one failure sits near the pirouette, the hardest part.

## Next
- Tab. 3 metrics (foot sliding, ground/self-penetration, tracking error) for IK vs this policy.
- Anneal the helper force toward 0.
- Bilevel outer loop over the retargeting parameters p (see the code walkthrough: `KP` offsets, leg scale).

## Follow-up: was 50M necessary?
Same deterministic protocol (20 start frames plus a full rollout from frame 0) per checkpoint:

| Checkpoint | Falls | Body error mean / max | Helper force mean / max |
|---|---|---|---|
| 15M | 1/20 | 4.8 / 14.1 cm | 22 / 146 N |
| 25M | 1/20 | 5.0 / 13.1 cm | 16 / 82 N |
| 38M (best) | 1/20 | 4.8 / 10.1 cm | 15 / 61 N |
| 50M | 1/20 | 5.5 / 11.8 cm | 14 / 99 N |

- Accuracy and falls were settled by about 15M. Later steps mainly reduced the helper force and the worst-case error.
- Takeaway: about 15–20M steps is enough for one clip. Consider eval-based early stopping (no improvement for ~5M steps). The single remaining fall (start 3.68 s, pirouette) is the same in every checkpoint, so it needs a targeted fix (adaptive sampling, or helper-force handling), not more steps.
