# 2026-11-25 · RL tracking on the other motions (queue)

**Blocks changed:** B5 (new `scripts/train_queue.sh`, no change to the env or trainer) · see [architecture](../architecture.md)

## What changed
- `scripts/train_queue.sh` trains one policy per clip, one after another (8 CPU cores, so runs can't go in parallel without slowing each other down). Each run goes to `outputs/tracking/<clip>_0-<t1>_q/`, logs to `outputs/tracking/train_<clip>_q.log`, and the queue progress goes to `outputs/tracking/queue.log`.
- Started 21:14, detached (`nohup caffeinate -i`), at about 3,450 steps/s:

| Clip | Motion | Length | Steps | Expected done |
|---|---|---|---|---|
| 02_01 | walk | 2.8 s | 8M | ~21:55 |
| 07_01 | walk | 2.6 s | 8M | ~22:35 |
| 05_06 | modern dance, cartwheel-like start, jeté | 7.3 s | 15M | ~23:50 |
| 60_01 | salsa | 37.3 s | 30M | ~02:15 |

## Why
- The 05_02 dance is solved (full clip: 1/20 falls, 4.8 cm). The next question is whether the same setup works across different motions.
- Step budgets follow the [full-clip finding](2026-11-25-full-clip-run.md) that progress flattens after ~15M steps for a 9 s clip: shorter for the walks, longer for the 37 s salsa.

## Correction
- Salsa `60_01` is **37.4 s**, not 18.7 s. Its source is 60 fps, and the earlier table assumed 120 fps. The IK retarget already used the right fps.

## IK-only baseline videos
- `scripts/retarget.py --gif` saves `outputs/retarget/<clip>_ik.gif`: the robot replays the IK joint angles kinematically (no physics) next to the orange source skeleton, 25 fps.
- Keypoint error mean / max: 05_02 2.2 / 20.8 cm, 07_01 1.9 / 3.6 cm, 05_06 2.6 / 14.8 cm (max errors at the spins).
- Poses match well, but feet slide and float/sink because nothing enforces contact or balance. That is the gap the RL tracker closes.
- [05_02](media/05_02_poses_ik.gif) · [07_01](media/07_01_poses_ik.gif) · [05_06](media/05_06_poses_ik.gif)

## Results (walks, 2026-11-26)
- The queue was paused overnight on request (21:44 → 14:22) and again at 15:48 during 05_06 (8.8M of 15M). Walks are done; 05_06 and 60_01 are pending.
- New `scripts/eval_policy.py <run>`: falls from about 20 start frames, a rollout from frame 0 (body error, helper force), and the Tab. 3 metrics for RL vs IK on the same segment.

| Clip | Falls | Body error mean / max | Helper force mean / max |
|---|---|---|---|
| 02_01 walk | 0 / 22 | 5.2 / 15.5 cm | 36 / 510 N |
| 07_01 walk | 0 / 20 | 6.8 / 36.1 cm | 99 / 624 N |
| 05_02 dance (full run, for reference) | 1 / 20 | 4.8 / 10.1 cm | 16 / 62 N |

| Clip | Method | Ground pen. time / depth | Self pen. time / depth | Foot slide | Foot float |
|---|---|---|---|---|---|
| 02_01 | IK | 0.23 / 2.6 cm | 0.31 / 2.7 cm | 20.5 cm/s | 1.0 cm |
| 02_01 | RL | 0.01 / 1.4 cm | 0 | 15.6 cm/s | 0.0 cm |
| 07_01 | IK | 0.12 / 1.5 cm | 0.10 / 2.4 cm | 44.9 cm/s | 1.5 cm |
| 07_01 | RL | 0.01 / 1.5 cm | 0 | 23.1 cm/s | 0.0 cm |
| 05_02 | IK | 0.49 / 1.5 cm | 0 | 30.3 cm/s | 1.6 cm |
| 05_02 | RL | 0.00 / 1.3 cm | 0 | 21.8 cm/s | 2.0 cm |

- **Reading:**
  - RL removes self-penetration and almost all ground penetration, because the simulator's contacts push back.
  - It also cuts foot sliding by 25–50 %. This is the paper's main claim, reproduced on our robot.
  - Sliding doesn't go to zero, as the paper also reports: the policy isn't forced to copy the contact pattern exactly.
- **Caveat:** 07_01 leans on the helper force (99 N mean, about 25 % of body weight), so part of its "physical" walk is not physical yet. Helper-force annealing is the fix.
- **05_06 dance (15M, done 16:20):** 0 / 21 falls, but it tracks poorly. The body error is 13.9 / 37.0 cm, and the helper force is 153 N on average (about 38 % of body weight). Ground penetration drops (0.22 → 0.01) and self-penetration goes to 0, but foot sliding is *worse* than IK (46.9 vs 40.5 cm/s). The policy "survives" mainly by leaning on the helper force and shuffling its feet, while the arms lag behind. Likely causes: the cartwheel-like start and the jeté are beyond what 15M steps learn, and the hip only extends +20°. Next: more steps, or start the segment after the cartwheel. [GIF](media/policy_05_06_poses_0-7.3_q_best.gif)
- GIFs: [02_01](media/policy_02_01_poses_0-2.8_q_best.gif) · [07_01](media/policy_07_01_poses_0-2.6_q_best.gif)

