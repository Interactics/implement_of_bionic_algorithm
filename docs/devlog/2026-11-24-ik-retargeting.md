# 2026-11-24 · Kinematic retargeting (IK baseline, no learning)

**Blocks changed:** B2 SMPL forward kinematics, B3 Kinematic retargeting, B7 Visualization · see [architecture](../architecture.md)

## What changed
- `scripts/amass.py`: SMPL forward kinematics. AMASS clip → (T, 22, 3) joint positions, shared by the other scripts.
- `scripts/play_amass.py`: plays a clip as a stick-figure skeleton (live viewer, or `--png` snapshots).
- `scripts/retarget.py`: per-frame IK from SMPL joints to humanoid joint angles. Writes `outputs/retarget/<clip>.npz` (qpos per frame, 120 fps) plus a snapshot `.png`; `--view` plays it live.

## Why
- The RL tracker needs a robot-space reference motion, and this is the non-physical baseline the paper compares against (optimization-based retargeters like GMR / OmniRetarget, Sec. 7.1).

## How it works (big picture first)
1. **Human joints:** forward kinematics on the clip's joint rotations, using the actor's rest skeleton (or an approximate one, [ADR 0002](../adr/0002-approximate-smpl-skeleton.md)).
2. **Correspondence:** 17 hand-picked keypoint pairs: pelvis (hip midpoint), shoulder midpoint, head, and per side the hip, knee, ankle, toe, shoulder, elbow and wrist ([ADR 0003](../adr/0003-torso-correspondence.md)).
3. **Targets in robot proportions:** each bone keeps the human's *direction* but takes the robot's *length*. The root position is scaled by the leg-length ratio (0.91).
4. **IK per frame:** damped least squares on MuJoCo point Jacobians (`mj_jac`), weighted (feet ×2, root ×3), warm-started from the previous frame. Steps are capped at 0.1 rad (trust region) and hinge joints are clamped to their limits. The root orientation is initialized each frame from the target pelvis frame, so spins don't get stuck.
5. **Grounding:** one vertical offset per clip, so that the 5th percentile of the lowest foot point sits on the floor.

## How verified
Mean / max keypoint distance between robot and targets:

| Clip | Mean | Max |
|---|---|---|
| 02_01 walk | 1.7 cm | 3.0 cm |
| 07_01 walk | 1.9 cm | 3.6 cm |
| 05_02 dance | 2.2 cm | 20.8 cm |
| 05_06 dance | 2.6 cm | 14.8 cm |
| 60_01 salsa | 2.2 cm | 18.3 cm |

The large maxima are frames where the dancer extends a hip further back than the humanoid allows (+20°). Visual check: snapshots side by side with the source skeleton.

Bug found on the way: mapping the robot's torso to SMPL's neck joint tilted the rigid torso, which pushed the abdomen to its −75° limit and gave 9–11 cm errors. Mapping it to the shoulder midpoint fixed it ([ADR 0003](../adr/0003-torso-correspondence.md)).

## Deviations from the paper
- The paper's retargeting is *learned jointly* with the tracking policy (bilevel, Sec. 3–5). This is a fixed per-frame IK instead: the baseline.
- It ignores physics, so foot sliding, floating and self-penetration are not prevented.

## Next
- Use these references for RL tracking → [RL tracking](2026-11-24-rl-tracking.md).
- Evaluation metrics from Tab. 3 (ground penetration, self-penetration, foot sliding, foot floating) on this baseline.
