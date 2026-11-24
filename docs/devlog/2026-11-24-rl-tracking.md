# 2026-11-24 · RL motion tracking pipeline

**Blocks changed:** B4 Tracking env, B5 PPO trainer, B7 Visualization (--play) · see [architecture](../architecture.md)

## What changed
- `scripts/tracking_env.py`: gymnasium env in which the humanoid tracks an IK reference in physics (ReActor lower level, Sec. 6).
- `scripts/train_tracking.py`: PPO training (stable-baselines3) with parallel envs, checkpoints every ~1M steps, `--resume` and `--play` (live viewer, reference bodies shown as orange spheres).
- New dependencies: stable-baselines3 2.9.2, gymnasium 1.4, torch 2.15.

## Why
- Physics-based tracking is the core of ReActor. It turns the kinematic reference into motion that is physically executable (no sliding or penetration), which the IK baseline can't guarantee.

## How it works (big picture first)
- **Reference:** `outputs/retarget/<clip>.npz` resampled from 120 to 50 Hz. Body positions, orientations and velocities are precomputed by forward kinematics. `--t0/--t1` select a segment.
- **Action (Eq. 18-19):** 21 joint setpoints for PD controllers (kp = 4 × motor gear, kd = 0.1 kp, applied at the 200 Hz physics rate), plus a 6-D residual root wrench (≤400 N, ≤100 Nm) with deadband 0.1.
- **Observation (Eq. 20 + reference):** root height, projected gravity, root linear and angular velocity, joint positions and velocities, the previous two actions. Then the next-frame reference: joint angles, root height and tilt, body positions relative to the robot in its heading frame. Also the robot's own body positions and the clip phase (228-D in total).
- **Reward (Tab. 1, same weights):** negative squared errors for root xy (2), root height (10), root orientation (2), root velocities (0.5), body positions (2) and body orientations (2), plus survival (+20). Penalties for torque, joint acceleration, action rate and action acceleration, and the L1 norm of the root wrench.
- **Episodes:** start at a random reference frame (RSI). End when the torso is >0.5 m or >1 rad off the reference, or at the end of the clip.
- **PPO (Tab. 2 where possible):** 3×512 ELU MLP, γ 0.97, GAE 0.95, clip 0.2, entropy 0.0025, 5 epochs, 4 mini-batches, max grad norm 1. Observations and rewards normalized.

## How verified
- Env self-check (`python scripts/tracking_env.py`): with zero action (pure PD on the reference) the robot survives 37 steps before falling, with positive per-step reward while tracking.
- Reward debugging: the joint-acceleration term was first computed from MuJoCo's instantaneous `qacc` (contact spikes, about −60 per step), and the torque term used unsaturated PD torque. Both are fixed: acceleration is now a finite difference over the control step, and torque is the applied actuator torque.
- Smoke training on `05_02` 0–4 s, 400k steps (~2.5 min, ~2,900 steps/s on 8 envs): episode length 30 → 105 steps, return −108 → +700.
- `--play` loads the checkpoint and runs in the viewer.

## Deviations from the paper ([ADR 0004](../adr/0004-rl-tracking-simplifications.md))
- Setpoints are residuals around the reference joint angles (paper: absolute).
- RSI instead of the learned initialization phase ψ.
- No bilevel optimization of retargeting parameters (the reference is fixed).
- Fixed learning rate with a KL early stop, instead of the adaptive learning rate. Rollouts are 8 envs × 256 steps instead of 4096 × 24.
- `log_std_init = −1`: with the default exploration noise, random wrench and torque penalties make early episodes negative, and the policy learns to fall on purpose.
- Single clip, so no adaptive motion sampling.

## Next
- Longer run (20M steps ≈ 2 h) on the 4 s dance segment, then the full clip.
- Tab. 3 evaluation metrics, so the IK baseline and RL tracking can be compared.
- Bilevel optimization of the retargeting parameters (the paper's main contribution).
