# ADR 0004 · Simplifications in the first RL tracking pipeline

**Status:** accepted for the CPU playground (2026-11-24); revisit when porting to Isaac Lab

## Context
ReActor's lower level (Sec. 6) trains on 4,096 GPU envs for ~6 h. On the Mac we get 8 CPU envs at ~2,900 steps/s, and the goal right now is a working playground, not paper-level results.

## Decision
Keep the paper's action and observation structure, reward terms and weights, termination and network. Simplify the following:

| Paper | Here | Why |
|---|---|---|
| Absolute PD setpoints | Residual around reference joint angles (±0.5 rad) | Zero action = follow reference → much faster learning |
| Learned init phase ψ from randomized pose | Reference state initialization (RSI) | We have a robot reference pose from IK |
| Bilevel retargeting-parameter optimization | Fixed IK reference | Build the tracker first; bilevel is the next step |
| Adaptive LR, 4096×24 rollouts | Fixed LR + KL early stop, 8×256 rollouts | Few envs need longer rollouts |
| Default exploration | `log_std_init = −1` | Otherwise early episodes are net negative and the policy learns to fall |

## Consequences
- Trains on short clips in hours on a laptop.
- The residual actions and RSI must be reverted to match the paper's method for the final experiments.
