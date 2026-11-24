# ADR 0001 · MuJoCo on CPU for development, Isaac Lab later

**Status:** accepted (2026-11-24)

## Context
The course asks for Isaac Lab (the paper uses Isaac Sim with 4,096 parallel envs on an RTX 5090). The development machine is an arm64 MacBook, where Isaac Sim/Lab can't run (it needs Linux or Windows with an NVIDIA GPU).
****
## Decision
Develop locally with the `mujoco` Python package and the DeepMind MuJoCo humanoid (MJCF). Keep the robot as MJCF and the RL logic (rewards, observations, termination) framework-agnostic, so both port to Isaac Lab (MJCF importer) on a GPU machine.

## Alternatives
- **Isaac Lab now:** impossible locally; would need cloud GPU from day one.
- **MuJoCo MJX / Brax:** the same model with massive parallelism, but needs a GPU for real speedups (JAX on Metal is unreliable).

## Consequences
- Fast iteration and no cloud cost, but training runs at about 2,900 steps/s instead of millions, so only short clips are practical on the Mac.
- A port to Isaac Lab is still required for paper-scale experiments.
