# ADR 0005 · Stabilize RL tracking training

**Status:** accepted (2026-11-25), implemented the same day. See the [devlog](../devlog/2026-11-25-stabilize-training.md).
**Blocks affected:** B4 Tracking env (reward, action handling), B5 PPO trainer (rollouts, learning rate, best-model saving) · see [architecture](../architecture.md)

## Context: what went wrong
The 70M-step run on `05_02` 0–4 s ([devlog](../devlog/2026-11-25-long-training.md)) learned, then collapsed:

| Steps | Episode length (of 200) | What the policy does |
|---|---|---|
| 4M (best) | 119 in training. **199/199 from the clip start** (deterministic) | Dances the whole segment; arms drift near the end |
| 16.4M | drops to ~7 | Collapse |
| 70M | ~7 | Every action pinned at ±1 (mean \|a\| = 1.00), robot spins and falls in 0.14 s |

Evidence from the curves ([plot](../devlog/2026-11-25-curves.png)):
- **Approximate KL** between consecutive policies spikes to 100–100,000 (target 0.02). Single updates move the policy far.
- **Clip fraction** is stuck near 0.5: half the samples sit outside PPO's trust region on every update.
- **Policy std** rises after the collapse, so the policy gets more random instead of settling.
- Value loss stays small (0.01–1), so the critic is *not* the problem.

## Root causes (three, they reinforce each other)
1. **Too few samples per update.** 8 envs × 256 steps = 2,048 samples, split into 4 mini-batches of 512, 5 epochs. The paper uses 4,096 × 24 ≈ 98,000. Small batches give noisy gradients, and with a fixed learning rate of 3e-4 one noisy step can move the policy a lot. SB3's `target_kl` only stops the epoch *after* the jump is measured.
2. **Unbounded reward.** Every tracking term is `−w · error²` with no floor. When the robot starts spinning, the root angular-velocity term alone reaches −53 per step, more than the +20 survival bonus. These outlier returns produce outlier advantages, which produce big policy steps.
3. **Actions clipped silently.** The policy outputs a Gaussian. The env does `np.clip(a, −1, 1)`, so once the policy's mean drifts past ±1 the robot behaves the same whatever the mean does. The policy gets no signal to come back, and it stays pinned.

Cause 1 makes the jumps possible, cause 2 triggers them, and cause 3 makes them permanent.

## Decision (proposed): four changes

### 1. Bigger, steadier policy updates (B5)
| Setting | Now | Proposed | Why |
|---|---|---|---|
| Rollout length `n_steps` | 256 | **1024** | 8 × 1024 = 8,192 samples per update, 4× less gradient noise |
| Mini-batches | 4 (512 each) | 4 (2,048 each) | Same count, bigger batches |
| Learning rate | fixed 3e-4 | **adaptive**, starting at 1e-4, kept in [1e-5, 1e-3] | The paper's method (Tab. 2). After each update: KL > 2 × 0.01 → lr ÷ 1.5; KL < 0.01 / 2 → lr × 1.5 |
| `target_kl` early stop | 0.02 | 0.02 (kept) | Extra safety net |

The adaptive rule is the one used by `rsl_rl`, the PPO library behind Isaac Lab, so it carries over directly when we port.

### 2. Bounded reward with exp kernels (B4)
Each tracking term changes from `−w · err²` to `w · exp(−err² / σ²)`. Each term then lies in **[0, w]**: perfect tracking gives w, bad tracking tends to 0, and it can never go below 0. This is the DeepMimic form, and it's common in motion-tracking work.

| Term | Weight (unchanged) | σ (error scale where the reward drops to 37 %) |
|---|---|---|
| root xy position | 2 | 0.3 m |
| root height | 10 | 0.1 m |
| root orientation | 2 | 0.5 rad |
| root linear velocity | 0.5 | 1 m/s |
| root angular velocity | 0.5 | 2 rad/s |
| body positions (mean over bodies) | 2 | 0.15 m |
| body orientations (mean over bodies) | 2 | 0.5 rad |

- Survival (+20) and the regularization penalties (torque, joint acceleration, action rate and acceleration, wrench) stay as they are.
- The **sum of the penalties is clipped at −10 per step**, so a single bad step can't outweigh survival.

**This departs from the paper**, whose Tab. 1 uses squared losses. That's the trade-off: squared losses do work with ~100k-sample batches on a GPU, but they are fragile at our batch size. When porting to Isaac Lab with 4,096 envs, we can switch back to Tab. 1 and compare.

### 3. Keep actions in range (B4)
- The action space becomes ±3 instead of ±1. The env still clips to ±1 before applying the action.
- New penalty `0.5 · Σ max(0, |a| − 1)²`, so the policy *feels* it when it leaves the range and is pulled back. The robot's behaviour within ±1 is unchanged.

### 4. Save the best model (B5)
- Every 1M steps, evaluate 10 deterministic episodes from random start frames, and save `best_model.zip` and its normalization statistics when the mean episode length improves.
- `--play` gets a `--best` flag to load it.
- A late collapse can then never destroy a good policy again.

## Alternatives considered
- **Only lower the learning rate.** Cheapest, but it doesn't fix causes 2 and 3; it just makes the collapse slower.
- **Clip the per-step reward instead of using exp kernels.** Simpler, but a clipped term has zero gradient once it's saturated, so the policy can't tell "bad" from "very bad".
- **More parallel envs.** The Mac has 8 cores, and more processes than cores slows down every env.
- **tanh-squashed actions.** Also bounds the action, but it changes the policy's log-probability (needs SB3's SAC-style squashing) and saturates the gradient the same way. The penalty is simpler.

## Verification plan
1. Unit check: the env self-check still passes, all reward terms are within their bounds, and a zero action survives as before.
2. **1-hour run** (~15M steps) on the same 4 s segment, compared with last night in TensorBoard.
3. Success criteria:
   - episode length keeps rising, with no collapse
   - KL stays below about 0.05
   - clip fraction below 0.3
   - the best model tracks 200/200 steps from random start frames, not just from the clip start
4. If it passes: a long run, then the full 9.4 s clip.

## Consequences
- Training should become stable enough to leave unattended.
- Rewards are no longer comparable to last night's run (different scale), so compare episode length and tracking error instead.
- Two deviations from the paper get added to [ADR 0004](0004-rl-tracking-simplifications.md): exp-kernel reward, and the action-range penalty.
