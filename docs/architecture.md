# Architecture

> Software view (scripts, data flow). For the **control / RL view** (loops, signals, rates, equations) see [control-architecture.md](control-architecture.md).

System model of the ReActor reimplementation. Diagrams are Mermaid (they render on GitHub). **Update this file whenever a block, interface or flow changes**; devlogs name the blocks they touch by ID (B1…B8).

## 1. Block diagram

```mermaid
flowchart LR
    subgraph Data
        B1["B1 Dataset extraction<br/>extract_amass_clips.py"]
        B2["B2 SMPL forward kinematics<br/>amass.py"]
    end
    subgraph Retargeting
        B3["B3 Kinematic retargeting (IK)<br/>retarget.py"]
    end
    subgraph RL["RL tracking (ReActor lower level)"]
        B4["B4 Tracking env<br/>tracking_env.py"]
        B5["B5 PPO trainer<br/>train_tracking.py"]
    end
    B6["B6 Robot model<br/>assets/humanoid.xml"]
    B7["B7 Visualization<br/>spawn_humanoid / play_amass / --view / --play"]
    B8["B8 Evaluation<br/>metrics.py, eval_policy.py"]

    AR[("AMASS CMU archive")] --> B1 -->|clip .npz| B2
    B2 -->|joints T×22×3| B3
    B6 --> B3
    B3 -->|reference qpos T×28| B4
    B6 --> B4
    B4 <-->|obs 228 / action 27 / reward| B5
    B5 -->|checkpoints| B7
    B3 -->|reference qpos| B7
    B2 -->|joints| B7
    B3 -->|IK qpos| B8
    B5 -->|policy rollouts| B8
    B8 -.->|"bilevel update of p (planned)"| B3
```

| Block | Implemented in | Input | Output | Status |
|---|---|---|---|---|
| B1 Dataset extraction | `scripts/extract_amass_clips.py` | `dataset/CMU.tar.bz2` | `dataset/CMU/<subj>/<clip>.npz` | done |
| B2 SMPL forward kinematics | `scripts/amass.py` | clip `.npz` (+ optional `dataset/smpl/*/model.npz`) | joint positions (T, 22, 3), fps | done (approx. skeleton, ADR 0002) |
| B3 Kinematic retargeting | `scripts/retarget.py` | joints, robot model | `outputs/retarget/<clip>.npz` (qpos), `.png` | done |
| B4 Tracking env | `scripts/tracking_env.py` | reference qpos, robot model, action | obs, reward, done | done (ADR 0004) |
| B5 PPO trainer | `scripts/train_tracking.py` | B4 (8 parallel envs) | `outputs/tracking/<run>/model_*.zip`, `progress.csv` | done |
| B6 Robot model | `assets/humanoid.xml` | — | MJCF, 21 actuated hinges + free root | done |
| B7 Visualization | `spawn_humanoid.py`, `play_amass.py`, `retarget.py --view`, `train_tracking.py --play` | any of the above | viewer / PNG | done |
| B8 Evaluation | `metrics.py`, `eval_policy.py` | B3 qpos (+ source clip for contacts), B5 best policy | falls, body error, helper force; ground/self penetration, foot sliding, foot floating (IK vs RL) | done |

## 2. Interfaces (data contracts)

| Interface | File / object | Shape | Units / frame | Producer → consumer |
|---|---|---|---|---|
| AMASS clip | `dataset/CMU/.../*_poses.npz` | `poses` T×156, `trans` T×3, `betas` 16, `gender`, `mocap_framerate` | axis-angle rad; m; z-up; 120 fps | B1 → B2 |
| Source joints | `np.ndarray` | T×22×3 | m, world z-up, SMPL joint order (`amass.NAMES`) | B2 → B3, B7 |
| Keypoint map | `retarget.KP` | 17 entries (body, offset, SMPL ids, parent, weight) | m, body frame | B3 internal |
| Reference motion | `outputs/retarget/<clip>.npz` | `qpos` T×28, `fps` | root pos m + quat (w,x,y,z), 21 hinge angles rad; 120 fps | B3 → B4, B7 |
| Observation | `TrackingEnv._obs()` | 228 | m, rad, m/s, rad/s; heading frame for positions | B4 → B5 |
| Action | policy output | 27 = 21 joints + 6 wrench | [−1, 1] → ±0.5 rad residual setpoints; ±400 N, ±100 Nm (deadband 0.1) | B5 → B4 |
| Reward | `TrackingEnv._reward()` | scalar (+ `last_terms` dict) | weights of paper Tab. 1 | B4 → B5 |
| Checkpoint | `outputs/tracking/<clip>_<t0>-<t1>/model_<n>_steps.zip` + `model_vecnormalize_<n>_steps.pkl` | — | SB3 PPO + obs/reward normalization | B5 → B7 |

## 3. One RL episode (B4 ↔ B5)

```mermaid
sequenceDiagram
    participant P as PPO policy (B5)
    participant E as TrackingEnv (B4)
    participant S as MuJoCo physics
    P->>E: reset()
    E->>S: RSI: set qpos/qvel to reference frame f (random) + noise
    E-->>P: obs₀
    loop every 20 ms (50 Hz) until done
        P->>E: action (21 residual setpoints + 6 wrench)
        E->>S: 4 × [PD torque = kp(q_ref+0.5a − q) − kd·q̇, clip to motor limit] + root wrench
        S-->>E: new state
        E->>E: reward = Σ tracking terms + 20 survival − regularization (Tab. 1)
        alt torso error > 0.5 m or > 1 rad
            E-->>P: obs, reward 0, terminated
        else reached end of clip
            E-->>P: obs, reward, truncated
        else
            E-->>P: obs, reward
        end
    end
```

```mermaid
stateDiagram-v2
    [*] --> Reset
    Reset --> Tracking: RSI at random reference frame
    Tracking --> Tracking: step (reward)
    Tracking --> Terminated: torso pos err > 0.5 m / ori err > 1 rad
    Tracking --> Truncated: end of reference clip
    Terminated --> Reset
    Truncated --> Reset
```
