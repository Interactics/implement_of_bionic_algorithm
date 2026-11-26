# Changelog

Newest first. Details in `docs/devlog/`, design decisions in `docs/adr/`.

## 2026-11-26
- **RL architecture figure** for slides: `docs/media/rl-architecture.svg` / `.png`, linked from the [control architecture](docs/control-architecture.md).
- **IK vs RL evaluation**: `scripts/eval_policy.py` (falls, body error, helper force, Tab. 3 metrics vs IK) + quiz 05. Walks 02_01 / 07_01: 0 falls; RL removes self-penetration, near-zero ground penetration, foot sliding −25…50 %. → [devlog](docs/devlog/2026-11-25-more-motions.md)

## 2026-11-25
- **Evaluation metrics (B8)**: `scripts/metrics.py` computes the paper's Tab. 3 metrics (ground/self penetration, foot sliding, foot floating); scored all 5 IK clips. Quiz 04. → [devlog](docs/devlog/2026-11-25-metrics.md), [quiz](docs/quiz/2026-11-25-quiz-04.md)
- **IK-only videos**: `retarget.py --gif` renders kinematic playback (no physics) beside the source skeleton; GIFs for 05_02, 07_01, 05_06. → [devlog](docs/devlog/2026-11-25-more-motions.md)
- **More motions**: `scripts/train_queue.sh` trains RL policies for 02_01, 07_01 (walks), 05_06 (dance), 60_01 (salsa, 37 s) back to back, started 21:14. → [devlog](docs/devlog/2026-11-25-more-motions.md)
- **Full-clip run**: `05_02` 0–9.3 s, 50M steps, run `full`: best model 1/20 falls, 4.8 cm mean body error, 15 N helper force. → [devlog](docs/devlog/2026-11-25-full-clip-run.md)
- **Stabilized RL training (ADR 0005)**: exp-kernel rewards, penalty clip, action-range penalty, 8×1024 rollouts, KL-adaptive LR, best-model eval (`--play --best`), `--tag` for separate runs. 1-hour test run `v2`: stable, best model never falls (6.6 cm mean body error). Quiz 03. → [quiz](docs/quiz/2026-11-25-quiz-03.md) → [devlog](docs/devlog/2026-11-25-stabilize-training.md)
- **Long training run**: 70M steps on `05_02` 0–4 s, detached (~4 h). → [devlog](docs/devlog/2026-11-25-long-training.md)
- **Training curves**: `scripts/plot_training.py` plots `progress.csv` to `<run>/curves.png` (matplotlib added). → [devlog](docs/devlog/2026-11-25-long-training.md)
- **TensorBoard**: training now also logs TensorBoard events to the run folder; view with `.venv/bin/tensorboard --logdir outputs/tracking`.
- **ADR 0005 (accepted, implemented)**: plan to stabilize RL training (bigger rollouts + adaptive LR, exp-kernel reward, action-range penalty, best-model saving). → [ADR 0005](docs/adr/0005-stabilize-rl-training.md)
- **Control architecture doc**: `docs/control-architecture.md` (outer policy loop 50 Hz, inner PD loop 200 Hz, signals, equations, learning loop) + quiz 02. → [doc](docs/control-architecture.md), [quiz](docs/quiz/2026-11-25-quiz-02.md)


## 2026-11-24
- **Quiz**: `docs/quiz/2026-11-24-quiz-01.md`, 8 concept questions on everything so far (answers collapsed). Docs are mirrored into the Obsidian vault. → [quiz](docs/quiz/2026-11-24-quiz-01.md)
- **Docs**: `docs/` with devlogs, ADRs 0001–0004 and `architecture.md` (block diagram, interfaces, RL episode diagrams).
- **RL tracking pipeline**: `scripts/tracking_env.py` (ReActor Sec. 6 env) + `scripts/train_tracking.py` (PPO train / resume / play). Smoke test learns. → [devlog](docs/devlog/2026-11-24-rl-tracking.md), [ADR 0004](docs/adr/0004-rl-tracking-simplifications.md)
- **Kinematic retargeting (IK baseline)**: `scripts/retarget.py`, shared SMPL forward kinematics in `scripts/amass.py`, skeleton player `scripts/play_amass.py`. → [devlog](docs/devlog/2026-11-24-ik-retargeting.md), [ADR 0002](docs/adr/0002-approximate-smpl-skeleton.md), [ADR 0003](docs/adr/0003-torso-correspondence.md)
- **Mocap data**: AMASS CMU (SMPL+H G), 5 clips kept in `dataset/` via `scripts/extract_amass_clips.py`; datasets git-ignored. → [devlog](docs/devlog/2026-11-24-setup-and-data.md)
- **Repo layout**: `assets/`, `scripts/`, `outputs/` (ignored), README, requirements, `.gitignore`. → [devlog](docs/devlog/2026-11-24-setup-and-data.md)
- **Spawn MuJoCo humanoid**: `assets/humanoid.xml` + `scripts/spawn_humanoid.py`; MuJoCo on CPU instead of Isaac Lab. → [ADR 0001](docs/adr/0001-mujoco-cpu-instead-of-isaac-lab.md)
