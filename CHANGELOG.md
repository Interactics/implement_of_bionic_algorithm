# Changelog

Newest first. Details in `docs/devlog/`, design decisions in `docs/adr/`.

## 2026-11-24
- **Quiz**: `docs/quiz/2026-11-24-quiz-01.md`, 8 concept questions on everything so far (answers collapsed). Docs are mirrored into the Obsidian vault. → [quiz](docs/quiz/2026-11-24-quiz-01.md)
- **Docs**: `docs/` with devlogs, ADRs 0001–0004 and `architecture.md` (block diagram, interfaces, RL episode diagrams).
- **RL tracking pipeline**: `scripts/tracking_env.py` (ReActor Sec. 6 env) + `scripts/train_tracking.py` (PPO train / resume / play). Smoke test learns. → [devlog](docs/devlog/2026-11-24-rl-tracking.md), [ADR 0004](docs/adr/0004-rl-tracking-simplifications.md)
- **Kinematic retargeting (IK baseline)**: `scripts/retarget.py`, shared SMPL forward kinematics in `scripts/amass.py`, skeleton player `scripts/play_amass.py`. → [devlog](docs/devlog/2026-11-24-ik-retargeting.md), [ADR 0002](docs/adr/0002-approximate-smpl-skeleton.md), [ADR 0003](docs/adr/0003-torso-correspondence.md)
- **Mocap data**: AMASS CMU (SMPL+H G), 5 clips kept in `dataset/` via `scripts/extract_amass_clips.py`; datasets git-ignored. → [devlog](docs/devlog/2026-11-24-setup-and-data.md)
- **Repo layout**: `assets/`, `scripts/`, `outputs/` (ignored), README, requirements, `.gitignore`. → [devlog](docs/devlog/2026-11-24-setup-and-data.md)
- **Spawn MuJoCo humanoid**: `assets/humanoid.xml` + `scripts/spawn_humanoid.py`; MuJoCo on CPU instead of Isaac Lab. → [ADR 0001](docs/adr/0001-mujoco-cpu-instead-of-isaac-lab.md)
