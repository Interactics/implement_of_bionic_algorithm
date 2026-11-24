# ReActor reimplementation

Course project (KIT, Bionic Algorithms WS26/27): reimplementation of
*Reinforcement Learning for Physics-Aware Motion Retargeting* (ReActor).

Tasks: joint retargeting of human mocap onto a robot, RL motion tracking,
and evaluation of motion fidelity, foot sliding and collisions.

## Docs

Start with [docs/architecture.md](docs/architecture.md) (system model), then [CHANGELOG.md](CHANGELOG.md) and [docs/devlog/](docs/devlog/) for what changed and why.

## Setup

macOS can't run Isaac Sim, so local development uses MuJoCo. The robot model is
MJCF, which Isaac Lab can import later on a Linux/NVIDIA machine.

```bash
uv venv -p 3.11 .venv          # 3.11 matches Isaac Lab
uv pip install -p .venv -r requirements.txt
```

## Layout

```
assets/    robot models (assets/humanoid.xml: DeepMind MuJoCo humanoid, Apache-2.0)
scripts/   amass.py (SMPL FK), play_amass.py, retarget.py, tracking_env.py, train_tracking.py, spawn_humanoid.py, extract_amass_clips.py
outputs/   generated images/videos (git-ignored)
```

## Usage

```bash
.venv/bin/python scripts/spawn_humanoid.py          # print body tree, render outputs/spawn.png
.venv/bin/python scripts/spawn_humanoid.py --view   # interactive viewer
```

### Mocap data

Download **CMU (SMPL+H G)** from https://amass.is.tue.mpg.de into `dataset/` (registration required), then keep only the clips listed in the script:

```bash
.venv/bin/python scripts/extract_amass_clips.py dataset/CMU.tar.bz2   # -> dataset/CMU/...
```

Play a clip as a skeleton (needs the "Extended SMPL+H model" from https://mano.is.tue.mpg.de in `dataset/smpl/{male,female}/model.npz`):

```bash
.venv/bin/mjpython scripts/play_amass.py dataset/CMU/05/05_02_poses.npz        # live viewer
.venv/bin/python   scripts/play_amass.py dataset/CMU/05/05_02_poses.npz --png  # outputs/05_02_poses.png
```

Kinematic retargeting (IK, no learning) onto the humanoid, saved to `outputs/retarget/<clip>.npz` (qpos per frame) + `.png`:

```bash
.venv/bin/python   scripts/retarget.py dataset/CMU/05/05_02_poses.npz          # solve + snapshot
.venv/bin/mjpython scripts/retarget.py dataset/CMU/05/05_02_poses.npz --view   # solve + live playback
```

RL motion tracking (ReActor lower level: PD setpoints + residual root wrench, Tab. 1 rewards, PPO), reference = the IK retarget above. Runs go to `outputs/tracking/<clip>_<t0>-<t1>/`:

```bash
.venv/bin/python   scripts/train_tracking.py --clip 05_02_poses --t1 4 --steps 20e6   # train (Ctrl-C safe, checkpoints every ~1M steps)
.venv/bin/python   scripts/train_tracking.py --clip 05_02_poses --t1 4 --resume       # continue
.venv/bin/mjpython scripts/train_tracking.py --clip 05_02_poses --t1 4 --play         # watch latest checkpoint
```

## Progress

- [x] Spawn MuJoCo humanoid
- [x] Kinematic retargeting baseline (per-frame IK)
- [ ] Joint retargeting from mocap (ReActor)
- [x] RL motion tracking pipeline (single clip, fixed reference)
- [ ] Bilevel retargeting-parameter optimization
- [ ] Evaluation (fidelity, foot sliding, collisions)
# implement_of_bionic_algorithm
