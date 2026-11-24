# ReActor reimplementation

Course project (KIT, Bionic Algorithms WS26/27): reimplementation of
*Reinforcement Learning for Physics-Aware Motion Retargeting* (ReActor).

Tasks: joint retargeting of human mocap onto a robot, RL motion tracking,
and evaluation of motion fidelity, foot sliding and collisions.

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
scripts/   runnable entry points
outputs/   generated images/videos (git-ignored)
```

## Usage

```bash
.venv/bin/python scripts/spawn_humanoid.py          # print body tree, render outputs/spawn.png
.venv/bin/mjpython scripts/spawn_humanoid.py --view # interactive viewer (macOS needs mjpython)
```

## Progress

- [x] Spawn MuJoCo humanoid
- [ ] Joint retargeting from mocap
- [ ] RL motion tracking
- [ ] Evaluation (fidelity, foot sliding, collisions)
# implement_of_bionic_algorithm
