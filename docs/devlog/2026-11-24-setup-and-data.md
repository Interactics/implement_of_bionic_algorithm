# 2026-11-24 · Setup, humanoid spawn, mocap data

**Blocks changed:** B1 Dataset extraction, B6 Robot model, B7 Visualization (spawn) · see [architecture](../architecture.md)

## What changed
- `.venv` (Python 3.11, uv) with `mujoco` 4.0.1 and pillow.
- `assets/humanoid.xml`: DeepMind MuJoCo humanoid (Apache-2.0), 17 bodies, 21 motors, free-floating root.
- `scripts/spawn_humanoid.py`: loads the humanoid, prints its body tree, renders `outputs/spawn.png`, runs 2 s of passive physics as a stability check.
- Repo layout: `assets/`, `scripts/`, `outputs/` (git-ignored), `README.md`, `requirements.txt`, `.gitignore`. `git init` on `main`, no commits yet.
- `dataset/CMU/`: 5 AMASS CMU clips (SMPL+H G) extracted with `scripts/extract_amass_clips.py`.

## Why
- The Mac can't run Isaac Sim, so development starts in MuJoCo with an MJCF model that Isaac Lab can import later ([ADR 0001](../adr/0001-mujoco-cpu-instead-of-isaac-lab.md)).
- AMASS is the paper's dataset (Sec. 7). The CMU subset has walking, dance and two-person clips. SMPL+H G matches the paper's SMPL body and uses each actor's real body shape.
- AMASS only offers whole sub-dataset archives (CMU ≈ 2 GB), so the script streams the archive once and keeps only the listed clips.

## How it works
- The humanoid faces +x, left is +y, z is up. AMASS is also z-up in metres, so no axis conversion is needed.
- `extract_amass_clips.py` reads `dataset/CMU.tar.bz2` and writes only the clips in its `CLIPS` table to `dataset/CMU/<subject>/<clip>.npz`.

| Clip | Motion | Length |
|---|---|---|
| 02_01, 07_01 | walk | ~2.7 s each |
| 05_02 | modern dance, pirouette | 9.4 s |
| 05_06 | modern dance, cartwheel-like start, jeté | 7.4 s |
| 60_01 | salsa | 37.4 s (60 fps source; corrected 2026-11-25, was listed as 18.7 s) |

- Each clip `.npz` holds `poses` (T×156 axis-angles, 52 joints; the first 22 are the body), `trans` (root position), `betas` (body shape), `gender`, `mocap_framerate` (120).

## How verified
- Spawn script: humanoid renders upright; after 2 s with zero torque it has fallen (head z = 0.09 m) without the sim diverging.
- Extraction: tested on a fake archive, then on the real one (5/5 clips, 7 MB total).
- `git status --ignored` shows `dataset/`, archives and `.venv` as ignored.

## Notes
- On macOS the blocking viewer (`mujoco.viewer.launch`) must run with plain `python`. `mjpython` is only for the passive viewer (`launch_passive`); the blocking viewer crashes under it.
- The built-in browser can't save downloads, so dataset downloads need a manual click. The license was accepted on the user's behalf after their explicit OK.

## Next
- Retarget the clips onto the humanoid → [IK retargeting](2026-11-24-ik-retargeting.md).
