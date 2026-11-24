# ADR 0002 · Approximate SMPL rest skeleton until the body model is downloaded

**Status:** accepted (2026-11-24)

## Context
AMASS stores joint rotations and body shape (`betas`), but joint positions need the SMPL+H body model (separate registration at mano.is.tue.mpg.de). The user chose not to download it for now.

## Decision
`scripts/amass.py` uses the real model from `dataset/smpl/{male,female}/model.npz` if it exists, and otherwise falls back to a hand-typed approximate SMPL rest skeleton (`APPROX_REST`), with a warning.

## Consequences
- Bone *directions* still come from the real mocap, and retargeting already rescales to the humanoid's bone lengths, so results look right (IK error ~2 cm).
- Actor-specific proportions are lost, which slightly affects absolute fidelity numbers. Dropping in the model later needs no code change.
