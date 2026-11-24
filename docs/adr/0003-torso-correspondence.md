# ADR 0003 · Map the humanoid torso to SMPL's shoulder midpoint, not the neck

**Status:** accepted (2026-11-24)

## Context
The humanoid's torso is one rigid body that carries the shoulders and head. SMPL has a flexible spine, and its neck joint sits *above* the shoulder line. Mapping the torso origin to SMPL's neck (joint 12) made the shoulder targets ~13 cm lower than the rigid torso allows. The IK compensated by tilting the torso and drove the abdomen to its −75° limit, giving 9–11 cm mean errors even on walking.

## Decision
Use the torso point `(0, 0, 0.06)`, at shoulder height, paired with the midpoint of SMPL's shoulders (joints 16 and 17). Shoulder targets then branch out purely sideways on both skeletons.

## Consequences
- Mean IK error dropped from ~9–11 cm to ~2 cm.
- Lesson for later correspondences (and the paper's Fig. 4): pair points that sit at the same place on rigid parts of both skeletons, not joints that merely share a name.
