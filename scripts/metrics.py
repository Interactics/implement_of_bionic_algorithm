"""Kinematic evaluation metrics (ReActor Tab. 3) for a humanoid qpos sequence.

  .venv/bin/python scripts/metrics.py outputs/retarget/05_02_poses.npz [...]   # IK results
  .venv/bin/python scripts/metrics.py --check                                 # self-check

Ground penetration: fraction of frames with floor penetration > 1 cm; depth = mean over those frames
                    (deepest contact per frame).
Self-penetration:   same, for body-body contacts; pairs on the same kinematic chain are ignored.
Foot sliding:       mean horizontal foot speed while the *reference* foot is in contact.
Foot floating:      mean foot-floor distance while the reference foot is in contact.
Reference contact: source foot joint low (< 5 cm above its floor) and slow (< 0.3 m/s), after
Shimada et al. 2020 (height + velocity heuristic).
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

from amass import ROOT, load_clip

model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/humanoid.xml"))
data = mujoco.MjData(model)
FLOOR = model.geom("floor").id
FEET = [[model.geom(f"foot{i}_{s}").id for i in (1, 2)] for s in ("left", "right")]
SMPL_FEET = [(7, 10), (8, 11)]  # (ankle, toe) left, right
PEN_THR = 0.01


def chain(b):
    out = set()
    while b:
        out.add(b)
        b = model.body_parentid[b]
    return out


CHAIN = [chain(b) for b in range(model.nbody)]


def same_chain(b1, b2):
    return b1 in CHAIN[b2] or b2 in CHAIN[b1]


def reference_contact(P, fps, h=0.05, v=0.3):
    """Source joints (T,J,3) -> (T,2) bool foot contact, left/right."""
    C = np.zeros((len(P), 2), bool)
    for s, joints in enumerate(SMPL_FEET):
        for j in joints:
            z = P[:, j, 2] - np.percentile(P[:, j, 2], 5)
            speed = np.linalg.norm(np.gradient(P[:, j, :2], axis=0), axis=-1) * fps
            C[:, s] |= (z < h) & (speed < v)
    return C


def evaluate(Q, fps, contact):
    ground, self_pen, float_d, foot_xy = [], [], [], []
    fromto = np.zeros(6)
    for q in Q:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        g = s = 0.0
        for c in data.contact[:data.ncon]:
            if c.dist >= 0:
                continue
            if FLOOR in (c.geom1, c.geom2):
                g = max(g, -c.dist)
            elif not same_chain(model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]):
                s = max(s, -c.dist)
        ground.append(g)
        self_pen.append(s)
        float_d.append([max(0.0, min(mujoco.mj_geomDistance(model, data, f, FLOOR, 1.0, fromto) for f in foot))
                        for foot in FEET])
        foot_xy.append([data.geom_xpos[foot, :2].mean(0) for foot in FEET])
    ground, self_pen, float_d = map(np.array, (ground, self_pen, float_d))
    speed = np.linalg.norm(np.gradient(np.array(foot_xy), axis=0), axis=-1) * fps  # (T,2)

    def pen(d):
        bad = d > PEN_THR
        return bad.mean(), d[bad].mean() * 100 if bad.any() else 0.0

    return dict(zip(["ground_time", "ground_cm", "self_time", "self_cm"], pen(ground) + pen(self_pen)),
                slide_cm_s=speed[contact].mean() * 100 if contact.any() else 0.0,
                float_cm=float_d[contact].mean() * 100 if contact.any() else 0.0,
                contact_frac=contact.mean())


def check():
    q = model.qpos0.copy()
    q[2] = 1.4  # standing pose, feet above the floor
    T = 20
    Q = np.repeat(q[None], T, 0)
    C = np.ones((T, 2), bool)
    m = evaluate(Q, 30, C)
    assert m["ground_time"] == 0 and m["slide_cm_s"] < 1e-6, m
    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)
    lowest = min(data.geom_xpos[g][2] for f in FEET for g in f)
    Q[:, 2] -= lowest  # sink feet ~one capsule radius into the floor
    m = evaluate(Q, 30, C)
    assert m["ground_time"] == 1 and m["ground_cm"] > 1, m
    Q[:, 0] = np.arange(T) * 0.01  # slide 1 cm/frame at 30 fps = 30 cm/s
    assert abs(evaluate(Q, 30, C)["slide_cm_s"] - 30) < 1, "slide"
    print("metrics self-check ok")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
        sys.exit()
    print(f"{'clip':14s} {'contact':>7s} | {'ground pen time / cm':>20s} | {'self pen time / cm':>18s} | "
          f"{'slide cm/s':>10s} | {'float cm':>8s}")
    for f in sys.argv[1:]:
        r = np.load(f)
        P, fps = load_clip(ROOT / str(r["source"]))
        m = evaluate(r["qpos"], float(r["fps"]), reference_contact(P, fps))
        print(f"{Path(f).stem:14s} {m['contact_frac']:7.2f} | {m['ground_time']:9.2f} / {m['ground_cm']:6.1f}"
              f"    | {m['self_time']:7.2f} / {m['self_cm']:6.1f}    | {m['slide_cm_s']:10.1f} | {m['float_cm']:8.1f}")
