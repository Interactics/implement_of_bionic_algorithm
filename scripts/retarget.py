"""Kinematic retargeting (no learning): AMASS/SMPL clip -> MuJoCo humanoid qpos via per-frame IK.

  .venv/bin/python   scripts/retarget.py [clip.npz]          # solve, save outputs/retarget/<clip>.npz + .png
  .venv/bin/mjpython scripts/retarget.py [clip.npz] --view   # same, then play it live (orange = source)

Method: source bone *directions* are kept, bone *lengths* are the humanoid's own
(root translation scaled by leg-length ratio). Then damped least-squares IK fits the
humanoid's keypoints to those targets frame by frame, warm-started, clamped to joint limits.
This is the non-physical baseline: it ignores dynamics, so foot sliding / penetration remain.
"""
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from amass import ROOT, load_clip
from play_amass import draw_skeleton

# (keypoint, humanoid body, offset in body frame, SMPL joint id(s), parent keypoint, IK weight)
KP = [
    ("root", "pelvis", (0, 0, -.04), (1, 2), None, 3.0),
    # "neck" = shoulder midpoint on both skeletons: the humanoid torso is one rigid body holding
    # the shoulders, so anchoring it at SMPL's neck joint (which sits above the shoulders) tilts it
    ("neck", "torso", (0, 0, .06), (16, 17), "root", 1.0),
    ("head", "head", (0, 0, 0), 15, "neck", 0.5),
]
for s, side, (hip, knee, ank, toe, sho, elb, wri) in [("r", "right", (2, 5, 8, 11, 17, 19, 21)),
                                                       ("l", "left", (1, 4, 7, 10, 16, 18, 20))]:
    KP += [
        (f"hip_{s}", f"thigh_{side}", (0, 0, 0), hip, "root", 1.0),
        (f"knee_{s}", f"shin_{side}", (0, 0, .02), knee, f"hip_{s}", 1.0),  # knee joint anchor
        (f"ankle_{s}", f"foot_{side}", (0, 0, .08), ank, f"knee_{s}", 2.0),  # ankle_y anchor
        (f"toe_{s}", f"foot_{side}", (.14, 0, 0), toe, f"ankle_{s}", 2.0),
        (f"shoulder_{s}", f"upper_arm_{side}", (0, 0, 0), sho, "neck", 1.0),
        (f"elbow_{s}", f"lower_arm_{side}", (0, 0, 0), elb, f"shoulder_{s}", 1.0),
        (f"wrist_{s}", f"hand_{side}", (0, 0, 0), wri, f"elbow_{s}", 1.0),
    ]
NAMES = [k[0] for k in KP]
PARENT = [NAMES.index(k[4]) if k[4] else -1 for k in KP]
W = np.repeat([k[5] for k in KP], 3)

model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/humanoid.xml"))
data = mujoco.MjData(model)
BODY = [model.body(k[1]).id for k in KP]
OFF = np.array([k[2] for k in KP], dtype=float)
HINGES = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and model.jnt_limited[j]]


def keypoints(d):
    return np.array([d.xpos[b] + d.xmat[b].reshape(3, 3) @ o for b, o in zip(BODY, OFF)])


def make_targets(P):
    """Source joints (T,22,3) -> humanoid-proportioned keypoint targets (T,K,3)."""
    S = np.stack([P[:, list(np.atleast_1d(k[3]))].mean(1) for k in KP], 1)
    mujoco.mj_kinematics(model, data)  # data at qpos0 = humanoid rest pose
    H0 = keypoints(data)
    L = np.array([np.linalg.norm(H0[k] - H0[p]) if p >= 0 else 0 for k, p in enumerate(PARENT)])
    leg = [NAMES.index(n) for n in ("knee_r", "ankle_r", "knee_l", "ankle_l")]
    src_leg = np.mean([np.linalg.norm(S[:, k] - S[:, PARENT[k]], axis=-1).mean() for k in leg])
    scale = L[leg].mean() / src_leg
    T = np.zeros_like(S)
    T[:, 0] = S[:, 0] * scale
    for k in range(1, len(KP)):  # parents come before children in KP
        v = S[:, k] - S[:, PARENT[k]]
        T[:, k] = T[:, PARENT[k]] + L[k] * v / np.linalg.norm(v, axis=-1, keepdims=True)
    return T, scale


def solve_ik(T, iters_first=200, iters=15, damping=1e-2, max_step=0.1):
    jacp = np.zeros((3, model.nv))
    q = model.qpos0.copy()
    q[:3] = T[0, 0] - (keypoints(data)[0] - data.qpos[:3])  # start with root at its target
    Q, err = np.zeros((len(T), model.nq)), np.zeros(len(T))
    for f in range(len(T)):
        # root orientation straight from the targets (pelvis frame), so spins/pirouettes don't
        # leave the IK stuck in a twisted local minimum; the joints then refine from the warm start
        y = T[f, NAMES.index("hip_l")] - T[f, NAMES.index("hip_r")]
        z = T[f, NAMES.index("neck")] - T[f, 0]
        x = np.cross(y, z)
        R = np.stack([x, np.cross(z, x), z], 1)
        R /= np.linalg.norm(R, axis=0)
        mujoco.mju_mat2Quat(q[3:7], R.flatten())
        for _ in range(iters_first if f == 0 else iters):
            data.qpos[:] = q
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            X = keypoints(data)
            J = np.zeros((3 * len(KP), model.nv))
            for k, (b, x) in enumerate(zip(BODY, X)):
                mujoco.mj_jac(model, data, jacp, None, x, b)
                J[3 * k:3 * k + 3] = jacp
            e = (T[f] - X).ravel()
            JW = J * W[:, None]
            dq = np.linalg.solve(JW.T @ J + damping * np.eye(model.nv), JW.T @ e)
            dq *= min(1.0, max_step / np.abs(dq).max())  # trust region: no giant first steps
            mujoco.mj_integratePos(model, q, dq, 1.0)
            for j in HINGES:
                a = model.jnt_qposadr[j]
                q[a] = np.clip(q[a], *model.jnt_range[j])
        Q[f] = q
        err[f] = np.linalg.norm(T[f] - X, axis=-1).mean()
    return Q, err


def ground(Q):
    """Shift the clip vertically so the feet stand on the floor (one offset per clip)."""
    lowest = []
    for q in Q:
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        lowest.append(min(data.geom_xpos[g][2] for g in range(model.ngeom)
                          if model.geom_bodyid[g] in (model.body("foot_right").id, model.body("foot_left").id)))
    Q[:, 2] += 0.027 - np.percentile(lowest, 5)  # foot capsule radius
    return Q


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    clip = Path(args[0]) if args else ROOT / "dataset/CMU/05/05_02_poses.npz"
    P, fps = load_clip(clip)
    T, scale = make_targets(P)
    t0 = time.time()
    Q, err = solve_ik(T)
    Q = ground(Q)
    print(f"{clip.name}: {len(Q)} frames, scale {scale:.2f}, IK {time.time() - t0:.1f}s, "
          f"keypoint error mean {err.mean() * 100:.1f} cm / max {err.max() * 100:.1f} cm")
    out = ROOT / "outputs/retarget" / f"{clip.stem}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, qpos=Q, fps=fps, source=str(clip))

    side = np.array([0, 1.2, 0])  # source skeleton drawn next to the robot
    src = P * scale + side - np.array([0, 0, P[:, :, 2].min() * scale])

    from PIL import Image

    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 4.0, 150, -10
    imgs = []
    with mujoco.Renderer(model, 480, 400) as r:
        for f in np.linspace(0, len(Q) - 1, 4).astype(int):
            data.qpos[:] = Q[f]
            mujoco.mj_forward(model, data)
            cam.lookat[:] = Q[f, :3] + side / 2
            r.update_scene(data, cam)
            draw_skeleton(r.scene, src[f])
            imgs.append(r.render())
    Image.fromarray(np.concatenate(imgs, axis=1)).save(out.with_suffix(".png"))
    print("saved", out, "and", out.with_suffix(".png"))

    if "--view" in sys.argv:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(model, data) as v:
            t0 = time.time()
            while v.is_running():
                f = int((time.time() - t0) * fps) % len(Q)
                with v.lock():
                    data.qpos[:] = Q[f]
                    mujoco.mj_forward(model, data)  # kinematic playback, no physics
                    v.user_scn.ngeom = 0
                    draw_skeleton(v.user_scn, src[f])
                v.sync()
                time.sleep(1 / 60)
