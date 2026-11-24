"""AMASS clip -> SMPL body joint positions (forward kinematics). Shared by play_amass / retarget."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NJ = 22  # SMPL body joints; the remaining 30 of SMPL+H are fingers
PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
NAMES = ["pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee", "spine2", "l_ankle", "r_ankle", "spine3",
         "l_foot", "r_foot", "neck", "l_collar", "r_collar", "head", "l_shoulder", "r_shoulder",
         "l_elbow", "r_elbow", "l_wrist", "r_wrist"]

# ponytail: hand-typed approximate SMPL rest joints (y-up, metres), used only until the real
# SMPL+H model is in dataset/smpl/. Bone directions come from the clip, so this is roughly right;
# the real model adds the actor's exact proportions.
APPROX_REST = np.array([
    [0, 0, 0], [.06, -.09, 0], [-.06, -.09, 0], [0, .11, -.02], [.1, -.47, 0], [-.1, -.47, 0],
    [0, .24, 0], [.09, -.87, -.04], [-.09, -.87, -.04], [0, .29, .02], [.12, -.93, .1], [-.12, -.93, .1],
    [0, .5, -.02], [.08, .4, 0], [-.08, .4, 0], [0, .6, .03], [.18, .43, -.01], [-.18, .43, -.01],
    [.44, .4, -.03], [-.44, .4, -.03], [.7, .4, -.03], [-.7, .4, -.03]])


def axis_angle_to_mat(aa):
    """Rodrigues, vectorised over leading dims: (...,3) -> (...,3,3)."""
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = aa / np.maximum(th, 1e-8)
    K = np.zeros(aa.shape + (3,))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K = K - np.swapaxes(K, -1, -2)
    th = th[..., None]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def rest_joints(d):
    """Actor-specific rest joints from the SMPL+H model, or the approximate fallback."""
    path = ROOT / "dataset/smpl" / str(d["gender"]) / "model.npz"
    if not path.exists():
        print(f"warning: {path} missing, using approximate SMPL skeleton")
        return APPROX_REST
    smpl = np.load(path)
    nb = smpl["shapedirs"].shape[-1]
    v = smpl["v_template"] + smpl["shapedirs"] @ d["betas"][:nb]
    return (smpl["J_regressor"] @ v)[:NJ]


def load_clip(path):
    """-> (T, 22, 3) world joint positions (z-up, metres), fps."""
    d = np.load(path)
    J = rest_joints(d)
    R = axis_angle_to_mat(d["poses"][:, : NJ * 3].reshape(-1, NJ, 3))
    G = np.zeros_like(R)
    P = np.zeros(R.shape[:2] + (3,))
    G[:, 0], P[:, 0] = R[:, 0], J[0] + d["trans"]
    for j in range(1, NJ):
        p = PARENTS[j]
        G[:, j] = G[:, p] @ R[:, j]
        P[:, j] = P[:, p] + G[:, p] @ (J[j] - J[p])
    return P, float(d["mocap_framerate"])
