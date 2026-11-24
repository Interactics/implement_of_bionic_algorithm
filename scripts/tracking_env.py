"""Physics-based motion tracking env (MuJoCo humanoid), following ReActor's lower-level RL (Sec. 6).

Reference = kinematic retarget from scripts/retarget.py (outputs/retarget/<clip>.npz).
Follows the paper: 50 Hz PD joint setpoints + residual root wrench with deadband (Eq. 18-19),
proprioceptive obs (Eq. 20), reward terms and weights of Tab. 1, early termination on torso error.
Deviations (to keep CPU training feasible), all marked `ponytail:` below:
  - setpoints are residuals around the reference joint angles, not absolute
  - reference state initialization (RSI) instead of the learned init phase psi (we have a robot
    reference pose from IK, which the paper did not)
  - no bilevel retargeting-parameter optimization yet (fixed reference)
"""
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DT = 0.02  # 50 Hz control
ACT_SCALE = 0.5  # rad, residual setpoint range
F_MAX, TAU_MAX, DEADBAND = 400.0, 100.0, 0.1  # root wrench (N, Nm), deadband on normalized action
KP_PER_GEAR, KD_RATIO = 4.0, 0.1  # PD gains: kp = 4 * motor gear -> saturates at ~0.25 rad error
TERM_POS, TERM_ORI = 0.5, 1.0  # early termination: torso position (m) / orientation (rad) error

W = dict(root_xy=2.0, root_z=10.0, root_ori=2.0, root_v=0.5, root_w=0.5, body_x=2.0, body_R=2.0,
         alive=20.0, torque=1e-4, qacc=1e-6, act_rate=1e-2, act_acc=1e-2, wrench=1e-2)  # Tab. 1


def quat_angle(q1, q2):
    """Geodesic angle between unit quaternions (..., 4) -> (...)."""
    d = np.abs(np.sum(q1 * q2, axis=-1))
    return 2 * np.arccos(np.clip(d, -1, 1))


def heading_inv(quat):
    """Rotation matrix that removes the yaw of a (w,x,y,z) quaternion."""
    w, x, y, z = quat
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


class TrackingEnv(gym.Env):
    def __init__(self, ref_path, t0=0.0, t1=None, residual_force=True, seed=None):
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/humanoid.xml"))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.n_sub = round(DT / m.opt.timestep)
        jnt = m.actuator_trnid[:, 0]
        self.qadr, self.dadr = m.jnt_qposadr[jnt], m.jnt_dofadr[jnt]
        self.kp = KP_PER_GEAR * m.actuator_gear[:, 0]
        self.kd = KD_RATIO * self.kp
        self.torso = m.body("torso").id
        self.residual_force = residual_force
        self._load_reference(ref_path, t0, t1)
        nj = m.nu
        self.na = nj + (6 if residual_force else 0)
        self.action_space = gym.spaces.Box(-1, 1, (self.na,), np.float32)
        self.reset(seed=seed)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, self._obs().shape, np.float32)

    def _load_reference(self, path, t0, t1):
        """Resample the 120 fps IK reference to 50 Hz and precompute body poses / velocities."""
        m, d = self.model, self.data
        ref = np.load(path)
        Q, fps = ref["qpos"], float(ref["fps"])
        t_src = np.arange(len(Q)) / fps
        t1 = t_src[-1] if t1 is None else min(t1, t_src[-1])
        t = np.arange(t0, t1, DT)
        Qr = np.stack([np.interp(t, t_src, Q[:, i]) for i in range(m.nq)], 1)
        Qr[:, 3:7] /= np.linalg.norm(Qr[:, 3:7], axis=1, keepdims=True)  # nlerp for root quat
        V = np.zeros((len(t), m.nv))
        for i in range(len(t) - 1):
            mujoco.mj_differentiatePos(m, V[i], DT, Qr[i], Qr[i + 1])
        V[-1] = V[-2]
        X, R = np.zeros((len(t), m.nbody, 3)), np.zeros((len(t), m.nbody, 4))
        for i, q in enumerate(Qr):
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            X[i], R[i] = d.xpos, d.xquat
        self.ref_q, self.ref_v, self.ref_x, self.ref_R = Qr, V, X[:, 1:], R[:, 1:]  # drop world body
        self.T = len(t)

    # ---------------------------------------------------------------- gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        self.f = int(self.np_random.integers(0, max(1, self.T - 10)))  # RSI
        d.qpos[:] = self.ref_q[self.f]
        d.qpos[7:] += self.np_random.normal(0, 0.02, m.nq - 7)
        d.qvel[:] = self.ref_v[self.f]
        mujoco.mj_forward(m, d)
        self.a1 = np.zeros(self.na)
        self.a2 = np.zeros(self.na)
        return self._obs(), {}

    def step(self, action):
        m, d = self.model, self.data
        a = np.clip(action, -1, 1)
        nj = m.nu
        f_next = min(self.f + 1, self.T - 1)
        # ponytail: residual setpoints around the reference; absolute setpoints (paper) learn slower
        target = self.ref_q[f_next, self.qadr] + ACT_SCALE * a[:nj]
        if self.residual_force:
            w = np.sign(a[nj:]) * np.maximum(0, np.abs(a[nj:]) - DEADBAND)  # Eq. 19
            d.xfrc_applied[self.torso] = np.concatenate([w[:3] * F_MAX, w[3:] * TAU_MAX])
        torque = np.zeros(nj)
        self.qvel_prev = d.qvel[6:].copy()
        for _ in range(self.n_sub):  # PD at physics rate
            torque = self.kp * (target - d.qpos[self.qadr]) - self.kd * d.qvel[self.dadr]
            d.ctrl[:] = np.clip(torque / m.actuator_gear[:, 0], -1, 1)
            mujoco.mj_step(m, d)
        torque = d.qfrc_actuator[self.dadr].copy()  # applied (saturated) joint torque, for the penalty
        self.f = f_next

        r, info = self._reward(a, torque)
        term = info["torso_pos_err"] > TERM_POS or info["torso_ori_err"] > TERM_ORI
        trunc = self.f >= self.T - 1
        self.a2, self.a1 = self.a1, a
        return self._obs(), (0.0 if term else r), term, trunc, info

    # ------------------------------------------------------------- internals
    def _reward(self, a, torque):
        m, d, f = self.model, self.data, self.f
        x, R = d.xpos[1:], d.xquat[1:]
        rx, rR = self.ref_x[f], self.ref_R[f]
        root_ori = quat_angle(d.qpos[3:7], self.ref_q[f, 3:7])
        nj = m.nu
        da = (a[:nj] - self.a1[:nj]) * ACT_SCALE
        dda = (a[:nj] - 2 * self.a1[:nj] + self.a2[:nj]) * ACT_SCALE
        terms = dict(
            root_xy=-np.sum((d.qpos[:2] - self.ref_q[f, :2]) ** 2),
            root_z=-(d.qpos[2] - self.ref_q[f, 2]) ** 2,
            root_ori=-root_ori ** 2,
            root_v=-np.sum((d.qvel[:3] - self.ref_v[f, :3]) ** 2),
            root_w=-np.sum((d.qvel[3:6] - self.ref_v[f, 3:6]) ** 2),
            body_x=-np.sum((x - rx) ** 2),
            body_R=-np.sum(quat_angle(R, rR) ** 2),
            alive=1.0,
            torque=-np.sum(torque ** 2),
            qacc=-np.sum(((d.qvel[6:] - self.qvel_prev) / DT) ** 2),  # over the control step
            act_rate=-np.sum(da ** 2),
            act_acc=-np.sum(dda ** 2),
            wrench=-np.sum(np.abs(d.xfrc_applied[self.torso])) if self.residual_force else 0.0,
        )
        r = sum(W[k] * v for k, v in terms.items())
        self.last_terms = {k: W[k] * v for k, v in terms.items()}
        info = dict(torso_pos_err=np.linalg.norm(d.qpos[:3] - self.ref_q[f, :3]), torso_ori_err=root_ori)
        return r, info

    def _obs(self):
        m, d, f = self.model, self.data, min(self.f + 1, self.T - 1)
        Rt = d.xmat[self.torso].reshape(3, 3)
        H = heading_inv(d.qpos[3:7])
        root = d.qpos[:3]
        ref_Rt = np.zeros(9)
        mujoco.mju_quat2Mat(ref_Rt, self.ref_q[f, 3:7])
        return np.concatenate([
            # proprioception, Eq. 20 (psi omitted, see RSI note)
            [root[2]], Rt.T @ [0, 0, -1], Rt.T @ d.qvel[:3], d.qvel[3:6],
            d.qpos[self.qadr], d.qvel[self.dadr], self.a1, self.a2,
            # reference g_t (next frame), in the robot's heading frame
            [self.ref_q[f, 2]], ref_Rt.reshape(3, 3).T @ [0, 0, -1], self.ref_q[f, self.qadr],
            (H @ (self.ref_x[f] - root).T).T.ravel(), (H @ (d.xpos[1:] - root).T).T.ravel(),
            [self.f / self.T],
        ]).astype(np.float32)


if __name__ == "__main__":  # self-check: zero action = pure PD on the reference, should survive a bit
    env = TrackingEnv(ROOT / "outputs/retarget/05_02_poses.npz", 0, 4)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs), obs.shape
    total, n = 0.0, 0
    for n in range(1, 500):
        obs, r, term, trunc, info = env.step(np.zeros(env.na))
        total += r
        if term or trunc:
            break
    print(f"obs {obs.shape}, act {env.na}, ref frames {env.T}; zero-action episode: {n} steps, return {total:.0f}, "
          f"{'terminated' if term else 'truncated'} (pos err {info['torso_pos_err']:.2f} m)")
